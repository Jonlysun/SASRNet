from __future__ import division
import torch
import sys
import numpy as np
import torch.nn.functional as F

sys.path.append('../')
from Forward_Warping_min.Forward_Warp import forward_warp


pixel_coords = None
def getTrans(mat_1, mat_2):
    mat_tmp = np.identity(4)
    mat_tmp[:3, :] = mat_1
    mat_1 = mat_tmp
    mat_tmp = np.identity(4)
    mat_tmp[:3, :] = mat_2
    mat_2 = mat_tmp
    del mat_tmp
    
    pose = np.reshape(np.matmul(mat_1, np.linalg.inv(mat_2)), [
                    4, 4]).astype(np.float32)
    return pose[:3, :]


backwarp_tenGrid = {}
def backwarp(tenInput, tenFlow):
    if str(tenFlow.size()) not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(
            tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(
            tenFlow.shape[0], -1, -1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.size())] = torch.cat([tenHorizontal, tenVertical], 1).type_as(tenFlow)
    # end

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    return F.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.size())] + tenFlow).permute(0, 2, 3, 1),
                         mode='bilinear', padding_mode='zeros')

def pose_mat2vec(Rt):
    Rt = Rt[:, :3, :]
    t_all = Rt[:, :, -1]
    R_all = Rt[:, :, :3]
    sy = R_all[:, 0, 0]*R_all[:, 0, 0] + R_all[:, 1, 0]*R_all[:, 1, 0]
    y_all = torch.atan2(-R_all[:, 2, 0], sy)
    theta = []
    for (R, t, s, y) in zip(R_all, t_all, sy, y_all):
        singular = s < 1e-6
        if not singular:
            x = torch.atan2(R[2, 1], R[2, 2])
            z = torch.atan2(R[1, 0], R[0, 0])
        else:
            x = torch.atan2(-R[1, 2], R[1, 1])
            z = 0
        theta.append(torch.stack([x, y, z]))

    theta = torch.stack(theta)
    pose_all = torch.cat([theta, t_all], dim=1)
    return pose_all

def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(1,h,w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1,h,w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1,h,w).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]

def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i,size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(input_name, 'x'.join(expected), list(input.size()))


def pixel2cam(depth, intrinsics_inv, patch_pixel_coords):
    global pixel_coords
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, h, w = depth.size()
    if patch_pixel_coords is not None:
        ones = torch.ones(b, h, w, 1).type_as(patch_pixel_coords)
        pixel_coords = torch.cat([patch_pixel_coords, ones], dim=-1).permute(0, 3, 1, 2)
    elif (pixel_coords is None) or pixel_coords.size(2) < h:
        set_id_grid(depth)
    current_pixel_coords = pixel_coords[:,:,:h,:w].expand(b,3,h,w).reshape(b, 3, -1)  # [B, 3, H*W]
    cam_coords = (intrinsics_inv @ current_pixel_coords).reshape(b, 3, h, w)
    return cam_coords * depth.unsqueeze(1)


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode, patch=None):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]

    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(1e-3)

    patch = patch.type(torch.cuda.FloatTensor)
    if patch is not None:
        X_norm = 2*((X / Z) - patch[:, 2])/(w-1) - 1
        Y_norm = 2*((Y / Z) - patch[:, 0])/(h-1) - 1  # Idem [B, H*W]
    else:
        # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
        X_norm = 2*(X / Z)/(w-1) - 1
        Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]

    Z = Z.reshape(b, h, w)
    valid_mask = torch.zeros_like(Z)
    valid_mask[Z > 1e-1] = 1

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.reshape(b, h, w, 2), valid_mask.unsqueeze(dim=1)

def cam2pixel2flow(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode, patch_pixel_coords):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]

    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(1e-3)

    X_norm = (X / Z)
    Y_norm = (Y / Z)

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    pixel_coords = pixel_coords.reshape(b, h, w, 2)
    flows = pixel_coords - patch_pixel_coords

    return flows


def euler2mat(angle):
    """Convert euler angles to rotation matrix.

     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:, :1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:,
                                            1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.

    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:, 3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat

def inverse_warp(img, depth, pose1, intrinsics, rotation_mode='euler', padding_mode='zeros', patch=None, size=None):
    """
    Inverse warp a source image to the target image plane.

    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_points: Boolean array indicating point validity
    """
    # check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'BHW')
    check_sizes(pose1, 'pose1', 'B34')
    # check_sizes(pose2, 'pose2', 'B34')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = img.size()

    cam_coords = pixel2cam(depth, intrinsics.inverse())  # [B,3,H,W]
    pose_mat = pose1  # [B,3,4]
    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]
    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    src_pixel_coords = cam2pixel(
        cam_coords, rot, tr, padding_mode)  # [B,H,W,2]
    # src_pixel_coords = is_valid_proj(depth, src_pixel_coords)

    img = F.grid_sample(img, src_pixel_coords.type(
        torch.cuda.FloatTensor), padding_mode=padding_mode)
    return img

def get_depthflow(tgt_dm, tgt_K, src_Ks, pose_trans_matrixs_tgt2src, patch_pixel_coords=None):
    '''
    tgt_dm: [B, h, w]
    tgt_K: [B, 3, 3]
    src_Ks: [B, nv, 3, 3]
    pose_trans_matrixs_src2tgt: [B, nv, 3, 4]
    patch_pixel_coords：[B, h, w, 2]
    '''
    B, h, w = tgt_dm.shape
    nv = src_Ks.shape[1]
    tgt_dm = tgt_dm.expand(B * nv, h, w)
    tgt_K = tgt_K.unsqueeze(dim=1)
    tgt_K = tgt_K.expand(B, nv, 3, 3)
    src_Ks = src_Ks.view(B * nv, 3, 3)
    tgt_K = tgt_K.contiguous().view(B * nv, 3, 3)

    pose_trans_matrixs_tgt2src = pose_trans_matrixs_tgt2src.view(B * nv, 3, 4)
    patch_pixel_coords = patch_pixel_coords.unsqueeze(dim=1)
    patch_pixel_coords = patch_pixel_coords.expand(B, nv, h, w, 2)
    patch_pixel_coords = patch_pixel_coords.contiguous().view(B * nv, h, w, 2)

    cam_coords = pixel2cam(tgt_dm, tgt_K.inverse(), patch_pixel_coords)
    proj_cam_to_src_pixel = src_Ks @ pose_trans_matrixs_tgt2src
    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    # src_pixel_coords = cam2pixel(cam_coords, rot, tr, padding_mode='zeros', patch=patch)
    src_flow = cam2pixel2flow(cam_coords, rot, tr, padding_mode='zeros', patch_pixel_coords=patch_pixel_coords)

    tgt_dm = tgt_dm.unsqueeze(dim=1)
    mask = torch.ones_like(tgt_dm)
    src_flow_tmp = src_flow.permute(0, 3, 1, 2)
    mask_estis = backwarp(mask, src_flow_tmp)
    src_flow = src_flow.view(B, nv, *src_flow.shape[1:])
    mask_estis = mask_estis.view(B, nv, *mask_estis.shape[1:])

    return src_flow, mask_estis


def getDepthEsti_forward(tgt_dm, tgt_K, src_Ks, pose_trans_matrixs_src2tgt, patch_pixel_coords=None):
    '''
    dm: [B, nv, h, w]
    tgt_K: [B, 3, 3]
    src_Ks: [B, nv, 3, 3]
    pose_trans_matrixs_src2tgt: [B, nv, 3, 4]
    patch_pixel_coords：[B, h, w, 2]
    '''
    B, nv, h, w = tgt_dm.shape
    tgt_dm = tgt_dm.view(B * nv, h, w)
    tgt_K = tgt_K.expand(B, nv, 3, 3)
    tgt_K = tgt_K.contiguous().view(B * nv, 3, 3)

    src_Ks = src_Ks.view(B * nv, 3, 3)
    pose_trans_matrixs_src2tgt = pose_trans_matrixs_src2tgt.view(B * nv, 3, 4)
    patch_pixel_coords = patch_pixel_coords.expand(B, nv, h, w, 2)
    patch_pixel_coords = patch_pixel_coords.contiguous().view(B * nv, h, w, 2)

    fw = forward_warp(interpolation_mode='Nearest')
    cam_coords = pixel2cam(tgt_dm, src_Ks.inverse(), patch_pixel_coords)
    proj_cam_to_src_pixel = tgt_K @ pose_trans_matrixs_src2tgt
    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    # src_pixel_coords = cam2pixel(cam_coords, rot, tr, padding_mode='zeros', patch=patch)
    src_flow = cam2pixel2flow(cam_coords, rot, tr, padding_mode='zeros', patch_pixel_coords=patch_pixel_coords)

    tgt_dm = tgt_dm.unsqueeze(dim=1)
    mask = torch.ones_like(tgt_dm)
    dm_estis = fw(tgt_dm, src_flow)
    mask_estis = fw(mask, src_flow)
    dm_estis = dm_estis.view(B, nv, *dm_estis.shape[1:])
    mask_estis = mask_estis.view(B, nv, *mask_estis.shape[1:])

    return dm_estis, mask_estis
