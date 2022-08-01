from random import randrange
import os
import math
import random
import torch
import torch.nn.functional as F
import numpy as np
import PIL
import logging
import sys
from pathlib import Path
import cv2

sys.path.append("../")
import ext
import co
import co.utils as co_utils
import config

meshGrids = {}

class Dataset(co.mytorch.BaseDataset):
    def __init__(
        self,
        *,
        name,
        tgt_im_paths,
        tgt_dm_paths,
        tgt_Ks,
        tgt_Rs,
        tgt_ts,
        tgt_counts,
        src_im_paths,
        src_dm_paths,
        src_Ks,
        src_Rs,
        src_ts,
        im_size=None,
        pad_width=None,
        patch=None,
        n_nbs=5,
        nbs_mode="sample",
        bwd_depth_thresh=0.1,
        invalid_depth_to_inf=True,
        ycbcr=False,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.tgt_im_paths = tgt_im_paths
        self.tgt_dm_paths = tgt_dm_paths
        self.tgt_Ks = tgt_Ks
        self.tgt_Rs = tgt_Rs
        self.tgt_ts = tgt_ts
        self.tgt_counts = tgt_counts

        self.src_im_paths = src_im_paths
        self.src_dm_paths = src_dm_paths
        self.src_Ks = src_Ks
        self.src_Rs = src_Rs
        self.src_ts = src_ts

        self.im_size = im_size
        self.pad_width = pad_width
        self.patch = patch
        self.n_nbs = n_nbs
        self.nbs_start = self.n_nbs // 2
        self.nbs_end = self.nbs_start + 1
        self.nbs_mode = nbs_mode
        self.bwd_depth_thresh = bwd_depth_thresh
        self.invalid_depth_to_inf = invalid_depth_to_inf
        self.ycbcr = ycbcr
        self.data_augment = config.data_augment
        self.len = len(self.tgt_dm_paths)-1
        self.single = config.single
        tmp = np.load(tgt_dm_paths[0])
        self.height, self.width = tmp.shape
        del tmp

        n_tgt_im_paths = len(tgt_im_paths) if tgt_im_paths else 0
        shape_tgt_im = (
            self.load_pad(tgt_im_paths[0]).shape if tgt_im_paths else None
        )
        logging.info(
            f"    #tgt_im_paths={n_tgt_im_paths}, # tgt_im={shape_tgt_im}, tgt_dm={self.load_pad(tgt_dm_paths[0]).shape}"
        )

        self.count = 0
        self.net = config.net

    def depth2flow(self, flow, patch):
        """
        Args:
            flow ([type]): [n_view, H, W, 2]
            patch ([type]): [h_from, h_to, w_from, w_to]

        Returns:
            flow [type]: [n_view, H, W, 2]
        """
        n_view = flow.shape[0]
        h_from, h_to, w_from, w_to = patch
        patch_height = h_to - h_from
        patch_width = w_to - w_from

        valid_mask = np.ones_like(flow)
        valid_mask[flow > 1] = 0
        valid_mask[flow < -1] = 0
        flow = np.concatenate(((flow[:, :, :, 0:1] + 1) / 2 * (patch_width - 1) + w_from,
                               (flow[:, :, :, 1:2] + 1) / 2 * (patch_height - 1) + h_from), axis=-1)

        h_grid = np.arange(h_from, h_to).reshape(1, patch_height, 1, 1).repeat(
            patch_width, axis=2).repeat(n_view, axis=0)
        w_grid = np.arange(w_from, w_to).reshape(1, 1, patch_width, 1).repeat(
            patch_height, axis=1).repeat(n_view, axis=0)
        meshGrid = np.concatenate((w_grid, h_grid), axis=-1)
        new_flow = flow - meshGrid
        new_flow = new_flow * valid_mask
        # valid_flow = new_flow * valid_mask
        return new_flow.astype(np.float32)

    def load_data(self, p, rr=None, lr=None, ud=None):
        if p.suffix == ".npy":
            npy = np.load(p)
            if self.train == True and self.data_augment == True:
                if rr is None:
                    rr = np.random.randint(0, 4)
                if lr is None:
                    lr = np.random.randint(0, 2)
                if ud is None:
                    ud = np.random.randint(0, 2)
                npy = np.rot90(npy, rr).copy()
                if lr == 1:
                    npy = np.fliplr(npy).copy()
                if ud == 1:
                    npy = np.flipud(npy).copy()
            return npy
        elif p.suffix in [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"]:
            im = PIL.Image.open(p)
            im = np.array(im)
            if self.ycbcr == True:
                im = co_utils.rgb2ycbcr(im, only_y=True)
                im = im.astype(np.float32) / 255
                im = np.expand_dims(im, axis=0)
                return im

            if self.train == True and self.data_augment == True:
                if rr is None:
                    rr = np.random.randint(0, 4)
                if lr is None:
                    lr = np.random.randint(0, 2)
                if ud is None:
                    ud = np.random.randint(0, 2)
                im = np.rot90(im, rr).copy()
                if lr == 1:
                    im = np.fliplr(im).copy()
                if ud == 1:
                    im = np.flipud(im).copy()

            im = (im.astype(np.float32) / 255) * 2 - 1
            im = im.transpose(2, 0, 1)
            return im
        else:
            raise Exception("invalid suffix")

    def pad(self, im, hr=False, h=0, w=0):
        if self.im_size is not None or h != 0:
            shape = [s for s in im.shape]
            if hr == False:
                shape[-2] = self.im_size[0]
                shape[-1] = self.im_size[1]
            elif hr == True:
                shape[-2] = h
                shape[-1] = w
            im_p = np.zeros(shape, dtype=im.dtype)
            sh = min(im_p.shape[-2], im.shape[-2])
            sw = min(im_p.shape[-1], im.shape[-1])
            im_p[..., :sh, :sw] = im[..., :sh, :sw]
            im = im_p
        if self.pad_width is not None:
            h, w = im.shape[-2:]
            mh = h % self.pad_width
            ph = 0 if mh == 0 else self.pad_width - mh
            mw = w % self.pad_width
            pw = 0 if mw == 0 else self.pad_width - mw
            shape = [s for s in im.shape]
            shape[-2] += ph
            shape[-1] += pw
            im_p = np.zeros(shape, dtype=im.dtype)
            im_p[..., :h, :w] = im
            im = im_p
        return im

    def load_pad(self, p, hr=False, h=0, w=0, rr=None, lr=None, ud=None):
        im = self.load_data(p, rr, lr, ud)
        return self.pad(im, hr=hr, h=h, w=w)

    def base_len(self):
        return len(self.tgt_dm_paths)
    
    def getTrans(self, mat_1, mat_2):
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

    
    def depth2flow(self, flow, patch):
        """
        Args:
            flow ([type]): [n_view, H, W, 2]
            patch ([type]): [h_from, h_to, w_from, w_to]

        Returns:
            flow [type]: [n_view, H, W, 2]
        """
        n_view = flow.shape[0]
        h_from, h_to, w_from, w_to = patch
        patch_height = h_to - h_from
        patch_width = w_to - w_from

        valid_mask = np.ones_like(flow)
        valid_mask[flow > 1] = 0
        valid_mask[flow < -1] = 0

        flow = np.concatenate(((flow[:, :, :, 0:1] + 1) / 2 * (patch_width - 1) + w_from,
                               (flow[:, :, :, 1:2] + 1) / 2 * (patch_height - 1) + h_from), axis=-1)
        h_grid = np.arange(h_from, h_to).reshape(1, patch_height, 1, 1).repeat(
            patch_width, axis=2).repeat(n_view, axis=0)
        w_grid = np.arange(w_from, w_to).reshape(1, 1, patch_width, 1).repeat(
            patch_height, axis=1).repeat(n_view, axis=0)
        meshGrid = np.concatenate((w_grid, h_grid), axis=-1)
        new_flow = flow - meshGrid
        # valid_flow = new_flow * valid_mask
        new_flow = new_flow * valid_mask
        return new_flow.astype(np.float32)

    def getPatchPixelCoords(self, h, w):
        if str((h, w)) not in meshGrids:
            h_grid = np.linspace(0, h-1, h, endpoint=True).reshape((h, 1, 1)).repeat(w, axis=1)
            w_grid = np.linspace(0, w-1, w, endpoint=True).reshape((1, w, 1)).repeat(h, axis=0)
            meshGrid = np.concatenate([w_grid, h_grid], axis=-1)
            meshGrids[str((h, w))] = meshGrid

        return meshGrids[str((h, w))]


    def base_getitem(self, idx, rng):

        nbs = [abs(i)+2 * min(self.len - i, 0)
               for i in range(idx-self.nbs_start, idx+self.nbs_end, 1)]
        ret = {}
        rr, lr, ud = None, None, None
        if self.data_augment == True:
            rr = np.random.randint(0, 4)
            lr = np.random.randint(0, 2)
            ud = np.random.randint(0, 2)

        dm = self.load_data(self.tgt_dm_paths[idx], rr=rr, lr=lr, ud=ud)
        if self.train == False:
            ret["HR_size"] = np.array(
                (dm.shape[0]*4, dm.shape[1]*4), dtype=np.int32)
        dm = self.pad(dm, hr=False)
        y = dm.shape[0]
        x = dm.shape[1]
        del dm

        if self.patch:
            patch_h_from = rng.randint(0, y - self.patch[0])
            patch_w_from = rng.randint(0, x - self.patch[1])
            patch_h_to = patch_h_from + self.patch[0]
            patch_w_to = patch_w_from + self.patch[1]
            patch = np.array(
                (patch_h_from, patch_h_to, patch_w_from, patch_w_to),
                dtype=np.int32,
            )
        else:
            patch = np.array(
                (0, y, 0, x), dtype=np.int32
            )

        pixel_coords = self.getPatchPixelCoords(y, x).astype(np.float32)
        ret["patch_pixel_coords"] = pixel_coords[patch[0]:patch[1], patch[2]:patch[3], :]
        
        dms = []
        src_Ks = []
        src_Rs = []
        src_ts = []
        tgts = []
        src_lrs = []
        src_dms = []
        for i, val in enumerate(nbs):
            dms.append(self.load_pad(
                self.tgt_dm_paths[val], hr=False, rr=rr, lr=lr, ud=ud))
            src_Ks.append(self.tgt_Ks[val])
            src_Rs.append(self.tgt_Rs[val])
            src_ts.append(self.tgt_ts[val])

            tgt = self.load_pad(Path(str(self.tgt_im_paths[val]).replace('LR', 'HR')), hr=True,
                                h=y * config.scale, w=x * config.scale, rr=rr, lr=lr, ud=ud)
            tgts.append(tgt[
                :,
                patch[0] * config.scale: patch[1] * config.scale,
                patch[2] * config.scale:  patch[3] * config.scale,
            ])

            if i % 2 == 0:
                tgt_lr = self.load_pad(
                    self.tgt_im_paths[val], hr=False, rr=rr, lr=lr, ud=ud)
                src_lrs.append(tgt_lr[
                    :,
                    patch[0]: patch[1],
                    patch[2]: patch[3],
                ])
                tgt_dm = self.load_pad(
                    self.tgt_dm_paths[val], hr=False, rr=rr, lr=lr, ud=ud)
                src_dms.append(
                    tgt_dm[patch[0]: patch[1], patch[2]: patch[3], ])
            else:
                inter_lr = self.load_pad(
                    self.tgt_im_paths[val], hr=False, rr=rr, lr=lr, ud=ud)
                inter_lr = inter_lr[:, patch[0]: patch[1], patch[2]:patch[3]]
                inter_dm = self.load_pad(
                    self.tgt_dm_paths[val], hr=False, rr=rr, lr=lr, ud=ud)
                inter_dm = inter_dm[patch[0]: patch[1], patch[2]:patch[3]]

        # 0, 2
        ret["tgts"] = np.array(tgts)
        ret["src_lrs"] = np.array(src_lrs)
        ret["src_dms"] = np.array(src_dms)

        # 1
        ret["inter_lr"] = np.array(inter_lr)
        ret["inter_dm"] = np.array(inter_dm)

        sr_sampling_maps_l = []
        sr_depthflow_l = []
        sr_valid_depth_masks_l = []
        sr_valid_map_masks_l = []
        for i in range(0, self.n_nbs, 2):
            sampling_maps, valid_depth_masks, valid_map_masks = ext.preprocess.get_sampling_map(
                dms[i],
                src_Ks[i],
                src_Rs[i],
                src_ts[i],
                [dms[abs(i - 2)]],
                [src_Ks[abs(i - 2)]],
                [src_Rs[abs(i - 2)]],
                [src_ts[abs(i - 2)]],
                patch,  # patch,
                self.bwd_depth_thresh,
                self.invalid_depth_to_inf,
            )
            sr_sampling_maps_l.append(sampling_maps)
            sr_depthflow_l.append(self.depth2flow(sampling_maps, patch))
            sr_valid_map_masks_l.append(valid_map_masks)
            sr_valid_depth_masks_l.append(valid_depth_masks)

        ret["sr_sampling_maps"] = np.array(sr_sampling_maps_l)
        ret["sr_depthflow"] = np.array(sr_depthflow_l)
        ret["sr_valid_depth_masks"] = np.array(sr_valid_depth_masks_l)
        ret["sr_valid_map_masks"] = np.array(sr_valid_map_masks_l)

        inter_tgt_K = src_Ks[1]
        src_Ks = np.concatenate([src_Ks[0:1], src_Ks[2:]], axis=0)
        ret["inter_tgt_K"] = inter_tgt_K.astype(np.float32)
        ret["src_Ks"] = src_Ks.astype(np.float32)

        tgt_R = src_Rs[1]
        tgt_t = src_ts[1]
        tgt_extrinsic = np.concatenate([tgt_R, tgt_t.reshape((3, 1))], axis=1)
        pose_trans_matrixs_tgt2src = []
        pose_trans_matrixs_src2tgt = []
        for i in range(0, 3, 2):
            src_R = src_Rs[i]
            src_t = src_ts[i]
            src_extrinsic = np.concatenate([src_R, src_t.reshape((3, 1))], axis=1)
            matrix_tgt2src = self.getTrans(src_extrinsic, tgt_extrinsic)
            matrix_src2tgt = self.getTrans(tgt_extrinsic, src_extrinsic)
            pose_trans_matrixs_tgt2src.append(matrix_tgt2src)
            pose_trans_matrixs_src2tgt.append(matrix_src2tgt)
        pose_trans_matrixs_tgt2src = np.array(pose_trans_matrixs_tgt2src)
        pose_trans_matrixs_src2tgt = np.array(pose_trans_matrixs_src2tgt)
        ret["pose_trans_matrixs_tgt2src"] = pose_trans_matrixs_tgt2src
        ret["pose_trans_matrixs_src2tgt"] = pose_trans_matrixs_src2tgt
        return ret


class Dataset_Vimeo(co.mytorch.BaseDataset):
    def __init__(
        self,
        name,
        image_dir,
        file_list,
        patch=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.image_dir = image_dir
        self.file_list = file_list
        alist = [line.rstrip() for line in open(
            os.path.join(image_dir, file_list))]
        self.image_filenames = [os.path.join(image_dir, x) for x in alist]

        self.patch = patch
        self.data_augment = config.data_augment

    def base_len(self):
        return len(self.image_filenames)

    def modcrop(self, img, modulo):
        (ih, iw) = img.size
        ih = ih - (ih % modulo)
        iw = iw - (iw % modulo)
        img = img.crop((0, 0, ih, iw))

        return img

    def load_img(self, filepath, scale, n_nbs):
        list = os.listdir(filepath)
        list.sort()

        target = [self.modcrop(PIL.Image.open(filepath + '/' + list[i]).convert('RGB'), scale) for i in
                  range(0, n_nbs, 1)]

        h, w = target[0].size
        h_in, w_in = int(h // scale), int(w // scale)

        rr = np.random.randint(0, 4)
        lr = np.random.randint(0, 2)
        ud = np.random.randint(0, 2)

        target_l = []
        input = []
        for j in range(0, len(target)):
            img = target[j].resize((h_in, w_in), PIL.Image.BICUBIC)
            img = np.array(img)
            img = np.rot90(img, rr).copy()
            if lr == 1:
                img = np.fliplr(img).copy()
            if ud == 1:
                img = np.flipud(img).copy()
            img = (img.astype(np.float32) / 255) * 2 - 1
            img = img.transpose(2, 0, 1)

            target_l.append(img)
            if j % 2 == 0:
                input.append(input)

        return input, target_l

    def base_getitem(self, idx, rng):

        input, target = self.load_img(
            self.image_filenames[idx], self.upscale_factor, self.n_nbs)

        input = np.array(input)
        target = np.array(target)

        y, x = input.shape[2], input.shape[3]
        if self.patch:
            patch_h_from = rng.randint(0, y - self.patch[0])
            patch_w_from = rng.randint(0, x - self.patch[1])
            patch_h_to = patch_h_from + self.patch[0]
            patch_w_to = patch_w_from + self.patch[1]

        else:
            patch_h_from = 0
            patch_w_from = 0
            patch_h_to = y
            patch_w_to = x

        input = input[:, :, patch_h_from:patch_h_to, patch_w_from:patch_w_to]
        target = target[:, :, patch_h_from:patch_h_to, patch_w_from:patch_w_to]

        ret = {}
        ret['input'] = input
        ret['target'] = target

        return ret
