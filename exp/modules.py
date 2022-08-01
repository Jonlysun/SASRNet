import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import re
from collections import OrderedDict
from warping import get_depthflow, getDepthEsti_forward, backwarp
try:
    from DCNv2.dcn_v2 import DCN_sep
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)

def grid_sample(feats, sample_grid, mode='bilinear', padding_mode='zeros'):
    return F.grid_sample(feats, sample_grid, mode=mode, padding_mode=padding_mode)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out

class ResBlockIP(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockIP, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x, y):
        x1 = x
        out = self.conv1(torch.cat([x, y], dim=1))
        out = self.relu(out)
        out = self.conv2(out)
        out = out + x1
        return out

class UNet(nn.Module):
    def __init__(self,
                 in_channels,
                 enc_channels=[64, 128, 256],
                 dec_channels=[128, 64],
                 out_channels=3,
                 n_enc_convs=2,
                 n_dec_convs=2):
        super(UNet, self).__init__()

        self.encs = nn.ModuleList()
        self.enc_translates = nn.ModuleList()
        pool = False
        for enc_channel in enc_channels:
            stage = self.create_stage(
                in_channels, enc_channel, n_enc_convs, pool
            )
            self.encs.append(stage)
            translate = nn.Conv2d(enc_channel, enc_channel, kernel_size=1)
            self.enc_translates.append(translate)
            in_channels, pool = enc_channel, True

        self.decs = nn.ModuleList()
        for idx, dec_channel in enumerate(dec_channels):
            in_channels = enc_channels[-idx - 1] + enc_channels[-idx - 2]
            stage = self.create_stage(
                in_channels, dec_channel, n_dec_convs, False
            )
            self.decs.append(stage)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        if out_channels <= 0:
            self.out_conv = None
        else:
            self.out_conv = nn.Conv2d(
                dec_channels[-1], out_channels, kernel_size=1, padding=0
            )

    def convrelu(self, in_channels, out_channels, kernel_size=3, padding=None):
        if padding is None:
            padding = (kernel_size - 1) // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )

    def create_stage(self, in_channels, out_channels, n_convs, pool):
        mods = []
        if pool:
            mods.append(nn.AvgPool2d(kernel_size=2))
        for _ in range(n_convs):
            mods.append(self.convrelu(in_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*mods)

    def forward(self, x):
        outs = []
        for enc, enc_translates in zip(self.encs, self.enc_translates):
            x = enc(x)
            outs.append(enc_translates(x))

        for dec in self.decs:
            x0, x1 = outs.pop(), outs.pop()
            x = torch.cat((self.upsample(x0), x1), dim=1)
            x = dec(x)
            outs.append(x)

        x = outs.pop()
        if self.out_conv:
            x = self.out_conv(x)
        return x

class SFE6(nn.Module):
    def __init__(self, input_channel, num_res_blocks, n_feats, res_scale):
        super(SFE6, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.conv_head = conv3x3(input_channel, n_feats)

        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RBs.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                     res_scale=res_scale))

        self.conv_tail = conv3x3(n_feats, n_feats)

    def forward(self, x):
        x = F.relu(self.conv_head(x))
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RBs[i](x)
        x = self.conv_tail(x)
        x = x + x1
        return x

class FE(nn.Module):
    '''
    extract 
    '''
    def __init__(self, nf, res_num):
        super().__init__()
        self.nf = nf
        self.res_num = res_num
        self.SFE = SFE6(4+27+1+1, self.res_num, self.nf * 2, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, src_dms, sr_sampling_maps, sr_valid_depth_masks, sr_valid_map_masks, dim):
        bs, nv, c, h, w = dim

        src_dms = torch.unsqueeze(src_dms, dim=2)
        # 2->0
        x_0 = grid_sample(F.unfold(x[:, 1].clone(), kernel_size=(3, 3), padding=1).view(bs, c * 9, h, w), sr_sampling_maps[:, 0, 0].clone())
        # 0->2
        x_2 = grid_sample(F.unfold(x[:, 0].clone(), kernel_size=(3, 3), padding=1).view(bs, c * 9, h, w), sr_sampling_maps[:, 1, 0].clone())

        x = torch.cat([x, src_dms], dim=2)
        x_0 = torch.cat([x[:, 0].clone(), x_0, sr_valid_depth_masks[:, 0, 0], sr_valid_map_masks[:, 0, 0]], dim=1)
        x_2 = torch.cat([x[:, 1].clone(), x_2, sr_valid_depth_masks[:, 1, 0], sr_valid_map_masks[:, 1, 0]], dim=1)

        x_0 = torch.unsqueeze(x_0, dim=1)
        x_2 = torch.unsqueeze(x_2, dim=1)
        x = torch.cat([x_0, x_2], dim=1)

        x = x.view(bs * nv, *x.shape[2:])
        x = self.SFE(x)
        x = x.view(bs, nv, self.nf * 2, h, w)

        x_feats = x

        return x_feats

class PCD_Align_IP(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''

    def __init__(self, nf=64, groups=8):
        super(PCD_Align_IP, self).__init__()
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # fea1
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1_1 = nn.Conv2d(
            nf * 2 + 1, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack_1 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                    deformable_groups=groups)
        # self.L3_result_1 = nn.Conv2d(nf * 2, nf, 1, 1, 0, bias=True)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1_1 = nn.Conv2d(
            nf * 2 + 1, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2_1 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack_1 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                    deformable_groups=groups)
        self.L2_fea_conv_1 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1_1 = nn.Conv2d(
            nf * 2 + 1, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2_1 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack_1 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                    deformable_groups=groups)
        self.L1_fea_conv_1 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, fea, vmm, dim, lea_offset=None):
        bs, c, h, w = dim
        '''align other neighboring frames to the reference frame in the feature level
        fea1, fea2: [L1, L2, L3], each with [B,C,H,W] features
        estimate offset bidirectionally
        '''
        fea = fea.contiguous().view(bs*2, c, h, w)
        fea2 = self.lrelu(self.fea_L2_conv1(fea))
        fea2 = self.lrelu(self.fea_L2_conv2(fea2))
        fea3 = self.lrelu(self.fea_L3_conv1(fea2))
        fea3 = self.lrelu(self.fea_L3_conv2(fea3))

        fea = fea.contiguous().view(bs, c * 2, h, w)
        fea2 = fea2.contiguous().view(bs, c * 2, h // 2, w // 2)
        fea3 = fea3.contiguous().view(bs, c * 2, h // 4, w // 4)

        vmm2 = F.interpolate(vmm, scale_factor=0.5,
                             mode='bilinear', align_corners=False)
        vmm3 = F.interpolate(vmm, scale_factor=0.25,
                             mode='bilinear', align_corners=False)

        # param. of fea1
        # L3
        L3_offset = torch.cat([fea3, vmm3], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1_1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2_1(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack_1(
            fea3[:, c:c*2, ...].contiguous(), L3_offset))
        # L2
        L2_offset = torch.cat([fea2, vmm2], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1_1(L2_offset))
        L3_offset_tmp = F.interpolate(
            L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2_1(
            torch.cat([L2_offset, L3_offset_tmp], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3_1(L2_offset))
        L2_fea = self.L2_dcnpack_1(fea2[:, c:c*2, ...].contiguous(), L2_offset)
        L3_fea = F.interpolate(L3_fea, scale_factor=2,
                               mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv_1(
            torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L1_offset = torch.cat([fea, vmm], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1_1(L1_offset))
        L2_offset_tmp = F.interpolate(
            L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2_1(
            torch.cat([L1_offset, L2_offset_tmp], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3_1(L1_offset))
        L1_fea = self.L1_dcnpack_1(fea[:, c:c*2, ...].contiguous(), L1_offset)
        L2_fea = F.interpolate(L2_fea, scale_factor=2,
                               mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv_1(torch.cat([L1_fea, L2_fea], dim=1))

        return L1_fea, F.interpolate(L1_offset, scale_factor=2, mode='bilinear', align_corners=False)

class MRSR(nn.Module):
    def __init__(self, nf, res_num, vs_help_sr, sr_help_vs):
        super().__init__()
        self.nf = nf
        self.num_res_blocks = res_num
        self.vs_help_sr = vs_help_sr
        self.sr_help_vs = sr_help_vs
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.RB22_ip_merge = nn.ModuleList()
        for j in range(self.num_res_blocks[2]):
            if j == 0:
                self.RB22_ip_merge.append(ResBlockIP(
                    in_channels=self.nf * 2 + 2 + 1 + self.nf, out_channels=self.nf))
            else:
                self.RB22_ip_merge.append(ResBlockIP(
                    in_channels=self.nf * 3, out_channels=self.nf))
        self.RB22_inter = ResBlockIP(
            in_channels=self.nf * 3, out_channels=self.nf)
        self.conv22_ip_merge_tail = conv3x3(self.nf, self.nf)
        self.conv22_ip_mask = conv3x3(self.nf * 3, 3)

        self.RB33_ip_merge = nn.ModuleList()
        for j in range(self.num_res_blocks[3]):
            if j == 0:
                self.RB33_ip_merge.append(ResBlockIP(
                    in_channels=self.nf // 2 * 2 + 2 + 1 + self.nf // 2, out_channels=self.nf // 2))
            else:
                self.RB33_ip_merge.append(ResBlockIP(
                    in_channels=self.nf // 2 * 3, out_channels=self.nf // 2))
        self.RB33_inter = ResBlockIP(in_channels=self.nf // 2 * 3, out_channels=self.nf // 2)
        self.conv33_ip_merge_tail = conv3x3(self.nf // 2, self.nf // 2)
        self.conv33_ip_mask = conv3x3(self.nf // 2 * 3, 3)

        if self.vs_help_sr:
            self.conv11_head_sr = nn.Sequential(
                conv3x3(self.nf * 4 + self.nf * 2 + 3, self.nf * 4),
                nn.LeakyReLU(0.1, inplace=True),
                conv3x3(self.nf * 4, self.nf * 2),
                nn.LeakyReLU(0.1, inplace=True),
                conv3x3(self.nf * 2, self.nf * 2),
                # conv3x3(self.nf * 6 + 4, self.nf * 2),
            )
            self.conv22_head_sr = nn.Sequential(
                conv3x3(self.nf * 2 + self.nf + 3, self.nf*2),
                nn.LeakyReLU(0.1, inplace=True),
                conv3x3(self.nf*2, self.nf),
                nn.LeakyReLU(0.1, inplace=True),
                conv3x3(self.nf, self.nf),
                # conv3x3(self.nf * 3 + 4, self.nf),
            )
            self.conv33_head_sr = nn.Sequential(
                conv3x3(self.nf + self.nf // 2 + 3, self.nf),
                nn.LeakyReLU(0.1, inplace=True),
                conv3x3(self.nf, self.nf // 2),
                nn.LeakyReLU(0.1, inplace=True),
                conv3x3(self.nf // 2, self.nf // 2),
                # conv3x3(self.nf + self.nf // 2 + 4, self.nf),
            )
        else:
            self.conv11_head_sr = nn.Sequential(
                conv3x3(self.nf * 4 + 2, self.nf * 4),
                nn.LeakyReLU(0.1, inplace=True),
                conv3x3(self.nf * 4, self.nf * 2),
                nn.LeakyReLU(0.1, inplace=True),
                conv3x3(self.nf * 2, self.nf * 2),
                # conv3x3(self.nf * 6 + 4, self.nf * 2),
            )
            self.conv22_head_sr = nn.Sequential(
                conv3x3(self.nf * 2 + 2, self.nf*2),
                nn.LeakyReLU(0.1, inplace=True),
                conv3x3(self.nf*2, self.nf),
                nn.LeakyReLU(0.1, inplace=True),
                conv3x3(self.nf, self.nf),
                # conv3x3(self.nf * 3 + 4, self.nf),
            )
            self.conv33_head_sr = nn.Sequential(
                conv3x3(self.nf + 2, self.nf),
                nn.LeakyReLU(0.1, inplace=True),
                conv3x3(self.nf, self.nf // 2),
                nn.LeakyReLU(0.1, inplace=True),
                conv3x3(self.nf // 2, self.nf // 2),
                # conv3x3(self.nf + self.nf // 2 + 4, self.nf),
            )

        self.RB = nn.ModuleList()
        self.conv_tail = nn.ModuleList()
        self.conv_up = nn.ModuleList()
        self.ps = nn.ModuleList()
        # 0, 1, 2
        for i in range(3):
            scale_factor = 2 ** i
            self.RB_tmp = nn.ModuleList()
            for j in range(self.num_res_blocks[1 + i]):
                self.RB_tmp.append(ResBlock(in_channels=self.nf * 2 // scale_factor, out_channels=self.nf * 2 // scale_factor,
                                            res_scale=1))
            self.RB.append(self.RB_tmp)
            self.conv_tail.append(
                conv3x3(self.nf * 2 // scale_factor, self.nf * 2 // scale_factor))
            if i != 2:
                self.conv_up.append(
                    conv3x3(self.nf * 2 // scale_factor, self.nf * 4 // scale_factor))
                self.ps.append(nn.PixelShuffle(2))

        self.merge_tail = conv3x3(self.nf // 2, 3)

        self.lr_pred = nn.Sequential(
            conv3x3(self.nf * 2 + 1, self.nf * 2),
            nn.LeakyReLU(0.1, inplace=True),
            conv3x3(self.nf * 2, self.nf),
            nn.LeakyReLU(0.1, inplace=True),
            conv3x3(self.nf, 3),
            # conv3x3(self.nf * 6 + 4, self.nf * 2),
        )

    def forward(self, x_lv, 
            x_inter_pred_refine, 
            src_dms,
            inter_dm_pred,
            inter1_mask,
            depth_flow_pred_forward,
            valid_depth_masks_pred_forward,
            src_Ks, 
            inter_tgt_K,
            patch_pixel_coords, 
            pose_trans_matrixs_src2tgt,
            sr_sampling_maps, 
            sr_valid_depth_masks, 
            sr_valid_map_masks, 
            dim):

        bs, nv, c, h, w = dim

        x_feat_1_0 = x_lv[:, 0, ...].clone()
        x_feat_1_2 = x_lv[:, 1, ...].clone()

        depth_flow_pred_backward = None
        valid_depth_masks_pred_backward = None

        if self.vs_help_sr:
            depth_flow_pred_1_0, valid_depth_masks_pred_1_0 = get_depthflow(src_dms[:, 0], src_Ks[:, 0], inter_tgt_K.unsqueeze(
                dim=1), pose_trans_matrixs_src2tgt[:, 0:1], patch_pixel_coords=patch_pixel_coords)
            depth_flow_pred_2_0, valid_depth_masks_pred_2_0 = get_depthflow(src_dms[:, 1], src_Ks[:, 1], inter_tgt_K.unsqueeze(
                dim=1), pose_trans_matrixs_src2tgt[:, 1:2], patch_pixel_coords=patch_pixel_coords)
            depth_flow_pred_backward = torch.cat(
                [depth_flow_pred_1_0, depth_flow_pred_2_0], dim=1)
            valid_depth_masks_pred_backward = torch.cat(
                [valid_depth_masks_pred_1_0, valid_depth_masks_pred_2_0], dim=1)
 
        # dms = dms.view(bs * (nv + 1), 1, h, w)
        for scale in range(3):
            scale_factor = 2 ** scale

            ip_vdm = valid_depth_masks_pred_forward.clone().view(bs * (nv-1) * 2, 1, h, w)
            ip_vdm = F.interpolate(ip_vdm, scale_factor=scale_factor, mode='bilinear').view(
                bs, (nv-1), 2, 1, h * scale_factor, w * scale_factor)
    
            ip_smm = depth_flow_pred_forward.clone().view(
                bs * (nv-1) * 2, h, w, 2).permute(0, 3, 1, 2)
            ip_smm = F.interpolate(ip_smm, [
                                h * scale_factor, w * scale_factor], mode='bilinear', align_corners=False) * scale_factor
            

            sr_vdm = sr_valid_depth_masks.clone().view(bs * nv, 1, h, w)
            sr_vdm = F.interpolate(sr_vdm, scale_factor=scale_factor, mode='bilinear').view(
                bs, (nv), 1, 1, h * scale_factor, w * scale_factor)
            sr_vmm = sr_valid_map_masks.clone().view(bs * nv, 1, h, w)
            sr_vmm = F.interpolate(sr_vmm, scale_factor=scale_factor, mode='bilinear').view(
                bs, (nv), 1, 1, h * scale_factor, w * scale_factor)
            sr_smm = sr_sampling_maps.clone().view(bs * (nv), h, w, 2).permute(0, 3, 1, 2)
            sr_smm = F.interpolate(sr_smm, scale_factor=scale_factor, mode='bilinear').permute(0, 2, 3, 1).view(bs, (nv),
                                                                                                                1, h * scale_factor,
                                                                                                                w * scale_factor, 2)

            if self.vs_help_sr:
                back_vdm = valid_depth_masks_pred_backward.clone().view(bs * nv, 1, h, w)
                back_vdm = F.interpolate(back_vdm, scale_factor=scale_factor, mode='bilinear').view(bs, nv, 1, 1, h * scale_factor, w * scale_factor)

                back_smm = depth_flow_pred_backward.clone().view(bs * (nv-1) * 2, h, w, 2).permute(0, 3, 1, 2)
                back_smm = F.interpolate(back_smm, [h * scale_factor, w * scale_factor], mode='bilinear', align_corners=False) * scale_factor
                back_smm = back_smm.view(bs, nv, *back_smm.shape[1:])

            if self.sr_help_vs and scale != 0:
                x_lv_tmp = x_lv.view(bs * nv, *x_lv.shape[2:])
                x_w = backwarp(x_lv_tmp.clone(), ip_smm.clone())
                x_w = x_w.view(bs, nv, *x_w.shape[1:])

                x_front = x_w[:, 0]
                x_back = x_w[:, 1]

            if scale == 0:
                inter_feature = torch.cat([x_inter_pred_refine, inter_dm_pred], dim=1)
                inter_lr_pred = self.lr_pred(inter_feature)

                # source views to source views
                x_feat_1_0_res_2 = grid_sample(x_feat_1_2.clone(), sr_sampling_maps[:, 0, 0].clone())
                x_feat_1_2_res_0 = grid_sample(x_feat_1_0.clone(), sr_sampling_maps[:, 1, 0].clone())
            
                if self.vs_help_sr:
                    x_feat_1_0_res_1 = backwarp(x_inter_pred_refine.clone(), back_smm[:, 0].clone())
                    x_feat_1_2_res_1 = backwarp(x_inter_pred_refine.clone(), back_smm[:, 1].clone())

                    x_feat_1_0_res = torch.cat(
                        [x_feat_1_0, x_feat_1_0_res_2, x_feat_1_0_res_1, back_vdm[:, 0, 0], sr_valid_depth_masks[:, 0, 0], sr_valid_map_masks[:, 0, 0]], dim=1)
                    x_feat_1_2_res = torch.cat(
                        [x_feat_1_2, x_feat_1_2_res_0, x_feat_1_2_res_1, back_vdm[:, 1, 0], sr_valid_depth_masks[:, 1, 0], sr_valid_map_masks[:, 1, 0]], dim=1)
                else:
                    x_feat_1_0_res = torch.cat(
                        [x_feat_1_0, x_feat_1_0_res_2, sr_valid_depth_masks[:, 0, 0], sr_valid_map_masks[:, 0, 0]], dim=1)
                    x_feat_1_2_res = torch.cat(
                        [x_feat_1_2, x_feat_1_2_res_0, sr_valid_depth_masks[:, 1, 0], sr_valid_map_masks[:, 1, 0]], dim=1)
            
                x_feat_1_0_res = self.conv11_head_sr(x_feat_1_0_res)
                x_feat_1_0 = x_feat_1_0 + x_feat_1_0_res

                x_feat_1_2_res = self.conv11_head_sr(x_feat_1_2_res)
                x_feat_1_2 = x_feat_1_2 + x_feat_1_2_res

                x_feat_1_0 = torch.unsqueeze(x_feat_1_0, dim=1)
                x_inter_pred_refine = torch.unsqueeze(x_inter_pred_refine, dim=1)
                x_feat_1_2 = torch.unsqueeze(x_feat_1_2, dim=1)
                x_feat_1 = torch.cat([x_feat_1_0, x_inter_pred_refine, x_feat_1_2], dim=1).view(bs*3, c*2, h, w)
                x11_res = x_feat_1

                for i in range(self.num_res_blocks[1]):
                    x11_res = self.RB[0][i](x11_res)
                x11_res = self.conv_tail[0](x11_res)
                x_feat_1 = x_feat_1 + x11_res
                x_feat_1 = self.conv_up[0](x_feat_1)
                x_feat_1 = self.lrelu(self.ps[0](x_feat_1))
                x_feat_1 = x_feat_1.view(bs, 3, c, h * 2, w * 2)

                x_lv = torch.cat([x_feat_1[:, 0:1], x_feat_1[:, 2:3]], dim=1)

            if scale == 1:
                x_feat_2_1 = x_feat_1[:, 1].clone()
                x_front_res = torch.cat(
                    [x_back, x_feat_2_1, inter1_mask[:, 0:1, ...], ip_vdm[:, 0, 0], ip_vdm[:, 0, 1]], dim=1)
                x_back_res = torch.cat(
                    [x_front, x_feat_2_1, inter1_mask[:, 1:2, ...], ip_vdm[:, 0, 1], ip_vdm[:, 0, 0]], dim=1)
                x_feat_2_1_res = torch.cat([x_front, x_back], dim=1)

                for i in range(self.num_res_blocks[2]):
                    if i == 0:
                        x_feat_2_1_res = self.RB22_inter(x_feat_2_1,  x_feat_2_1_res)
                        x_front_res = self.RB22_ip_merge[i](x_front, x_front_res)
                        x_back_res = self.RB22_ip_merge[i](x_back, x_back_res)
                    else:
                        x_feat_2_1_res_tmp = self.RB22_ip_merge[i](x_feat_2_1_res, torch.cat([x_front_res, x_back_res], dim=1))
                        x_front_res_tmp = self.RB22_ip_merge[i](x_front_res, torch.cat([x_back_res, x_feat_2_1_res], dim=1))
                        x_back_res = self.RB22_ip_merge[i](x_back_res, torch.cat([x_feat_2_1_res, x_front_res], dim=1))
                        x_front_res = x_front_res_tmp
                        x_feat_2_1_res = x_feat_2_1_res_tmp
                x_front_res = self.conv22_ip_merge_tail(x_front_res)
                x_back_res = self.conv22_ip_merge_tail(x_back_res)
                x_feat_2_1_res = self.conv22_ip_merge_tail(x_feat_2_1_res)
                x_front2 = x_front + x_front_res
                x_back2 = x_back + x_back_res
                x_feat_2_1 = x_feat_2_1 + x_feat_2_1_res
                inter2_mask = self.conv22_ip_mask(torch.cat([x_front2, x_back2, x_feat_2_1], dim=1))
                alphas = torch.stack([inter2_mask[:, 0:1, ...], inter2_mask[:, 1:2, ...], inter2_mask[:, 2:3, ...]])
                alphas = torch.softmax(alphas, dim=0)
                rgbs = torch.stack([x_front2, x_back2, x_feat_2_1])
                x_inter2 = (alphas * rgbs).sum(dim=0)
                inter2_mask = F.interpolate(inter2_mask, scale_factor=2, mode='bilinear')
                x_feat_2_1 = x_inter2

                x_feat_2_0 = x_feat_1[:, 0].clone()
                x_feat_2_2 = x_feat_1[:, 2].clone()

                x_feat_2_0_res_2 = grid_sample(x_feat_2_2.clone(), sr_smm[:, 0, 0].clone())
                x_feat_2_2_res_0 = grid_sample(x_feat_2_0.clone(), sr_smm[:, 1, 0].clone())

                if self.vs_help_sr:
                    x_feat_2_0_res_1 = backwarp(x_feat_2_1.clone(), back_smm[:, 0].clone())
                    x_feat_2_2_res_1 = backwarp(x_feat_2_1.clone(), back_smm[:, 1].clone())

                    x_feat_2_0_res = torch.cat(
                        [x_feat_2_0, x_feat_2_0_res_2, x_feat_2_0_res_1, back_vdm[:, 0, 0], sr_vdm[:, 0, 0], sr_vmm[:, 0, 0]], dim=1)
                    x_feat_2_2_res = torch.cat(
                        [x_feat_2_2, x_feat_2_2_res_0, x_feat_2_2_res_1, back_vdm[:, 1, 0], sr_vdm[:, 1, 0], sr_vmm[:, 1, 0]], dim=1)
                else:
                    x_feat_2_0_res = torch.cat(
                        [x_feat_2_0, x_feat_2_0_res_2, sr_vdm[:, 0, 0], sr_vmm[:, 0, 0]], dim=1)
                    x_feat_2_2_res = torch.cat(
                        [x_feat_2_2, x_feat_2_2_res_0, sr_vdm[:, 1, 0], sr_vmm[:, 1, 0]], dim=1)

                x_feat_2_0_res = self.conv22_head_sr(x_feat_2_0_res)
                x_feat_2_0 = x_feat_2_0 + x_feat_2_0_res

                x_feat_2_2_res = self.conv22_head_sr(x_feat_2_2_res)
                x_feat_2_2 = x_feat_2_2 + x_feat_2_2_res

                x_feat_2_0 = torch.unsqueeze(x_feat_2_0, dim=1)
                x_feat_2_1 = torch.unsqueeze(x_feat_2_1, dim=1)
                x_feat_2_2 = torch.unsqueeze(x_feat_2_2, dim=1)
                x_feat_2 = torch.cat([x_feat_2_0, x_feat_2_1, x_feat_2_2], dim=1).view(
                    bs * 3, c, h * 2, w * 2)

                x22_res = x_feat_2
                for i in range(self.num_res_blocks[2]):
                    x22_res = self.RB[1][i](x22_res)
                x22_res = self.conv_tail[1](x22_res)
                x_feat_2 = x_feat_2 + x22_res

                x_feat_2 = self.conv_up[1](x_feat_2)
                x_feat_2 = self.lrelu(self.ps[1](x_feat_2))
                x_feat_2 = x_feat_2.view(bs, 3, c // 2, h * 4, w * 4)

                x_lv = torch.cat([x_feat_2[:, 0:1], x_feat_2[:, 2:3]], dim=1)

            if scale == 2:
                x_feat_3_1 = x_feat_2[:, 1].clone()
                x_front_res = torch.cat([x_back, x_feat_3_1, inter2_mask[:, 0:1, ...], ip_vdm[:, 0, 0], ip_vdm[:, 0, 1]],
                                        dim=1)
                x_back_res = torch.cat([x_front, x_feat_3_1, inter2_mask[:, 1:2, ...], ip_vdm[:, 0, 1], ip_vdm[:, 0, 0]],
                                       dim=1)
                x_feat_3_1_res = torch.cat([x_front, x_back], dim=1)

                for i in range(self.num_res_blocks[3]):
                    if i == 0:
                        x_feat_3_1_res = self.RB33_inter(
                            x_feat_3_1, x_feat_3_1_res)
                        x_front_res = self.RB33_ip_merge[i](
                            x_front, x_front_res)
                        x_back_res = self.RB33_ip_merge[i](x_back, x_back_res)
                    else:
                        x_feat_3_1_res_tmp = self.RB33_ip_merge[i](x_feat_3_1_res,
                                                                   torch.cat([x_front_res, x_back_res], dim=1))
                        x_front_res_tmp = self.RB33_ip_merge[i](x_front_res,
                                                                torch.cat([x_back_res, x_feat_3_1_res], dim=1))
                        x_back_res = self.RB33_ip_merge[i](
                            x_back_res, torch.cat([x_feat_3_1_res, x_front_res], dim=1))
                        x_front_res = x_front_res_tmp
                        x_feat_3_1_res = x_feat_3_1_res_tmp
                x_front_res = self.conv33_ip_merge_tail(x_front_res)
                x_back_res = self.conv33_ip_merge_tail(x_back_res)
                x_feat_3_1_res = self.conv33_ip_merge_tail(x_back_res)
                x_front3 = x_front + x_front_res
                x_back3 = x_back + x_back_res
                x_feat_3_1 = x_feat_3_1 + x_feat_3_1_res
                inter3_mask = self.conv33_ip_mask(
                    torch.cat([x_front3, x_back3, x_feat_3_1], dim=1))
                alphas = torch.stack(
                    [inter3_mask[:, 0:1, ...], inter3_mask[:, 1:2, ...], inter3_mask[:, 2:3, ...]])
                alphas = torch.softmax(alphas, dim=0)
                rgbs = torch.stack([x_front3, x_back3, x_feat_3_1])
                x_inter3 = (alphas * rgbs).sum(dim=0)
                x_feat_3_1 = x_inter3

                x_feat_3_0 = x_feat_2[:, 0].clone()
                x_feat_3_2 = x_feat_2[:, 2].clone()

                x_feat_3_0_res_2 = grid_sample(x_feat_3_2.clone(), sr_smm[:, 0, 0].clone())
                x_feat_3_2_res_0 = grid_sample(x_feat_3_0.clone(), sr_smm[:, 1, 0].clone())

                if self.vs_help_sr:
                    x_feat_3_0_res_1 = backwarp(x_feat_3_1.clone(), back_smm[:, 0].clone())
                    x_feat_3_2_res_1 = backwarp(x_feat_3_1.clone(), back_smm[:, 1].clone())

                    x_feat_3_0_res = torch.cat(
                        [x_feat_3_0, x_feat_3_0_res_2, x_feat_3_0_res_1, back_vdm[:, 0, 0], sr_vdm[:, 0, 0], sr_vmm[:, 0, 0]], dim=1)
                    x_feat_3_2_res = torch.cat(
                        [x_feat_3_2, x_feat_3_2_res_0, x_feat_3_2_res_1, back_vdm[:, 1, 0], sr_vdm[:, 1, 0], sr_vmm[:, 1, 0]], dim=1)
                else:
                    x_feat_3_0_res = torch.cat(
                        [x_feat_3_0, x_feat_3_0_res_2, sr_vdm[:, 0, 0], sr_vmm[:, 0, 0]], dim=1)
                    x_feat_3_2_res = torch.cat(
                        [x_feat_3_2, x_feat_3_2_res_0, sr_vdm[:, 1, 0], sr_vmm[:, 1, 0]], dim=1)

                x_feat_3_0_res = self.conv33_head_sr(x_feat_3_0_res)
                x_feat_3_0 = x_feat_3_0 + x_feat_3_0_res

                x_feat_3_2_res = self.conv33_head_sr(x_feat_3_2_res)
                x_feat_3_2 = x_feat_3_2 + x_feat_3_2_res

                x_feat_3_0 = torch.unsqueeze(x_feat_3_0, dim=1)
                x_feat_3_1 = torch.unsqueeze(x_feat_3_1, dim=1)
                x_feat_3_2 = torch.unsqueeze(x_feat_3_2, dim=1)
                x_feat_3 = torch.cat([x_feat_3_0, x_feat_3_1, x_feat_3_2], dim=1).view(
                    bs * 3, c // 2, h * 4, w * 4)

                x33_res = x_feat_3
                for i in range(self.num_res_blocks[3]):
                    x33_res = self.RB[2][i](x33_res)
                x33_res = self.conv_tail[2](x33_res)
                x_feat_3 = x_feat_3 + x33_res

                x_feat_3 = self.merge_tail(x_feat_3)
                # x_feat_3_vmm = x_feat_3.view(bs, 3, 3, h * 4, w * 4)

        return x_feat_3, inter_lr_pred

class DEM(nn.Module):
    '''
    This module aims to map source view features into virtual view and blends them to generate a coarse target feature
    '''
    def __init__(self, nf, blocks, depth_train_only=False):
        super().__init__()

        self.nf = nf
        self.blocks = blocks
        self.depth_train_only = depth_train_only
        
        self.Depth_esti = UNet(in_channels=12, 
                            enc_channels=[64, 128, 256], 
                            dec_channels=[128, 64], 
                            out_channels=1, 
                            n_enc_convs=3, 
                            n_dec_convs=3)

        if not depth_train_only:
            state_dict = torch.load(config.pretrained_dp_esti)
            new_state_dict = OrderedDict()

            for k, v in state_dict.items():
                new_k = re.split(r'[.]', k, 1)[-1]
                new_state_dict[new_k] = v

            self.Depth_esti.load_state_dict(new_state_dict)
            for p in self.Depth_esti.parameters():
                p.requires_grad = False
            
            self.RB11_ip_merge = nn.ModuleList()
            for j in range(self.blocks):
                if j == 0:
                    self.RB11_ip_merge.append(ResBlockIP(
                        in_channels=self.nf * 2 * 2 + 2, out_channels=self.nf * 2))
                else:
                    self.RB11_ip_merge.append(ResBlockIP(
                        in_channels=self.nf * 2 * 2, out_channels=self.nf * 2))
            self.conv11_ip_merge_tail = conv3x3(self.nf * 2, self.nf * 2)
            self.conv11_ip_mask = conv3x3(self.nf * 2 * 2, 2)

    def forward(self, img, src_dms, inter_tgt_K, src_Ks, pose_src2tgt, pose_tgt2src, patch_pixel_coords, x_feats=None):
        bs, nv, _, h, w = img.shape

        # forward warping
        dm_estis, map_masks = getDepthEsti_forward(src_dms, 
                                            inter_tgt_K, 
                                            src_Ks, 
                                            pose_src2tgt, 
                                            patch_pixel_coords=patch_pixel_coords)

        src_dms = src_dms.unsqueeze(dim=2)
        src_info = torch.cat([img[:, 0], img[:, 1], src_dms[:, 0], src_dms[:, 1]], dim=1)
        dm_info = torch.cat([dm_estis[:, 0], dm_estis[:, 1], map_masks[:, 0], map_masks[:, 1]], dim=1)
        dm_info = torch.cat([src_info, dm_info], dim=1)
        inter_dm_pred = self.Depth_esti(dm_info)
        inter_dm_pred = inter_dm_pred.squeeze(dim=1)

        if self.depth_train_only:
            return inter_dm_pred
        else:
            # warp source views to target view 
            depth_flow_pred_forward, valid_depth_masks_pred_forward = get_depthflow(
                 inter_dm_pred, inter_tgt_K, src_Ks, pose_tgt2src, patch_pixel_coords=patch_pixel_coords)
            ip_smm = depth_flow_pred_forward.view(bs * (nv-1) * 2, h, w, 2).permute(0, 3, 1, 2)
            ip_vdm = valid_depth_masks_pred_forward.view(bs, (nv-1), 2, 1, h, w)
    
            x_lv_tmp = x_feats.view(bs * nv, *x_feats.shape[2:])            
            # inverse warping
            x_inter_pred = backwarp(x_lv_tmp.clone(), ip_smm.clone())
            x_inter_pred = x_inter_pred.view(bs, nv, *x_inter_pred.shape[1:])

            x_front = x_inter_pred[:, 0]
            x_back = x_inter_pred[:, 1]

            x_front_res = torch.cat([x_back, ip_vdm[:, 0, 0], ip_vdm[:, 0, 1]], dim=1)
            x_back_res = torch.cat([x_front, ip_vdm[:, 0, 1], ip_vdm[:, 0, 0]], dim=1)

            for i in range(self.blocks):
                if i == 0:
                    x_front_res = self.RB11_ip_merge[i](
                        x_front, x_front_res)
                    x_back_res = self.RB11_ip_merge[i](x_back, x_back_res)
                else:
                    x_front_res_tmp = self.RB11_ip_merge[i](
                        x_front_res, x_back_res)
                    x_back_res = self.RB11_ip_merge[i](
                        x_back_res, x_front_res)
                    x_front_res = x_front_res_tmp
            x_front_res = self.conv11_ip_merge_tail(x_front_res)
            x_back_res = self.conv11_ip_merge_tail(x_back_res)
            x_front1 = x_front + x_front_res
            x_back1 = x_back + x_back_res
            inter1_mask = self.conv11_ip_mask(torch.cat([x_front1, x_back1], dim=1))
            alphas = torch.stack([inter1_mask[:, 0:1, ...], inter1_mask[:, 1:2, ...]])
            alphas = torch.softmax(alphas, dim=0)
            rgbs = torch.stack([x_front1, x_back1])
            x_inter1 = (alphas * rgbs).sum(dim=0)
            inter1_mask = F.interpolate(inter1_mask, scale_factor=2, mode='bilinear')
            x_inter_pred = x_inter1

            return inter_dm_pred, x_inter_pred, inter1_mask, depth_flow_pred_forward, valid_depth_masks_pred_forward

class FR(nn.Module):
    def __init__(self, nf, ):
        super().__init__()
        self.nf = nf
        self.pcd_align_front = PCD_Align_IP(nf=self.nf*2, groups=8)

        self.merge_background_conv_lv = nn.Sequential(
            nn.Conv2d(nf * 2 * 3 + 1, nf * 2, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf * 2, nf * 2, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf * 2, nf * 2, kernel_size=3,
                      stride=1, padding=1, bias=True),
        )


    def forward(self, x_inter, x_feats, valid_depth_masks_pred_forward):
        bs, nv, c, h, w= x_feats.shape

        x_feat_1_0 = x_feats[:, 0, ...].clone()
        x_feat_1_2 = x_feats[:, 1, ...].clone()

        ip_vdm = valid_depth_masks_pred_forward.view(bs, (nv-1), 2, 1, h, w)

        vmm = (1-ip_vdm[:, 0, 0]) * (1-ip_vdm[:, 0, 1])

        x_inter_complement_front = torch.cat([x_inter, x_feat_1_0], dim=1)
        x_inter_complement_back = torch.cat([x_inter, x_feat_1_2], dim=1)
        lea_offset_front, lea_offset_back = None, None
        x_inter_complement_front, lea_offset_front = self.pcd_align_front(x_inter_complement_front, vmm, [bs, self.nf*2, h, w])
        x_inter_complement_back, lea_offset_back = self.pcd_align_front(x_inter_complement_back, vmm, [bs, self.nf*2, h, w])
        x_inter_complement = self.merge_background_conv_lv(torch.cat(
            [x_inter_complement_front, x_inter_complement_back, x_inter, vmm], dim=1))
        x_inter_refine = x_inter + x_inter_complement * vmm
        
        return x_inter_refine

class SASRNet(nn.Module):
    def __init__(self, nf=64):
        super().__init__()
        self.nf = nf
        self.depth_train_only = config.depth_train_only
        self.feature_refine = config.feature_refine
        self.vs_help_sr = config.vs_help_sr
        self.sr_help_vs = config.sr_help_vs
        self.num_res_blocks = list(map(int, config.num_res_blocks.split('+')))

        if self.depth_train_only:
            self.depth_estimation_mapping = DEM(self.nf, self.num_res_blocks[1], depth_train_only=True)
        
        else:
            self.feature_extraction = FE(self.nf, self.num_res_blocks[0])
            
            self.depth_estimation_mapping = DEM(self.nf, self.num_res_blocks[1], depth_train_only=False)

            self.feature_refinement = FR(self.nf)

            self.mutal_referenced_SR = MRSR(self.nf, self.num_res_blocks, self.vs_help_sr, self.sr_help_vs)

            self.print_parameter_use()

    def print_parameter_use(self):

        total_params = sum(p.numel() for p in self.depth_estimation_mapping.parameters())
        print(f'{total_params:,} self.depth_estimation_mapping total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.depth_estimation_mapping.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} self.depth_estimation_mapping training parameters.')

        total_params = sum(p.numel() for p in self.feature_extraction.parameters())
        print(f'{total_params:,} self.feature_extraction  total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.feature_extraction.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} self.feature_extraction training parameters.')
    
        total_params = sum(p.numel() for p in self.feature_extraction.parameters())
        print(f'{total_params:,} self.feature_extraction  total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.feature_extraction.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} self.feature_extraction training parameters.')

        total_params = sum(p.numel() for p in self.mutal_referenced_SR.parameters())
        print(f'{total_params:,} self.mutal_referenced_SR total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.mutal_referenced_SR.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} self.mutal_referenced_SR training parameters.')

    def forward(self, **kwargs):
        """
        tgts torch.Size([1, 3, 3, 400, 400])                                                                                                   
        tgt_lrs torch.Size([1, 2, 3, 100, 100])                                                                                                
        src_dms torch.Size([1, 2, 100, 100])                                                                                                   
        inter_lr torch.Size([1, 3, 100, 100])                                                                                                  
        inter_dm torch.Size([1, 100, 100])                                                                                                                                                                                       
        sr_sampling_maps torch.Size([1, 2, 1, 100, 100, 2])                                                                                   
        sr_valid_depth_masks torch.Size([1, 2, 1, 1, 100, 100])                                                       
        sr_valid_map_masks torch.Size([1, 2, 1, 1, 100, 100])
        """
        img = kwargs["src_lrs"]
        src_dms = kwargs["src_dms"]

        sr_sampling_maps = kwargs["sr_sampling_maps"]
        sr_valid_depth_masks = kwargs["sr_valid_depth_masks"]
        sr_valid_map_masks = kwargs["sr_valid_map_masks"]

        patch_pixel_coords = kwargs["patch_pixel_coords"]
        pose_trans_matrixs_tgt2src = kwargs["pose_trans_matrixs_tgt2src"]
        pose_trans_matrixs_src2tgt = kwargs["pose_trans_matrixs_src2tgt"]
        inter_tgt_K = kwargs["inter_tgt_K"]
        src_Ks = kwargs["src_Ks"]
        bs, nv, c, h, w = img.shape

        if self.depth_train_only:
            inter_dm_pred = self.depth_estimation_mapping(img, 
                                                    src_dms, 
                                                    inter_tgt_K, 
                                                    src_Ks, 
                                                    pose_trans_matrixs_src2tgt, 
                                                    pose_trans_matrixs_tgt2src, 
                                                    patch_pixel_coords)
            return {'inter_dm_pred': inter_dm_pred}
        else:
            x_feats = self.feature_extraction(img, src_dms, sr_sampling_maps, sr_valid_depth_masks, sr_valid_map_masks, [bs, nv, c, h, w])

            with torch.no_grad():
                inter_dm_pred, x_inter_pred, inter1_mask, depth_flow_pred_forward, valid_depth_masks_pred_forward = self.depth_estimation_mapping(img, 
                                                                                    src_dms, 
                                                                                    inter_tgt_K, 
                                                                                    src_Ks, 
                                                                                    pose_trans_matrixs_src2tgt, 
                                                                                    pose_trans_matrixs_tgt2src, 
                                                                                    patch_pixel_coords, 
                                                                                    x_feats=x_feats)

            if self.feature_refine:
                x_inter_pred = self.feature_refinement(x_inter_pred, x_feats, valid_depth_masks_pred_forward)

            x, inter_lr_pred = self.mutal_referenced_SR.forward(x_feats, 
                                                            x_inter_pred, 
                                                            src_dms,
                                                            inter_dm_pred.unsqueeze(dim=1),
                                                            inter1_mask,
                                                            depth_flow_pred_forward,
                                                            valid_depth_masks_pred_forward,                                                            
                                                            src_Ks, 
                                                            inter_tgt_K,
                                                            patch_pixel_coords,
                                                            pose_trans_matrixs_src2tgt, 
                                                            sr_sampling_maps, 
                                                            sr_valid_depth_masks, 
                                                            sr_valid_map_masks, 
                                                            [bs, nv, self.nf, h, w])

            x = torch.clamp(x, -1, 1)

            return {"out": x, 'inter_lr_pred': inter_lr_pred, 'inter_dm_pred': inter_dm_pred}

def get_net():
    model = SASRNet(nf=config.nf)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    return model
