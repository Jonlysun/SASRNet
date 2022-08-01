import os
import argparse
from PIL import Image
import cv2
import math
import numpy as np
import shutil as shutil

def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def bgr2ycbcr(img, only_y=True):
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    # print(img1)
    # print('img1-2')
    # print(img2)
    mse = np.mean((img1 - img2)**2)
    # print(mse)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def get_result(path, save_path, file_txt, dataset_name, scenes, gt_path, gt_lr_path, epoch, save_every):
        
    total_str_rgb = ''
    total_str_y = ''
    total_str_rgb_sr = ''
    total_str_y_sr = ''
    total_str_rgb_ip = ''
    total_str_y_ip = ''

    file_txt.write('#' * 80 + '\n')
    file_txt.write(path + '\n')

    total_count = 0.0
    total_psnr_rgb = 0.0
    total_ssim_rgb = 0.0
    total_psnr_y = 0.0
    total_ssim_y = 0.0
    total_count_sr = 0.0
    total_psnr_rgb_sr = 0.0
    total_ssim_rgb_sr = 0.0
    total_psnr_y_sr = 0.0
    total_ssim_y_sr = 0.0
    total_count_ip = 0.0
    total_psnr_rgb_ip = 0.0
    total_ssim_rgb_ip = 0.0
    total_psnr_y_ip = 0.0
    total_ssim_y_ip = 0.0

    str_rgb = 'SASRNet'
    str_y = 'SASRNet'
    str_rgb_sr = 'SASRNet'
    str_y_sr = 'SASRNet'
    str_rgb_ip = 'SASRNet'
    str_y_ip = 'SASRNet'

    for dataset_count, scene in enumerate(scenes):

        tmp_total_count = 0.0
        tmp_total_psnr_rgb = 0.0
        tmp_total_ssim_rgb = 0.0
        tmp_total_psnr_y = 0.0
        tmp_total_ssim_y = 0.0
        tmp_total_count_sr = 0.0
        tmp_total_psnr_rgb_sr = 0.0
        tmp_total_ssim_rgb_sr = 0.0
        tmp_total_psnr_y_sr = 0.0
        tmp_total_ssim_y_sr = 0.0
        tmp_total_count_ip = 0.0
        tmp_total_psnr_rgb_ip = 0.0
        tmp_total_ssim_rgb_ip = 0.0
        tmp_total_psnr_y_ip = 0.0
        tmp_total_ssim_y_ip = 0.0

        sub_savepath = os.path.join(save_path, epoch, scene)
        if not os.path.exists(sub_savepath):
            os.makedirs(sub_savepath)
        image_l = []
        image_lr = []

        if dataset_name == 'Tanks_and_Temples':
            if scene == 'Truck':
                subdir = os.path.join(path, f'tat_all_training_{scene}_0.5_n3', epoch)
            else:
                subdir = os.path.join(path, f'tat_all_intermediate_{scene}_0.5_n3', epoch)
        elif dataset_name == 'ETH':
            subdir = os.path.join(path, f'tat_all_{scene}_0.5_n3', epoch)

        subsubdirs = os.listdir(subdir)
        subsubdirs.sort()

        for subsubdir in subsubdirs:
            dirs = os.path.join(subdir, subsubdir)
            files = sorted(os.listdir(dirs))
            lr_path = files[2]
            lr_path = os.path.join(dirs, lr_path)
            image_lr.append(lr_path)

        for root, dirs, files in os.walk(os.path.join(subdir, subsubdirs[1])):
            # print(root) #当前目录路径
            # print(dirs) #当前路径下所有子目录
            # print(files) #当前路径下所有非目录子文件
            if len(files) >= 2:
                files.sort()
                image_path = os.path.join(root, files[0])
                image_l.append(image_path)
                image_path = os.path.join(root, files[1])
                image_l.append(image_path)

        for i in range(2, len(subsubdirs)):
            if i % 2 == 0:
                for root, dirs, files in os.walk(os.path.join(subdir, subsubdirs[i-1])):
                    # print(root) #当前目录路径
                    # print(dirs) #当前路径下所有子目录
                    # print(files) #当前路径下所有非目录子文件
                    if len(files) >= 2:
                        files.sort()
                        # image_path = os.path.join(root, files[2])
                        image_path = os.path.join(root, files[-1])
                        image_l.append(image_path)
            else:
                for root, dirs, files in os.walk(os.path.join(subdir, subsubdirs[i])):
                    # print(root) #当前目录路径
                    # print(dirs) #当前路径下所有子目录
                    # print(files) #当前路径下所有非目录子文件
                    if len(files) >= 2:
                        files.sort()
                        image_path = os.path.join(root, files[1])
                        image_l.append(image_path)

        
        print("HR pred images number:", len(image_l))
        print("LR pred images number: ", len(image_lr))

        assert len(image_lr) == len(image_l)

        gt_l = []
        for root, dirs, files in os.walk(gt_path):
            # print(root) #当前目录路径
            # print(dirs) #当前路径下所有子目录
            # print(files) #当前路径下所有非目录子文件
            if len(files) >= 10:
                files.sort()
                for file in files:
                    if os.path.splitext(file)[1] == '.png' and 'ibr3d_pw_0.50' in root and scene in root:
                        image_path = os.path.join(root, file)
                        gt_l.append(image_path)
        
        gt_lr = []
        for root, dirs, files in os.walk(gt_lr_path):
            # print(root) #当前目录路径
            # print(dirs) #当前路径下所有子目录
            # print(files) #当前路径下所有非目录子文件
            if len(files) >= 10:
                files.sort()
                for file in files:
                    if os.path.splitext(file)[1] == '.png' and 'ibr3d_pw_0.50' in root and scene in root:
                        image_path = os.path.join(root, file)
                        gt_lr.append(image_path)

        print("HR gt images number:", len(gt_l))
        print("LR gt images number:", len(gt_lr))

        assert len(gt_l) == len(gt_lr) == len(image_lr)
        
        for i, gt_image_path in enumerate(gt_l):
            gt_image = Image.open(gt_image_path)
            sr_image = Image.open(image_l[i])
            copy_path = sub_savepath + '/' + str(i).zfill(4) + '.png'
            shutil.copy(image_l[i], copy_path)

            gt_image = np.array(gt_image)
            sr_image = np.array(sr_image)
            # print('gt_image', gt_image.shape)
            # print('sr_image', sr_image.shape)
            psnr_rgb = calculate_psnr(sr_image, gt_image)
            ssim_rgb = calculate_ssim(sr_image, gt_image)
            gt_image = rgb2ycbcr(gt_image, only_y=True)
            sr_image = rgb2ycbcr(sr_image, only_y=True)
            psnr_y = calculate_psnr(sr_image, gt_image)
            ssim_y = calculate_ssim(sr_image, gt_image)
        
            tmp_total_count = tmp_total_count + 1
            tmp_total_psnr_rgb = tmp_total_psnr_rgb + psnr_rgb
            tmp_total_ssim_rgb = tmp_total_ssim_rgb + ssim_rgb
            tmp_total_psnr_y = tmp_total_psnr_y + psnr_y
            tmp_total_ssim_y = tmp_total_ssim_y + ssim_y

            
            if save_every == True:
                file_txt.write(image_l[i] + '\n')
                file_txt.write('psnr_rgb:' + str(psnr_rgb) + '\nssim_rgb:' + str(ssim_rgb) + '\npsnr_y:'+ str(psnr_y) + '\nssim_y:' + str(ssim_y) + '\n')
            print(image_l[i])
            print('psnr_rgb:', psnr_rgb)
            print('ssim_rgb:', ssim_rgb)
            print('psnr_y:', psnr_y)
            print('ssim_y:', ssim_y)

            if i % 2 == 0:
                tmp_total_count_sr = tmp_total_count_sr + 1
                tmp_total_psnr_rgb_sr = tmp_total_psnr_rgb_sr + psnr_rgb
                tmp_total_ssim_rgb_sr = tmp_total_ssim_rgb_sr + ssim_rgb
                tmp_total_psnr_y_sr = tmp_total_psnr_y_sr + psnr_y
                tmp_total_ssim_y_sr = tmp_total_ssim_y_sr + ssim_y

            else:
                
                tmp_total_count_ip = tmp_total_count_ip + 1
                tmp_total_psnr_rgb_ip = tmp_total_psnr_rgb_ip + psnr_rgb
                tmp_total_ssim_rgb_ip = tmp_total_ssim_rgb_ip + ssim_rgb
                tmp_total_psnr_y_ip = tmp_total_psnr_y_ip + psnr_y
                tmp_total_ssim_y_ip = tmp_total_ssim_y_ip + ssim_y

        total_count = total_count + tmp_total_count
        total_psnr_rgb = total_psnr_rgb + tmp_total_psnr_rgb
        total_ssim_rgb = total_ssim_rgb + tmp_total_ssim_rgb
        total_psnr_y = total_psnr_y + tmp_total_psnr_y
        total_ssim_y = total_ssim_y + tmp_total_ssim_y
        total_count_sr = total_count_sr + tmp_total_count_sr
        total_psnr_rgb_sr = total_psnr_rgb_sr + tmp_total_psnr_rgb_sr
        total_ssim_rgb_sr = total_ssim_rgb_sr + tmp_total_ssim_rgb_sr
        total_psnr_y_sr = total_psnr_y_sr + tmp_total_psnr_y_sr
        total_ssim_y_sr = total_ssim_y_sr + tmp_total_ssim_y_sr
        total_count_ip = total_count_ip + tmp_total_count_ip
        total_psnr_rgb_ip = total_psnr_rgb_ip + tmp_total_psnr_rgb_ip
        total_ssim_rgb_ip = total_ssim_rgb_ip + tmp_total_ssim_rgb_ip
        total_psnr_y_ip = total_psnr_y_ip + tmp_total_psnr_y_ip
        total_ssim_y_ip = total_ssim_y_ip + tmp_total_ssim_y_ip

        print('*' * 80)
        print(scene)
        print('psnr_rgb:', tmp_total_psnr_rgb / tmp_total_count)
        print('ssim_rgb:', tmp_total_ssim_rgb / tmp_total_count)
        print('psnr_y:', tmp_total_psnr_y / tmp_total_count)
        print('ssim_y:', tmp_total_ssim_y / tmp_total_count)

        print('psnr_rgb_sr:', tmp_total_psnr_rgb_sr / tmp_total_count_sr)
        print('ssim_rgb_sr:', tmp_total_ssim_rgb_sr / tmp_total_count_sr)
        print('psnr_y_sr:', tmp_total_psnr_y_sr / tmp_total_count_sr)
        print('ssim_y_sr:', tmp_total_ssim_y_sr / tmp_total_count_sr)

        print('psnr_rgb_ip:', tmp_total_psnr_rgb_ip / tmp_total_count_ip)
        print('ssim_rgb_ip:', tmp_total_ssim_rgb_ip / tmp_total_count_ip)
        print('psnr_y_ip:', tmp_total_psnr_y_ip / tmp_total_count_ip)
        print('ssim_y_ip:', tmp_total_ssim_y_ip / tmp_total_count_ip)

        str_rgb = str_rgb + ' & ' + str(round(tmp_total_psnr_rgb / tmp_total_count, 2))
        str_rgb = str_rgb + ' & ' + str(round(tmp_total_ssim_rgb / tmp_total_count, 4))
        str_y = str_y + ' & ' + str(round(tmp_total_psnr_y / tmp_total_count, 2))
        str_y = str_y + ' & ' + str(round(tmp_total_ssim_y / tmp_total_count, 4))
        str_rgb_sr = str_rgb_sr + ' & ' + str(round(tmp_total_psnr_rgb_sr / tmp_total_count_sr, 2))
        str_rgb_sr = str_rgb_sr + ' & ' + str(round(tmp_total_ssim_rgb_sr / tmp_total_count_sr, 4))
        str_y_sr = str_y_sr + ' & ' + str(round(tmp_total_psnr_y_sr / tmp_total_count_sr, 2))
        str_y_sr = str_y_sr + ' & ' + str(round(tmp_total_ssim_y_sr / tmp_total_count_sr, 4))
        str_rgb_ip = str_rgb_ip + ' & ' + str(round(tmp_total_psnr_rgb_ip / tmp_total_count_ip, 2))
        str_rgb_ip = str_rgb_ip + ' & ' + str(round(tmp_total_ssim_rgb_ip / tmp_total_count_ip, 4))
        str_y_ip = str_y_ip + ' & ' + str(round(tmp_total_psnr_y_ip / tmp_total_count_ip, 2))
        str_y_ip = str_y_ip + ' & ' + str(round(tmp_total_ssim_y_ip / tmp_total_count_ip, 4))

        file_txt.write('*' * 80 + '\n')
        file_txt.write(scene + '\n')
        file_txt.write('psnr_rgb :' + str(tmp_total_psnr_rgb / tmp_total_count) + '\n')
        file_txt.write('ssim_rgb :' + str(tmp_total_ssim_rgb / tmp_total_count) + '\n')
        file_txt.write('psnr_y :' + str(tmp_total_psnr_y / tmp_total_count) + '\n')
        file_txt.write('ssim_y :' + str(tmp_total_ssim_y / tmp_total_count) + '\n')

        file_txt.write('psnr_rgb_sr :' + str(tmp_total_psnr_rgb_sr / tmp_total_count_sr) + '\n')
        file_txt.write('ssim_rgb_sr :' + str(tmp_total_ssim_rgb_sr / tmp_total_count_sr) + '\n')
        file_txt.write('psnr_y_sr :' + str(tmp_total_psnr_y_sr / tmp_total_count_sr) + '\n')
        file_txt.write('ssim_y_sr :' + str(tmp_total_ssim_y_sr / tmp_total_count_sr) + '\n')

        file_txt.write('psnr_rgb_ip :' + str(tmp_total_psnr_rgb_ip / tmp_total_count_ip) + '\n')
        file_txt.write('ssim_rgb_ip :' + str(tmp_total_ssim_rgb_ip / tmp_total_count_ip) + '\n')
        file_txt.write('psnr_y_ip :' + str(tmp_total_psnr_y_ip / tmp_total_count_ip) + '\n')
        file_txt.write('ssim_y_ip :' + str(tmp_total_ssim_y_ip / tmp_total_count_ip) + '\n')


    str_rgb = str_rgb + ' & ' + str(round(total_psnr_rgb / total_count, 2))
    str_rgb = str_rgb + ' & ' + str(round(total_ssim_rgb / total_count, 4))
    str_y = str_y + ' & ' + str(round(total_psnr_y / total_count, 2))
    str_y = str_y + ' & ' + str(round(total_ssim_y / total_count, 4))
    str_rgb_sr = str_rgb_sr + ' & ' + str(round(total_psnr_rgb_sr / total_count_sr, 2))
    str_rgb_sr = str_rgb_sr + ' & ' + str(round(total_ssim_rgb_sr / total_count_sr, 4))
    str_y_sr = str_y_sr + ' & ' + str(round(total_psnr_y_sr / total_count_sr, 2))
    str_y_sr = str_y_sr + ' & ' + str(round(total_ssim_y_sr / total_count_sr, 4))
    str_rgb_ip = str_rgb_ip + ' & ' + str(round(total_psnr_rgb_ip / total_count_ip, 2))
    str_rgb_ip = str_rgb_ip + ' & ' + str(round(total_ssim_rgb_ip / total_count_ip, 4))
    str_y_ip = str_y_ip + ' & ' + str(round(total_psnr_y_ip / total_count_ip, 2))
    str_y_ip = str_y_ip + ' & ' + str(round(total_ssim_y_ip / total_count_ip, 4))

    str_rgb = str_rgb + ' \\\\ \n'
    str_y = str_y + ' \\\\ \n'
    str_rgb_sr = str_rgb_sr + ' \\\\ \n'
    str_y_sr = str_y_sr + ' \\\\ \n'
    str_rgb_ip = str_rgb_ip + ' \\\\ \n'
    str_y_ip = str_y_ip + ' \\\\ \n'

    total_str_rgb = total_str_rgb + str_rgb
    total_str_y = total_str_y + str_y
    total_str_rgb_sr = total_str_rgb_sr + str_rgb_sr
    total_str_y_sr = total_str_y_sr + str_y_sr
    total_str_rgb_ip = total_str_rgb_ip + str_rgb_ip
    total_str_y_ip = total_str_y_ip + str_y_ip

    print('*' * 80)
    print('total result')
    print('psnr_rgb:', total_psnr_rgb / total_count)
    print('ssim_rgb:', total_ssim_rgb / total_count)
    print('psnr_y:', total_psnr_y / total_count)
    print('ssim_y:', total_ssim_y / total_count)

    print('psnr_rgb_sr:', total_psnr_rgb_sr / total_count_sr)
    print('ssim_rgb_sr:', total_ssim_rgb_sr / total_count_sr)
    print('psnr_y_sr:', total_psnr_y_sr / total_count_sr)
    print('ssim_y_sr:', total_ssim_y_sr / total_count_sr)

    print('psnr_rgb_ip:', total_psnr_rgb_ip / total_count_ip)
    print('ssim_rgb_ip:', total_ssim_rgb_ip / total_count_ip)
    print('psnr_y_ip:', total_psnr_y_ip / total_count_ip)
    print('ssim_y_ip:', total_ssim_y_ip / total_count_ip)

    file_txt.write('*' * 80 + '\n')
    file_txt.write('total result' + '\n')
    file_txt.write('psnr_rgb :' + str(total_psnr_rgb / total_count) + '\n')
    file_txt.write('ssim_rgb :' + str(total_ssim_rgb / total_count) + '\n')
    file_txt.write('psnr_y :' + str(total_psnr_y / total_count) + '\n')
    file_txt.write('ssim_y :' + str(total_ssim_y / total_count) + '\n')

    file_txt.write('psnr_rgb_sr :' + str(total_psnr_rgb_sr / total_count_sr) + '\n')
    file_txt.write('ssim_rgb_sr :' + str(total_ssim_rgb_sr / total_count_sr) + '\n')
    file_txt.write('psnr_y_sr :' + str(total_psnr_y_sr / total_count_sr) + '\n')
    file_txt.write('ssim_y_sr :' + str(total_ssim_y_sr / total_count_sr) + '\n')

    file_txt.write('psnr_rgb_ip :' + str(total_psnr_rgb_ip / total_count_ip) + '\n')
    file_txt.write('ssim_rgb_ip :' + str(total_ssim_rgb_ip / total_count_ip) + '\n')
    file_txt.write('psnr_y_ip :' + str(total_psnr_y_ip / total_count_ip) + '\n')
    file_txt.write('ssim_y_ip :' + str(total_ssim_y_ip / total_count_ip) + '\n')

    file_txt.write('*' * 80 + '\n')
    file_txt.write('total_str_rgb' + '\n')
    file_txt.write(total_str_rgb)

    file_txt.write('*' * 80 + '\n')
    file_txt.write('total_str_y' + '\n')
    file_txt.write(total_str_y)

    file_txt.write('*' * 80 + '\n')
    file_txt.write('total_str_rgb_sr' + '\n')
    file_txt.write(total_str_rgb_sr)

    file_txt.write('*' * 80 + '\n')
    file_txt.write('total_str_y_sr' + '\n')
    file_txt.write(total_str_y_sr)

    file_txt.write('*' * 80 + '\n')
    file_txt.write('total_str_rgb_ip' + '\n')
    file_txt.write(total_str_rgb_ip)

    file_txt.write('*' * 80 + '\n')
    file_txt.write('total_str_y_ip' + '\n')
    file_txt.write(total_str_y_ip)

    file_txt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Tanks_and_Temples', choices=['Tanks_and_Temples', 'ETH'])
    args = parser.parse_args()

    if args.dataset == 'Tanks_and_Temples':
        scenes = ['Train', 'Playground', 'M60', 'Truck']
        gt_path = '/home/user2/datasets/Tanks_and_Temples_HLBIC_FR/HR/x4/'
        gt_lr_path = '/home/user2/datasets/Tanks_and_Temples_HLBIC_FR/LR/x4/'
    else:
        scenes = ['delivery_area', 'electro', 'forest', 'playground', 'terrains']
        gt_path = '/home/user2/datasets/ETH_v1/HR'
        gt_lr_path = '/home/user2/datasets/ETH_v1/LR'

    path = '/home/user2/SASRNet/exp/experiments/last-200000-ssimalpha-final'
    save_path = '/home/user2/SASRNet/exp/experiments/result-last-200000-ssimalpha-final/'

    epoch = '199999'
    save_every = False

    file_txt=open(os.path.join('exp/experiments', f'SASRNet-{args.dataset}-result-{epoch}.txt'),mode='w')

    get_result(path, save_path, file_txt, args.dataset, scenes, gt_path, gt_lr_path, epoch, save_every)

