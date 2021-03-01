
import tensorflow as tf
from PIL import Image, ImageEnhance
import numpy as np
import glob
import scipy
import random
import cv2, os


# def load_batch(x, y):
#     x1 = []
#     y1 = []
#     for i in range(len(x)):
#         # img=tifffile.imread(x[i])/255.0
#         img = cv2.imread(x[i],-1)
#         gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#         grad=grade(gray)
#
#         img = img / 255.0
#         # gray = gray / 255.0
#         # ndvi = rgb2vi(img)
#         image = cv2.merge([grad, img])
#         # image=img
#
#         # mul_img=cv2.merge([gray,img,ndvi])
#         lab = cv2.imread(y[i], -1) / 255.0
#         image, lab = data_augmentation(image, lab)
#
#         lab = lab.reshape(512, 512, 1)
#         # lab = lab.reshape(512, 512, 1)
#         x1.append(image)
#         y1.append(lab)
#
#     y1 = np.array(y1).astype(np.float32)
#     return x1, y1
#
# def load_t(x):
#     # img=tifffile.imread(x[i])/255.0
#     img = cv2.imread(x, -1)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     grad = grade(gray)
#
#     img = img / 255.0
#     # gray = gray / 255.0
#
#     # ndvi = rgb2vi(img)
#     mul_img = cv2.merge([grad, img])
#     # mul_img = img
#     # mul_img = cv2.merge([gray,img])
#
#     return mul_img
def load_batch(x, y):
    x1 = []
    y1 = []
    for i in range(len(x)):
        # img=tifffile.imread(x[i])/255.0
        img = cv2.imread(x[i],-1)
        lab = cv2.imread(y[i], -1)
        img1, lab1 = data_augmentation(img, lab)
        # img1=img
        # lab1=lab

        # cv2.imwrite(r'C:\Data\tianchi\train\agu.png',img1)
        gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        grad=grade(gray)
        img1 = img1 / 255.0
        # gray = gray / 255.0
        # ndvi = rgb2vi(img1)
        image = cv2.merge([img1,grad])
        # image=img1
        # mul_img=cv2.merge([gray,img,ndvi])
        #0-255
        # label = (lab1/255).reshape(512, 512, 1)
        label = lab1.reshape(512, 512, 1)
        x1.append(image)
        y1.append(label)
    y1 = np.array(y1).astype(np.float32)
    return x1, y1

def load_t(x):
    # img=tifffile.imread(x[i])/255.0
    img = cv2.imread(x, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad = grade(gray)
    img = img / 255.0
    # gray = gray / 255.0
    # ndvi = rgb2vi(img)
    mul_img = cv2.merge([img,grad])
    # mul_img = img
    # mul_img = cv2.merge([gray,img])
    return mul_img

def grade(img):
    x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    mi=np.min(dst)
    ma=np.max(dst)
    return (dst-mi)/(ma-mi)
# def grade(img):
#     x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
#     y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
#     absX = cv2.convertScaleAbs(x)
#     absY = cv2.convertScaleAbs(y)
#     dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
#     mi = np.min(img)
#     ma = np.max(img)
#     img=(img - mi) / (ma - mi)
#
#     mi=np.min(dst)
#     ma=np.max(dst)
#     dst= (dst-mi)/(ma-mi)
#     return img+dst


def prepare_data():


    #     glob.glob(r'/media/lc/vge_lc/DL_DATE_BUILDING/WHU/cropped image tiles and raster labels/test/image/*.png')))
    # test_gt = np.array(sorted(
    #     glob.glob(r'/media/lc/vge_lc/DL_DATE_BUILDING/WHU/cropped image tiles and raster labels/test/gt/*.png')))

    # img = np.array(sorted(
    #     glob.glob(r'/media/lc/数据专用-红线项目/hn_gf2/train_data/img_result_256/*.png')))
    # label = np.array(sorted(
    #     glob.glob(r'/media/lc/数据专用-红线项目/hn_gf2/train_data/gt5_rsult256/*.png')))
    # Urban 512*512 5979
    # img = np.array(sorted(glob.glob(r"/media/lc/数据专用-红线项目/urban3/train_urb512/img/*.png")))
    # label = np.array(sorted(glob.glob(r'/media/lc/数据专用-红线项目/urban3/train_urb512/gt/*.png')))
    # test_img = np.array(sorted(glob.glob(r'/media/lc/数据专用-红线项目/urban3/uba512_test/test_img/*.png')))
    # test_gt = np.array(sorted(glob.glob(r'/media/lc/数据专用-红线项目/urban3/uba512_test/test_gt/*.png')))

    # img = np.array(sorted(glob.glob(r'/home/dell/install/mapnet/dataset/train/image/*.png')))
    # label = np.array(sorted(glob.glob(r'/home/dell/install/mapnet/dataset/train/gt2/*.png')))
    # test_img = np.array(sorted(glob.glob(r'/home/dell/install/mapnet/dataset/test/image/*.png')))
    # test_gt = np.array(sorted(glob.glob(r'/home/dell/install/mapnet/dataset/test/gt2/*.png')))

    # 0-1 iou=0.93
    # img = np.array(sorted(glob.glob(r'F:\DL_DATE_BUILDING\WHU\train\image/*.png')))
    # label = np.array(sorted(glob.glob(r'F:\DL_DATE_BUILDING\WHU\train\gt/*.png')))
    # test_img = np.array(sorted(glob.glob(r'F:\DL_DATE_BUILDING\WHU\test\imgt/*.png')))
    # test_gt = np.array(sorted(glob.glob(r'F:\DL_DATE_BUILDING\WHU\test\gtt/*.png')))

    # 0-255 iou=0.61
    # img = np.array(sorted(glob.glob(r'/media/dell/DATA/cl/pytorch/Khaos_0303/dataset/train/*.tif')))
    # label = np.array(sorted(glob.glob(r'/media/dell/DATA/cl/pytorch/Khaos_0303/dataset/train_GT/*.tif')))
    # test_img = np.array(sorted(glob.glob(r'/media/dell/DATA/cl/pytorch/Khaos_0303/dataset/valid/*.tif')))
    # test_gt = np.array(sorted(glob.glob(r'/media/dell/DATA/cl/pytorch/Khaos_0303/dataset/valid_GT/*.tif')))

    #whu 0-1
    img = np.array(sorted(glob.glob(r'F:/DL_DATE_BUILDING/WHU/train/image/*.png')))
    label = np.array(sorted(glob.glob(r'F:/DL_DATE_BUILDING/WHU/train/gt/*.png')))
    test_img = np.array(sorted(glob.glob(r'F:/DL_DATE_BUILDING/WHU/test/imgt/*.png')))
    test_gt = np.array(sorted(glob.glob(r'F:/DL_DATE_BUILDING/WHU/test/gtt/*.png')))
    # test_img = np.array(sorted(glob.glob(r'F:\data\DL_Train_Data\SEG\WHU\3. The cropped aerial image tiles and raster labels\3. The cropped image tiles and raster labels\test\image/*.tif')))
    # test_gt = np.array(sorted(glob.glob(r'F:\data\DL_Train_Data\SEG\WHU\3. The cropped aerial image tiles and raster labels\3. The cropped image tiles and raster labels\test\label/*.tif')))

    # 0-255
    # img = np.array(sorted(glob.glob(r'D:\F\massbuilding\train\img/*.png')))
    # label = np.array(sorted(glob.glob(r'D:\F\massbuilding\train\gt/*.png')))
    # test_img = np.array(sorted(glob.glob(r'D:\F\massbuilding\test\imgs/*.png')))
    # test_gt = np.array(sorted(glob.glob(r'D:\F\massbuilding\test\gts/*.png')))

    #0-1
    # img = np.array(sorted(glob.glob(r'/home/dell/install/mapnet/dataset/tianchi/img1/Sub/*.tif')))
    # label = np.array(sorted(glob.glob(r'/home/dell/install/mapnet/dataset/tianchi/lab1/Sub/*.tif')))
    # test_img = np.array(sorted(glob.glob(r'/media/dell/DATA/cl/pytorch/Khaos_0303/dataset/valid/*.tif')))
    # test_gt = np.array(sorted(glob.glob(r'/media/dell/DATA/cl/pytorch/Khaos_0303/dataset/valid_GT/*.tif')))

    # img = np.array(sorted(glob.glob(r'/home/dell/install/mapnet/dataset/train22/img/*.png')))
    # label = np.array(sorted(glob.glob(r'/home/dell/install/mapnet/dataset/train22/gt/*.png')))
    # test_img = np.array(sorted(glob.glob(r'/home/dell/install/mapnet/dataset/test22/img/*.png')))
    # test_gt = np.array(sorted(glob.glob(r'/home/dell/install/mapnet/dataset/test22/gt/*.png')))

    # img = np.array(sorted(glob.glob(r'/media/dell/DATA/liaoc/tianchi/dataset/train/*.tif')))
    # label = np.array(sorted(glob.glob(r'/media/dell/DATA/liaoc/tianchi/dataset/train_GT/*.tif')))
    # test_img = np.array(sorted(glob.glob(r'/media/dell/DATA/liaoc/tianchi/dataset/test/*.tif')))
    # test_gt = np.array(sorted(glob.glob(r'/media/dell/DATA/liaoc/tianchi/dataset/test_GT/*.tif')))

    return img, label, test_img, test_gt
    # return img, label, img, label

def rotate_flip_image(img, gt):
    # [k = 0-2:rotate random 90,180,270,degree, 3: vertical-flip, 4: horizontal-flip, 5: both-flip]
    k = np.random.randint(6)
    ###### Image rotation ######
    if k < 3:
        angle = (k+1) * 90
        M = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), angle, 1.0)
        new_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)
        new_gt = cv2.warpAffine(gt, M, (gt.shape[1], gt.shape[0]), flags=cv2.INTER_NEAREST)
    ###### Image flip ######
    if k == 3:
        flip_axis = 0
    elif k == 4:
        flip_axis = 1
    elif k == 5:
        flip_axis = -1
    if k > 2 and k < 6:
        new_img = cv2.flip(img, flip_axis)
        new_gt = cv2.flip(gt, flip_axis)
    return new_img, new_gt


def change_contrast_brightness(img):
    choice = np.random.randint(6)
    if choice == 0:
        updated_img = blur_image(img)
    elif choice == 1:
        updated_img = add_gaussian_noise(img)
    elif choice == 2:
        updated_img = augment_contrast_brightness(img)
    else:
        updated_img = clahe(img)
    return updated_img

def blur_image(img):
    choice = np.random.randint(2)
    if choice == 0:
        blur_img = cv2.GaussianBlur(img, (5, 5), 0)
    else:
        blur_img = cv2.medianBlur(img, 5)
    return blur_img

def add_gaussian_noise(img):
    mean = np.mean(img)
    fixed_sd = 15
    noisy_img = img + np.random.normal(mean, fixed_sd, img.shape).astype(np.int32)
    noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img_clipped

def clahe(img):
    gridsize = 100
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize, gridsize))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr

def augment_contrast_brightness(img):
    img = img.astype(np.float32)
    contrast = np.random.randint(980, 1021) / 1000.
    brightness = np.random.randint(-2, 3)
    updated_img = cv2.addWeighted(img, contrast, img, 0, brightness)
    updated_img = np.clip(updated_img, 0, 255)
    return updated_img.astype(np.uint8)

def data_augmentation(img, gt):
    ####### rotation and flip augmentation #######
    option = np.random.randint(8)
    if option < 6:
        img, gt = rotate_flip_image(img, gt)
    ####### contrast and brightness augmentation #######
    option = np.random.randint(10)
    if option < 7:
        img = change_contrast_brightness(img)
    return img, gt
    ####### scale augmentation #######
    option = np.random.randint(6)
    if option < 3:
        x = random.randint(0, 512 - 350)
        y = random.randint(0, 512 - 350)
        image = img[y:y + 350, x:x + 350, :]
        label = gt[y:y + 350, x:x + 350]
        img = cv2.resize(image, (512, 512),interpolation=cv2.INTER_CUBIC)
        gt = cv2.resize(label, (512, 512),interpolation=cv2.INTER_CUBIC)
    return img, gt