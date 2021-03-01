import cv2
import os
import glob
import numpy as np

image_path=np.array(sorted(glob.glob(r"C:\Data\img\*.png")))
save_path=r"C:\Data\result"

def load_t(x):

    img = cv2.imread(x,-1)
    basename=os.path.basename(x)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad = grade(gray)
    savepath=os.path.join(save_path,"55"+basename)
    cv2.imwrite(savepath,255*grad)


def grade(img):
    x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=5)
    y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=5)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    mi=np.min(dst)
    ma=np.max(dst)
    return (dst-mi)/(ma-mi)
for name in image_path:
    load_t(name)