
from skimage import io
from skimage.morphology import erosion, dilation, disk
import tensorflow as tf
import os
import numpy as np
import glob
import cv2
# from load_data import load_t
from checkpoint.ckp_whu_9134_maphigh_multiloss_c4grade.data import load_t
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
batch_size=1
img=tf.placeholder(tf.float32,[batch_size,512,512,4])

# from model import mapnet
# from mapnet import mapnet
# from unet import unet
# from pspnet import pspnet
# from model0109 import mapnet
from checkpoint.ckp_whu_9134_maphigh_multiloss_c4grade.mapnet_high import mapnet
checkpoint_dir = r'./checkpoint/ckp_whu_9134_maphigh_multiloss_c4grade'
savepath='test_result_temp'
test_img=np.array(sorted(glob.glob(r'C:\Data\img/*.png')))
# labels = sorted(glob.glob(r'/home/dell/install/mapnet/dataset/whu/test/gt/*.png'))
# test_img = np.array(sorted(glob.glob(r'/home/dell/install/mapnet/dataset/massbuilding/test/img/*.png')))
# labels = np.array(sorted(glob.glob(r'/home/dell/install/mapnet/dataset/massbuilding/test/gt/*.png')))
# test_img = np.array(sorted(glob.glob(r'D:\F\massbuilding\test\img/*.png')))
# labels = np.array(sorted(glob.glob(r'D:\F\massbuilding\test\gt/*.png')))
# test_img = np.array(sorted(glob.glob(r'D:\Jupyter_projects\mapnet\test\image/*.png')))
labels = np.array(sorted(glob.glob(r'D:\Jupyter_projects\mapnet\test\gt/*.png')))


# test_img = np.array(sorted(glob.glob(r'/home/dell/install/mapnet/dataset/urban3D/test/image/*.png')))
# labels = np.array(sorted(glob.glob(r'/home/dell/install/mapnet/dataset/urban3D/test/label/*.png')))
# test_img = np.array(sorted(glob.glob(r'/home/dell/install/mapnet/dataset/aerial/test/img/*.tif')))
# labels = np.array(sorted(glob.glob(r'/home/dell/install/mapnet/dataset/aerial/test/gt/*.tif')))


save_pat = r"/home/lc/Jupyter_projects/resatt/dataset/rgb_result"
result_vector = []

pred,_,edge1=mapnet(img,is_training=False)
pred=tf.nn.sigmoid(pred)
saver=tf.train.Saver(tf.global_variables())


def count():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


def count_flops(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    print('FLOPs: {}'.format(flops.total_float_ops))


def save():
    tf.global_variables_initializer().run()

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print(ckpt_name)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        graph = tf.get_default_graph()
        count_flops(graph)

    for j in range(0,len(test_img)):
        # print(test_img)
        x_batch = test_img[j]
        # print(x_batch)
        i = x_batch.split('/')[-1]
        x_batch=load_t(x_batch)
        # x_batch = cv2.imread(x_batch) / 255.0
        #x_batch=tifffile.imread(x_batch) / 255.0

        x_batch = np.expand_dims(x_batch, axis=0)

        #                y_batch=test_labels[j]
        #                i=y_batch.split('/')[-1]
        #                y_batch=scipy.misc.imread(y_batch)
        #                y_batch=np.expand_dims(y_batch,axis=0)
        #                y_batch=np.expand_dims(y_batch,axis=-1)
        feed_dict = {img: x_batch

                     }

        # predict,pyd1,x31,x11,pred1,= sess.run([pred,x1,x2,x3,x_], feed_dict=feed_dict)
        predict,assd,edge = sess.run([pred,_,edge1], feed_dict=feed_dict)
        # print(at)
        # scipy.misc.imsave('./output/featuremap/conv1/{}.png'.format(i), np.squeeze(pred))

        # color = np.ones([predict.shape[0], predict.shape[1], 1])
        # color[predict < 0.3] = 0
        # color[predict > 0.7] = 0
        #
        # # print (predict.shape)
        # result = np.squeeze(color)

        # predict[predict < 0.2] = 0
        # predict[predict > 0.8] = 0
        # predict[predict >= 0.2] = 1
        # #
        predict[predict < 0.5] = 0
        predict[predict >= 0.5] = 255

        result = np.squeeze(predict)
        # i=i.split('.')[0]
        i=os.path.basename(i)[0:-4]
        # scipy.misc.imsave('/home/lc/Jupyter_projects/resatt/img_att_pred/{}.png'.format(i), result)
        #scipy.misc.imsave('./test_result_temp/' + i, result)
        # print(r'./test_result_temp/{}.png'.format(i))
        cv2.imwrite(r'./test_result_temp/{}.png'.format(i), result)

        # pyd1=255.0*pyd1
        # x31=255.0*x31
        # x11=255.0*x11
        # pred1=255.0*pred1
        #
        #
        edge=edge*255
        for a in range(edge.shape[-1]):
            ft1 = edge[0, :, :, a]
            img_test1 = cv2.resize(ft1, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite('C:\Data\edge/{}_'.format(a) + i+".png", img_test1)
        #
        # for a in range(x31.shape[-1]):
        #     ft2 = x31[0, :, :, a]
        #     img_test2 = cv2.resize(ft2, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
        #     cv2.imwrite('./output/featuremap/x2/{}_'.format(a) + i+".png", img_test2)
        #
        # for a in range(x11.shape[-1]):
        #     ft0 = x11[0, :, :, a]
        #     img_test0 = cv2.resize(ft0, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
        #     cv2.imwrite('./output/featuremap/x3/{}_'.format(a) + i+".png", img_test0)
        #
        # for a in range(pred1.shape[-1]):
        #     ft3 = pred1[0, :, :, a]
        #     img_test3 = cv2.resize(ft3, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
        #     cv2.imwrite('./output/featuremap/x4/{}_'.format(a) + i+".png", img_test3)
        """
        for a in range(pydout.shape[-1]):
            py = pydout[0, :, :, a]
            img_test1 = cv2.resize(py, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
            scipy.misc.imsave('./output/featuremap/st32/{}_'.format(a) + i+".png", img_test1)
        """
        #all_model_checkpoint_paths: "model.ckpt-48925"
        #all_model_checkpoint_paths: "model.ckpt-54965"


def pair_to_rgb(gen_img, tar_img, background='back', use_dilation=False, disk_value=2):
    """
    args:
        gen_img: (ndarray) in [img_rows, img_cols], dytpe=unit8
        tar_img: (ndarray) in [img_rows, img_cols], dytpe=unit8
        background: (str) ['black', 'white']
    return:
        rgb_img: red -> false positive;
                 green -> true positive;
                 blue -> false positive;
    """
    # enhance outline border
    if use_dilation:
        gen_img = dilation(gen_img, disk(disk_value))
        tar_img = dilation(tar_img, disk(disk_value))

    if background == "black":
        # saving rgb results
        rgb_img = np.zeros((gen_img.shape[0], gen_img.shape[1], 3), np.uint8)
        # assign false negative as red channel
        rgb_img[:, :, 0][np.logical_and(gen_img == 1, tar_img == 0)] = 255
        # assign true positive as green channel
        rgb_img[:, :, 1][np.logical_and(gen_img == 1, tar_img == 1)] = 255
        # assign false positive as blue channel
        rgb_img[:, :, 2][np.logical_and(gen_img == 0, tar_img == 1)] = 255
    else:
        # saving rgb results
        rgb_img = np.ones((gen_img.shape[0], gen_img.shape[1], 3), np.uint8) * 255
        # assign false negative as red channel
        rgb_img[:, :, 1][np.logical_and(gen_img == 1, tar_img == 0)] = 0
        rgb_img[:, :, 2][np.logical_and(gen_img == 1, tar_img == 0)] = 0
        # assign true positive as green channel
        rgb_img[:, :, 0][np.logical_and(gen_img == 1, tar_img == 1)] = 0
        rgb_img[:, :, 2][np.logical_and(gen_img == 1, tar_img == 1)] = 0
        # assign false positive as blue channel
        rgb_img[:, :, 0][np.logical_and(gen_img == 0, tar_img == 1)] = 0
        rgb_img[:, :, 1][np.logical_and(gen_img == 0, tar_img == 1)] = 0
    return rgb_img


def iou(predict, label):
    Intersect = []
    Union = []
    for i in range(2):
        Ii = np.sum(np.logical_and(predict == i, label == i))
        Ui = np.sum(predict == i) + np.sum(label == i) - np.sum(np.logical_and(predict == i, label == i))
        Intersect.append(Ii)
        Union.append(Ui)
    return Intersect, Union


def f_score(predict, label):
    tp1 = []
    fp1 = []
    tn1 = []
    fn1 = []
    for i in range(1, 2):
        tp = np.sum(np.logical_and(predict == i, label == i))
        fp = np.sum(np.logical_and(predict == i, label != i))
        tn = np.sum(np.logical_and(predict != i, label != i))
        fn = np.sum(np.logical_and(predict != i, label == i))
        tp1.append(tp)
        fp1.append(fp)
        tn1.append(tn)
        fn1.append(fn)
    return tp1, fp1, tn1, fn1


def cal_iou(save_rgb):
    l1 = []
    l2 = []
    l3 = []
    l4 = []
    TP = []
    FP = []
    TN = []
    FN = []
    for i in range(len(labels)):
        label = io.imread(labels[i])
        predict = io.imread(predicts[i]) // 255
        # label=label//255
        if save_rgb:
            save_dir = os.path.join(save_pat, labels[i].split('/')[-1])
            rgb_img = pair_to_rgb(predict, label)
            cv2.imwrite(save_dir, rgb_img)

        ap = np.sum(label == predict)
        total = np.sum(label != 2)
        Inter, Uni = iou(predict, label)
        tp1, fp1, tn1, fn1 = f_score(predict, label)
        TP.append(tp1)
        FP.append(fp1)
        TN.append(tn1)
        FN.append(fn1)

        l1.append(Inter)
        l2.append(Uni)
        l3.append(ap)
        l4.append(total)

    a = np.sum(l1, axis=0)
    b = np.sum(l2, axis=0)
    IoU = a * 1.0 / b
    print('iou:{}'.format(IoU))
    mean_iu = np.sum(IoU[:2]) / 2
    print('mean_iu:{}'.format(mean_iu))
    precision = np.sum(TP, axis=0) / (np.sum(TP, axis=0) + np.sum(FP, axis=0))
    print('--precision:{}'.format(precision))

    recall = np.sum(TP, axis=0) / (np.sum(TP, axis=0) + np.sum(FN, axis=0))
    print('--recall:{}'.format(recall))
    F_score = 2 * (precision * recall) / (precision + recall)
    #    print(IoU)
    #    print(mean_iu)
    print('F_score:{}'.format(F_score))
    mean_ap = np.sum(l3) * 1.0 / np.sum(l4)
    print('mean_ap:{}'.format(mean_ap))

    return IoU, mean_iu, mean_ap


with tf.Session() as sess:
      save()
      predicts = sorted(glob.glob(r'C:\Users\Administrator\PycharmProjects\mapnet\buildingext\\' + savepath + '/*.png'))
      cal_iou(save_rgb=False)

