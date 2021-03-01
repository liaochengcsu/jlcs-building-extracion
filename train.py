#
import os
import cv2
import time
# import utils
import tensorflow as tf
import numpy as np
import skimage.io as io
import argparse
import keras as K

from data import load_batch,prepare_data,load_t
# from MAPNet.mapnet import mapnet
# from mapnet import pspnet
# from buildingext.mapnet_git import mapnet
from model import mapnet
# from mapnet_git import mapnet
# from hrnet_lc import hrnetv2
# from MAPNet.mapnet_ori import mapnet
# from MAPNet.mapnet_diat import mapnet_diat


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=2, help='Number of images in each batch')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Number of images in each batch')
parser.add_argument('--checkpoint_step', type=int, default=20, help='How often to save checkpoints (epochs)')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--clip_size', type=int, default=450, help='Width of cropped input image to network')
parser.add_argument('--num_epochs', type=int, default=250, help='Number of etensorflow.python.layers.convolutional.Conv2D object at 0x7fb04f3f6390pochs to train for')
parser.add_argument('--h_flip', type=bool, default=True, help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=bool, default=True, help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--color', type=bool, default=True, help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--rotation', type=bool, default=True, help='randomly rotate, the imagemax rotation angle in degrees.')
parser.add_argument('--start_valid', type=int, default=30, help='Number of epoch to valid')
parser.add_argument('--valid_step', type=int, default=2, help="Number of step to validation")


args = parser.parse_args()
num_images=[]
train_img, train_label,valid_img,valid_lab= prepare_data()
num_batches=len(train_img)//(args.batch_size)

img=tf.compat.v1.placeholder(tf.float32,[None,args.crop_height,args.crop_width,4])
is_training=tf.compat.v1.placeholder(tf.bool)
label=tf.compat.v1.placeholder(tf.float32,[None,args.crop_height,args.crop_height,1])

pred, edge=mapnet(img,is_training)
# pred=mapnet(img,is_training)
print(pred)
# python train.py --name label2city_512p_feat --instance_feat --gpu_ids 0 --label_nc 2 --resize_or_crop none
pred1=tf.nn.sigmoid(pred)
edge1=tf.nn.sigmoid(edge)


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


def binary_focal_loss_fixed(y_true, y_pred):
     alpha = tf.constant(2, dtype=tf.float32)
     gamma = tf.constant(0.25, dtype=tf.float32)
     y_true = tf.cast(y_true, tf.float32)
     alpha_t = y_true * alpha + (K.backend.ones_like(y_true) - y_true) * (1 - alpha)
     p_t = y_true * y_pred + (K.backend.ones_like(y_true) - y_true) * ( K.backend.ones_like(y_true) - y_pred) + K.backend.epsilon()
     focal_loss = - alpha_t * K.backend.pow((K.backend.ones_like(y_true) - p_t), gamma) * K.backend.log(p_t)
     return K.backend.mean(focal_loss)


def dice_loss(y_true, y_pred):
    # y_pred need be compute after sigmoid
    smooth = 1.
    y_true_f = K.backend.flatten(y_true)
    y_pred_f = K.backend.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.backend.sum(intersection) + smooth) / (K.backend.sum(y_true_f) + K.backend.sum(y_pred_f) + smooth)
    return 1. - score


update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    labelt = label
    sig = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=pred)
    edgeloss=dice_loss(y_true=labelt,y_pred=edge1)
    sigmoid_cross_entropy_loss = tf.reduce_mean(0.6*sig+0.4*edgeloss)
    train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(sigmoid_cross_entropy_loss)

    # labelat= tf.layers.max_pooling2d(labelt, 4, 4)
    # sig = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=pred)
    # sigmoid_cross_entropy_loss = tf.reduce_mean(sig)
    # train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(sigmoid_cross_entropy_loss)
    # edgeloss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labelat, logits=edge)
    # sigmoid_cross_entropy_edgeloss = tf.reduce_mean(edgeloss)
    # train_edge = tf.train.AdamOptimizer(args.learning_rate).minimize(sigmoid_cross_entropy_loss)

    # sig = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=pred)
    # sigmoid_cross_entropy_loss = tf.reduce_mean(sig)
    # train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(sigmoid_cross_entropy_loss)
    """
   
    # sig=tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=pred)
    # sigmoid_cross_entropy_loss = tf.reduce_mean(sig)
    labelt=label
    sig = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=pred)
    edgeloss = dice_loss(y_true=labelt, y_pred=pred1)
    # print(edgeloss)
    sigmoid_cross_entropy_loss = tf.reduce_mean(0.6 * sig + 0.4 * edgeloss)
    train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(sigmoid_cross_entropy_loss)
    """
saver=tf.train.Saver(var_list=tf.global_variables(),max_to_keep=5)
# graph = tf.get_default_graph()
# count_flops(graph)
# print(count())


def load():
    import re
    print("Reading checkpoints...")
    checkpoint_dir = './checkpoint/'
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
        print("Checkpoint {} read Successed".format(ckpt_name))
        return True, counter
    else:
        print("Checkpoint not find ")
        return False, 0


def train():
    tf.global_variables_initializer().run()
    could_load, checkpoint_counter = load()
    if could_load:
        start_epoch = (int)(checkpoint_counter / num_batches)
        start_batch_id = checkpoint_counter - start_epoch * num_batches
        counter = checkpoint_counter
        print("Checkpoint Load Successed")
    else:
        start_epoch = 0
        start_batch_id = 0
        counter = 1
        print("train from scratch...")
    train_iter=[]
    train_loss=[]
    IOU=0.5
    F1=0.7
    # utils.count_params()
    print("Total train image:{}".format(len(train_img)))
    print("Total validate image:{}".format(len(valid_img)))
    print("Total epoch:{}".format(args.num_epochs))
    print("Batch size:{}".format(args.batch_size))
    print("Learning rate:{}".format(args.learning_rate))
    print("Checkpoint step:{}".format(args.checkpoint_step))

    print("Data Argument:")
    print("h_flip: {}".format(args.h_flip))
    print("v_flip: {}".format(args.v_flip))
    print("rotate: {}".format(args.rotation))
    print("clip size: {}".format(args.clip_size))
    loss_tmp = []
    for i in range(start_epoch, args.num_epochs):
        epoch_time=time.time()
        id_list = np.random.permutation(len(train_img))

        # if (i > args.start_valid):
        #     if (i - args.start_valid) % args.valid_step == 0:
        #         val_iou = validation()
        #         print("last iou valu:{}".format(IOU))
        #         print("new_iou value:{}".format(val_iou))
        #         if val_iou > IOU:
        #             print("Save the checkpoint...")
        #             saver.save(sess, './mapmodfy/checkpoint/model.ckpt', global_step=counter, write_meta_graph=True)
        #             IOU = val_iou
        for j in range(start_batch_id, num_batches):
            img_d = []
            lab_d = []

            for ind in range(args.batch_size):
                id = id_list[j * args.batch_size + ind]
                img_d.append(train_img[id])
                lab_d.append(train_label[id])

            x_batch, y_batch = load_batch(img_d, lab_d)
            # print(x_batch)
            feed_dict = {img: x_batch,
                         label: y_batch,
                         is_training: True
                         }

            _, loss, pred1 = sess.run([train_step, sigmoid_cross_entropy_loss, pred], feed_dict=feed_dict)
            loss_tmp.append(loss)
            # print(loss)
            if (counter % 200 == 0):
                tmp = np.median(loss_tmp)
                train_iter.append(counter)
                train_loss.append(tmp)
                print('Epoch', i, '|Iter', counter, '|Loss', tmp)
                loss_tmp.clear()

            counter += 1
        start_batch_id = 0
        print('Time:', time.time() - epoch_time)

        # if(i > args.start_valid and (i+1)%5==0):
        #     saver.save(sess, './checkpoint/model.ckpt', global_step=counter)

        if (i>args.start_valid):
            if (i-args.start_valid)%args.valid_step==0:
                val_iou = validation()
                print("last iou valu:{}".format(IOU))
                print("new_iou value:{}".format(val_iou))
                if val_iou > IOU:
                    print("Save the checkpoint...")
                    saver.save(sess, './checkpoint/model.ckpt', global_step=counter, write_meta_graph=True)
                    IOU = val_iou
    saver.save(sess, './checkpoint/model.ckpt', global_step=counter)

def f_iou(predict, label):

    tp = np.sum(np.logical_and(predict == 1, label == 1))
    fp = np.sum(predict==1)
    fn = np.sum(label == 1)
    return tp,fp+fn-tp


# def f_score(predict, label):
#
#     tp = np.sum(np.logical_and(predict == 1, label == 1))
#     fp = np.sum(np.logical_and(predict == 1, label != 1))
#     tn = np.sum(np.logical_and(predict != 1, label != 1))
#     fn = np.sum(np.logical_and(predict != 1, label == 1))
#     if tp==0:
#         return 1.0,1.0
#     else:
#         return tp*1.0/(fn+fp+tp),2*tp*1.0/(2*tp+fn+fp)

#
def validation():

    print("validate...")
    inter=0
    unin=0
    for j in range(0,len(valid_img)):

        x_batch = valid_img[j]
        # x_batch = cv2.imread(x_batch) / 255.0
        # x_batch = cv2.imread(x_batch)
        x_batch=load_t(x_batch)
        x_batch = np.expand_dims(x_batch, axis=0)
        feed_dict = {img: x_batch,
                     is_training:False
                     }

        predict = sess.run(pred1, feed_dict=feed_dict)

        predict[predict < 0.5] = 0
        predict[predict >= 0.5] = 1
        result = np.squeeze(predict)
        #for the label value with 255
        # gt_value=io.imread(valid_lab[j])/255.0
        gt_value = io.imread(valid_lab[j])
        intr,unn=f_iou(gt_value,result)

        inter=inter+intr
        unin=unin+unn
    return inter*1.0/unin

# def validation():
#
#     print("validate...")
#     miou=0
#     mf1=0
#     for j in range(0,len(valid_img)):
#
#         x_batch = valid_img[j]
#         x_batch = io.imread(x_batch) / 255.0
#         x_batch = np.expand_dims(x_batch, axis=0)
#         feed_dict = {img: x_batch,
#                      is_training:False
#
#                      }
#
#         predict = sess.run(pred1, feed_dict=feed_dict)
#
#         predict[predict < 0.5] = 0
#         predict[predict >= 0.5] = 1
#         result = np.squeeze(predict)
#         gt_value=io.imread(valid_lab[j])
#         iou_,f1_=f_score(gt_value,result)
#
#         miou=miou+iou_
#         mf1=mf1+f1_
#     return miou*1.0/len(valid_img),mf1*1.0/len(valid_img)

with tf.Session() as sess:
    train()

# import os
# import time
# import utils
# import tensorflow as tf
# import numpy as np
# import skimage.io as io
# import argparse
#
# from data import load_batch,prepare_data
# # from MAPNet.mapnet import mapnet
# # from MAPNet.resnet_att import pspnet
# # from MAPNet.mapnet_diat import mapnet_diat
# # from MAPNet.hrnetv2 import hrnetv2
# # from MAPNet.pspnet import pspnet
# # from MAPNet.unet import unet
# from unet import unet_model
# # from MAPNet.resnet101 import resnet101
#
# # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# # os.environ["CUDA_VISIBLE_DEVICES"]="-1"
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--batch_size', type=int, default=6, help='Number of images in each batch')
# parser.add_argument('--learning_rate', type=float, default=0.001, help='Number of images in each batch')
#
# parser.add_argument('--checkpoint_step', type=int, default=20, help='How often to save checkpoints (epochs)')
# parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
# parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
# parser.add_argument('--clip_size', type=int, default=450, help='Width of cropped input image to network')
# parser.add_argument('--num_epochs', type=int, default=80, help='Number of epochs to train for')
# parser.add_argument('--h_flip', type=bool, default=True, help='Whether to randomly flip the image horizontally for data augmentation')
# parser.add_argument('--v_flip', type=bool, default=True, help='Whether to randomly flip the image vertically for data augmentation')
# parser.add_argument('--color', type=bool, default=True, help='Whether to randomly flip the image vertically for data augmentation')
# parser.add_argument('--rotation', type=bool, default=True, help='randomly rotate, the imagemax rotation angle in degrees.')
# parser.add_argument('--start_valid', type=int, default=20, help='Number of epoch to valid')
# parser.add_argument('--valid_step', type=int, default=1, help="Number of step to validation")
#
#
# args = parser.parse_args()
# num_images=[]
# train_img, train_label,valid_img,valid_lab= prepare_data()
# num_batches=len(train_img)//(args.batch_size)
#
# img=tf.placeholder(tf.float32,[None,args.crop_height,args.crop_width,3])
# is_training=tf.placeholder(tf.bool)
# label=tf.placeholder(tf.float32,[None,args.crop_height,args.crop_height,1])
#
# pred=unet_model(img,is_training)
# pred1=tf.nn.sigmoid(pred)
#
# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# with tf.control_dependencies(update_ops):
#
#     sig=tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=pred)
#     sigmoid_cross_entropy_loss = tf.reduce_mean(sig)
#     train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(sigmoid_cross_entropy_loss)
# saver=tf.train.Saver(var_list=tf.global_variables())
#
#
# def load():
#     import re
#     print("Reading checkpoints...")
#     checkpoint_dir = './checkpoint/'
#
#     ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
#     if ckpt and ckpt.model_checkpoint_path:
#         ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
#         saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
#         counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
#         print("Checkpoint {} read Successed".format(ckpt_name))
#         return True, counter
#     else:
#         print("Checkpoint not find ")
#         return False, 0
#
# def train():
#
#     tf.global_variables_initializer().run()
#
#     could_load, checkpoint_counter = load()
#     if could_load:
#         start_epoch = (int)(checkpoint_counter / num_batches)
#         start_batch_id = checkpoint_counter - start_epoch * num_batches
#         counter = checkpoint_counter
#         print("Checkpoint Load Successed")
#
#     else:
#         start_epoch = 0
#         start_batch_id = 0
#         counter = 1
#         print("train from scratch...")
#
#     train_iter=[]
#     train_loss=[]
#     IOU=0.65
#     F1=0.7
#     utils.count_params()
#     print("Total train image:{}".format(len(train_img)))
#     print("Total validate image:{}".format(len(valid_img)))
#     print("Total epoch:{}".format(args.num_epochs))
#     print("Batch size:{}".format(args.batch_size))
#     print("Learning rate:{}".format(args.learning_rate))
#     print("Checkpoint step:{}".format(args.checkpoint_step))
#
#     print("Data Argument:")
#     print("h_flip: {}".format(args.h_flip))
#     print("v_flip: {}".format(args.v_flip))
#     print("rotate: {}".format(args.rotation))
#     print("clip size: {}".format(args.clip_size))
#     loss_tmp = []
#     for i in range(start_epoch, args.num_epochs):
#         epoch_time=time.time()
#         id_list = np.random.permutation(len(train_img))
#
#         for j in range(start_batch_id, num_batches):
#             img_d = []
#             lab_d = []
#
#             for ind in range(args.batch_size):
#                 id = id_list[j * args.batch_size + ind]
#                 img_d.append(train_img[id])
#                 lab_d.append(train_label[id])
#
#             x_batch, y_batch = load_batch(img_d, lab_d)
#             feed_dict = {img: x_batch,
#                          label: y_batch,
#                          is_training:True
#                          }
#
#             _, loss, pred1 = sess.run([train_step, sigmoid_cross_entropy_loss, pred], feed_dict=feed_dict)
#
#             loss_tmp.append(loss)
#             if (counter % 5 == 0):
#                 tmp = np.median(loss_tmp)
#                 train_iter.append(counter)
#                 train_loss.append(tmp)
#                 print('Epoch', i, '|Iter', counter, '|Loss', tmp)
#                 loss_tmp.clear()
#
#             counter += 1
#         start_batch_id = 0
#         print('Time:', time.time() - epoch_time)
#
#         # saver.save(sess, './checkpoint/model.ckpt', global_step=counter)
#
#         if (i>args.start_valid):
#             if (i-args.start_valid)%args.valid_step==0:
#                 val_iou = validation()
#                 print("last iou valu:{}".format(IOU))
#                 print("new_iou value:{}".format(val_iou))
#                 if val_iou > IOU:
#                     print("Save the checkpoint...")
#                     saver.save(sess, './checkpoint/model.ckpt', global_step=counter, write_meta_graph=True)
#                     IOU = val_iou
#     saver.save(sess, './checkpoint/model.ckpt', global_step=counter)
#
# def f_iou(predict, label):
#
#     tp = np.sum(np.logical_and(predict == 1, label == 1))
#     fp = np.sum(predict==1)
#     fn = np.sum(label == 1)
#     return tp,fp+fn-tp
#
#
# # def f_score(predict, label):
# #
# #     tp = np.sum(np.logical_and(predict == 1, label == 1))
# #     fp = np.sum(np.logical_and(predict == 1, label != 1))
# #     tn = np.sum(np.logical_and(predict != 1, label != 1))
# #     fn = np.sum(np.logical_and(predict != 1, label == 1))
# #     if tp==0:
# #         return 1.0,1.0
# #     else:
# #         return tp*1.0/(fn+fp+tp),2*tp*1.0/(2*tp+fn+fp)
#
# #
# def validation():
#
#     print("validate...")
#     inter=0
#     unin=0
#     for j in range(0,len(valid_img)):
#
#         x_batch = valid_img[j]
#         x_batch = io.imread(x_batch) / 255.0
#         x_batch = np.expand_dims(x_batch, axis=0)
#         feed_dict = {img: x_batch,
#                      is_training:False
#
#                      }
#
#         predict = sess.run(pred1, feed_dict=feed_dict)
#
#         predict[predict < 0.5] = 0
#         predict[predict >= 0.5] = 1
#         result = np.squeeze(predict)
#         gt_value=io.imread(valid_lab[j])
#         intr,unn=f_iou(gt_value,result)
#
#         inter=inter+intr
#         unin=unin+unn
#     return inter*1.0/unin
#
#
# # def validation():
# #
# #     print("validate...")
# #     miou=0
# #     mf1=0
# #     for j in range(0,len(valid_img)):
# #
# #         x_batch = valid_img[j]
# #         x_batch = io.imread(x_batch) / 255.0
# #         x_batch = np.expand_dims(x_batch, axis=0)
# #         feed_dict = {img: x_batch,
# #                      is_training:False
# #
# #                      }
# #
# #         predict = sess.run(pred1, feed_dict=feed_dict)
# #
# #         predict[predict < 0.5] = 0
# #         predict[predict >= 0.5] = 1
# #         result = np.squeeze(predict)
# #         gt_value=io.imread(valid_lab[j])
# #         iou_,f1_=f_score(gt_value,result)
# #
# #         miou=miou+iou_
# #         mf1=mf1+f1_
# #     return miou*1.0/len(valid_img),mf1*1.0/len(valid_img)
#
#
# with tf.Session() as sess:
#     train()
#
