# create miniImageNet dataset
'''
https://github.com/ethanhe42/mini-ImageNet
modify the code to create miniimagenet_yu dataset, which does not resize the original images'size
'''



import os, shutil
import random as rd
rd.seed(1)
import sys
import argparse

parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
parser.add_argument('--set', default="train", type=str)
parser.add_argument('--data_dir', default='/home/xiaoyan/Documents/Data/ImageNet/train', type=str) # extrat from ImageNet train
parser.add_argument('--dst_dir', default='/home/xiaoyan/Documents/Data/miniimagenet_yu', type=str)
args = parser.parse_args()

data_dir = args.data_dir
dst_dir = args.dst_dir
# minitrain = open('outputs/sourcetrain.txt', 'w')
# minival = open('outputs/sourceval.txt', 'w')
# minitest = open('outputs/sourcetest.txt', 'w')
# imagenet_train_dir = 'train.txt'




class_list = os.listdir(data_dir)

map_number2node_cls_dict ={}
image_dict = {}

for cls in class_list:
    cls_path = os.path.join(data_dir, cls)
    images = os.listdir(cls_path)
    images = sorted(images) 
    # use ncode_cls as key rather than cls(number)
    ncode_cls = images[0].split('_')[0]
    image_dict[ncode_cls] = images

    map_number2node_cls_dict[ncode_cls]= cls

def read_csv(file_name, img_idx_dict):
    with open(file_name) as fin:
        dummy = fin.readline()
        while True:
            line = fin.readline()
            if line == '':
                break
            line = line[:-1].split(',')
            img, cls = line[0:2]
            img_idx = int(img[9:-4]) - 1
            if not cls in img_idx_dict:
                img_idx_dict[cls] = []
            img_idx_dict[cls].append(img_idx)

""" read corresponding .csv file """
set_idx_dict = {}
read_csv(f'{args.set}.csv', set_idx_dict)


newf2c = {}

""" data copy """
try:
    os.mkdir(dst_dir)
except:
    pass
try:
    os.mkdir(os.path.join(dst_dir, 'train'))
    os.mkdir(os.path.join(dst_dir, 'val'))
    os.mkdir(os.path.join(dst_dir, 'test'))
except:
    pass


###### create mapping dict
cls_idx = 0

for i, ncode_cls in enumerate(set_idx_dict):
    print(f"{i} processing for class {ncode_cls}--------------------------")
    idx_list = set_idx_dict[ncode_cls]
    rd.shuffle(idx_list)
    try:
        os.mkdir(os.path.join(dst_dir, args.set, ncode_cls))
    except:
        pass
    for idx_idx in range(len(idx_list)): #len(idx_list)=600, value could be 0-1029-or larger
        idx = idx_list[idx_idx]
        number_cls=map_number2node_cls_dict[ncode_cls]
        src = os.path.join(data_dir, number_cls, image_dict[ncode_cls][idx])
        if cls not in newf2c:
            newf2c[ncode_cls] = str(cls_idx)
            cls_idx += 1
        dst = os.path.join(dst_dir, args.set, ncode_cls, image_dict[ncode_cls][idx])
        # if idx_idx >= 50:
        #     dst = os.path.join(dst_dir, 'train', cls, image_dict[cls][idx])
        #     minitrain.write(src + ' ' + newf2c[cls]+'\n')
        # else:
        #     dst = os.path.join(dst_dir, 'val', cls, image_dict[cls][idx])
        #     minival.write(src + ' ' + newf2c[cls]+'\n')
        print(src + ' -> ' + dst)
        os.symlink(src,dst)
        #shutil.copyfile(src, dst)

# for cls in val_idx_dict:
#     idx_list = val_idx_dict[cls]
#     try:
#         os.mkdir(os.path.join(dst_dir, 'test', cls))
#     except:
#         pass
#     for idx in idx_list:
#         src = os.path.join(data_dir, cls, image_dict[cls][idx])
#         dst = os.path.join(dst_dir, 'test', cls, image_dict[cls][idx])
#         if cls not in newf2c:
#             newf2c[cls] = str(cls_idx)
#             cls_idx += 1
#         minitest.write(src + ' ' + newf2c[cls]+'\n')
#         print(src + ' -> ' + dst)
#         os.symlink(src,dst)
#         #shutil.copyfile(src, dst)

# minitest.close()
# minitrain.close()
# minival.close()
