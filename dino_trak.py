'''
calculate the trak score for DINO model 
patch-wise dino score
'''

import os
import sys
import argparse

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits

from sklearn.cluster import MiniBatchKMeans
import numpy as np
from termcolor import colored
from scipy.optimize import linear_sum_assignment

from patchtrak import TRAKer



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
    parser.add_argument('--batch_size_per_gpu', default=256, type=int, help='Per-GPU batch-size')
    parser.add_argument('--temperature', default=0.07, type=float,
        help='Temperature used in the voting coefficient')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--dump_features', default="Results/", 
        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load_features', default=None, help="""If the features have
        already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/home/xiaoyan/Documents/Data/ImageNet', type=str)

    parser.add_argument("--task", type=str, default="DINO_self_supervised_learning")
    parser.add_argument('--centroids_folder', default='Results/', type=str)

    args = parser.parse_args()




# ============ building network ... ============
    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch](num_classes=0)
        model.fc = nn.Identity()
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()


## load the centroid features

# # load the image and pick the the target class centroid
image_path ="/home/xiaoyan/Documents/Data//home/xiaoyan/Documents/Data/miniimagenet_yu/train/n02101006/n02101006_354.JPEG"
target_class = 10
image1 = img
image2 = image1
# calulate the patch number


patch_num2 = patch_num1
target_class = 0
split = 'train'

centroid_feats = torch.load(os.path.join(args.centroids_folder, f"{split}_centroid.pth"))
centroid_feat = centroid_feats[target_class]
# initialize TRAKer
# put the load centroid features and a pair of images
    


traker = TRAKer(model = model,
                task = args.task,
                patch_num = patch_num1,
                centroid = centroid_feat)
traker.featurize(image= image1)
traker.finalize_featurize()

# need to reset everything and and zero gradients
traker.start_scoring_checkpoint(checkpoint,
                                patch_num=patch_num2)
traker.score(image=image2)
scores=traker.finalize_score()