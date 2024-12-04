'''
calculate the trak score for DINO model 
patch-wise dino score
'''

# command:
# python -m torch.distributed.run --nproc_per_node=1 dino_trak.py 

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

from patchtrak import patchTRAKer
from PIL import Image

def num_patches_per_sample(stride, patch_size, img, include_cls=False):
    num_patches = 0
    H, W = img.shape[-2:]
    num_patches_per_sample = (1 + (H - patch_size) // stride[0], 1 + (W - patch_size) // stride[1])
    num_patches += num_patches_per_sample[0] * num_patches_per_sample[1]
    if include_cls:
        num_patches += 1
    return num_patches

def calculate_num_patches_per_batch(stride, patch_size, data_loader, include_cls):
    num_patches_per_batch=[]
    for batch in data_loader:
        num_patches_in_batch = 0
        for i in range(batch[0].shape[0]):
            num_patches_in_batch += num_patches_per_sample(stride, patch_size, batch[0][i].unsqueeze(0), include_cls)
        num_patches_per_batch.append(num_patches_in_batch)
    return num_patches_per_batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
    parser.add_argument('--batch_size_per_gpu', default=64, type=int, help='Per-GPU batch-size')
    parser.add_argument('--train_imgs_folder', default='/home/xiaoyan/Documents/Data/miniimagenet_yu/', type=str)
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
    
    # related to distributed computing
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")


    parser.add_argument('--data_path', default='/home/xiaoyan/Documents/Data/ImageNet', type=str)

    parser.add_argument("--task", type=str, default="dino_self_supervised_learning")
    parser.add_argument('--centroids_folder', default='Results/', type=str)


    parser.add_argument('--include_cls', default=False, type=utils.bool_flag, help='Include the cls token graident.')
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



print(f"check if using cuda: {torch.cuda.is_available()}")

# must run utils.init_distributed_mode(args) before using torch.utils.data.DistributedSampler
utils.init_distributed_mode(args)
print("git:\n  {}\n".format(utils.get_sha()))
print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
cudnn.benchmark = True


## set the dataset(same as eval_kmeans.py) and load the centroid features
class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx

transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
dataset_train = datasets.ImageFolder(os.path.join(args.train_imgs_folder, "train"), transform=transform)

sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, shuffle=False) # shuffle= True in default TODO: check if shuffle is necessary
data_loader_train = torch.utils.data.DataLoader(
    dataset_train,
    sampler=sampler,
    batch_size=args.batch_size_per_gpu,
    num_workers=args.num_workers,
    pin_memory=True,
    drop_last=False,
) 


## load the centroid features
centroid_feats = torch.load(os.path.join(args.centroids_folder, f"train_centroids.pth"))



# initialize TRAKer
# put the load centroid features and a pair of images

trainloader_num_patches_per_batch  = calculate_num_patches_per_batch(stride=(args.patch_size,args.patch_size),
                                                patch_size=args.patch_size, 
                                                data_loader=data_loader_train,
                                                include_cls=args.include_cls)


# example of grad_wrt: [ 'blocks.11.attn.qkv.weight', 'blocks.11.attn.qkv.bias', 'blocks.11.attn.proj.weight',
#  'blocks.11.attn.proj.bias', 'blocks.11.norm2.weight', 'blocks.11.norm2.bias']
print(f"initialize the patchTRAKer---------------")
traker = patchTRAKer(model = model,
                patch_size = args.patch_size,
                stride = (args.patch_size, args.patch_size),
                task = args.task, # dino_self_supervised_learning
                train_set_size = len(dataset_train),
                centroids = centroid_feats,
                grad_wrt = ["blocks.11.attn.qkv.weight"],
                facet = ["wkey"],
                include_cls= args.include_cls,
                total_num_patches = sum(trainloader_num_patches_per_batch))

model_id = 0
traker.load_checkpoint(args.pretrained_weights, model_id=model_id) # args.pretrained_weights= None, actually have already been loaded before

# # caculate the features for the training set
# for i, (batch, num_patches) in enumerate(zip( data_loader_train, trainloader_num_patches_per_batch)):
#     traker.featurize(batch = batch, num_pacthes_in_batch = num_patches)
#     print('Finish featurizing the batch: ', i)
# traker.finalize_features()


# # need to reset everything and and zero gradients
# iterate through each target image and the exp_name will be set to the target image name
# Therefore, the trak features will be saved in the folder with the target image name for each target image
target_img_list =[1222]

for target_img_idx in target_img_list:
    path, label = dataset_train.samples[target_img_idx]
    target_img = dataset_train.loader(path)
    print(type(target_img),"the type and shape" )
    if dataset_train.transform is not None:
        target_img = dataset_train.transform(target_img)
    
    target_img_name = os.path.basename(path).split('.')[0]
    print(target_img_name, "target_img_name")

   

    num_patches = num_patches_per_sample(stride = (args.patch_size,args.patch_size), 
                                        patch_size = args.patch_size, img = target_img, include_cls=False)
    traker.start_scoring_checkpoint(checkpoint=args.pretrained_weights,
                                    model_id = model_id, # only one model id
                                    exp_name=f'test_{target_img_name}',
                                    num_targets = num_patches)    

   
    traker.score_one_sample(sample= target_img.unsqueeze(0),label = torch.tensor(label).unsqueeze(0),num_patches_in_img = num_patches)
    scores = traker.finalize_scores(exp_name=f'test_{target_img_name}')


    print(f"scores shape: {scores.shape}")

    # TODO finish visual part function in _utils/plot.py
    # TODO about register and unregister hooks 
    # give if and else to   skip features part

    # DINO loss
    visualize_similarity_map(feat_a, feat_b, img_a, img_b)


# load the score and calculate the cosine similarity



