# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# command:
# python -m torch.distributed.run --nproc_per_node=1 eval_kmean.py 

# note:
# batch size 256 is faster than 1024, maybe related to the number of workers

# TODO:
# try mini imagenet
# https://github.com/gitabcworld/MatchingNetworks/blob/master/utils/create_miniImagenet.py

# load model with teacher head

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

from vision_transformer import DINOHead

import utils
import vision_transformer as vits

from sklearn.cluster import MiniBatchKMeans
import numpy as np
from termcolor import colored
from scipy.optimize import linear_sum_assignment

def extract_feature_pipeline(args):
    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = ReturnIndexDataset(os.path.join(args.data_path, "train"), transform=transform)
    dataset_val = ReturnIndexDataset(os.path.join(args.data_path, "val"), transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ building network ... ============
    if args.use_dino_head:
        # model will be backbone + dino head
        assert args.pretrained_weights!= None, "Please specify the path to the pretrained weights for the full model(including dino head)"
        backbone = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        head = DINOHead(embed_dim =backbone.embed_dim, out_dim=65536, ause_bn=False)
        model = nn.Sequential(backbone, head)
        # TODO still in construction
        raise NotImplementedError
    else:
        # only backbone
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

    # ============ extract features ... ============
    print("Extracting features for train set...")
    train_features = extract_features(model, data_loader_train, args.use_cuda)
    print("Extracting features for val set...")
    test_features = extract_features(model, data_loader_val, args.use_cuda)

    if utils.get_rank() == 0:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

    train_labels = torch.tensor([s[-1] for s in dataset_train.samples]).long()
    test_labels = torch.tensor([s[-1] for s in dataset_val.samples]).long()
    # save features and labels
    if args.dump_features and dist.get_rank() == 0:
        torch.save(train_features.cpu(), os.path.join(args.dump_features, "trainfeat.pth"))
        torch.save(test_features.cpu(), os.path.join(args.dump_features, "testfeat.pth"))
        torch.save(train_labels.cpu(), os.path.join(args.dump_features, "trainlabels.pth"))
        torch.save(test_labels.cpu(), os.path.join(args.dump_features, "testlabels.pth"))
    return train_features, test_features, train_labels, test_labels


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, multiscale=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    for samples, index in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        if multiscale:
            feats = utils.multi_scale(samples, model)
        else:
            feats = model(samples).clone()

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features

def run_kmeans(x, nmb_clusters, max_points, val_features=None, use_faiss=True, 
               verbose=False, seed=None):
    """ Run kmeans algorithm with faiss or sklearn 
    """
    if use_faiss:
        import faiss
        n_data, d = x.shape
        clus = faiss.Clustering(d, nmb_clusters)
        if seed is None: clus.seed = np.random.randint(1234)
        else: clus.seed = seed
        clus.niter = 20
        clus.max_points_per_centroid = max_points
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = False
        flat_config.device = 0
        index = faiss.GpuIndexFlatIP(res, d, flat_config)

        # perform the training
        clus.train(x, index)
        _, I = index.search(val_features, 1)
        centroids = faiss.vector_float_to_array(clus.centroids).reshape(nmb_clusters, d)
        preds = I.ravel()

    else:
        if seed is None: seed = np.random.randint(1234)
        kmeans = MiniBatchKMeans(n_clusters = nmb_clusters, random_state = seed,
                             batch_size = 1000)
        kmeans.fit(x)
        preds = kmeans.predict(val_features)
        centroids = kmeans.cluster_centers_
     

    return preds, centroids


@torch.no_grad()
def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    """ Hungarian matching of clustered predictions with targets
        output: matched pairs with cluster id to target id
    """
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]

    assert (preds_k == targets_k)  # one to one
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res


def _majority_match(flat_preds, flat_targets):
    """ Majority matching of clustered predictions with targets
        output: matched pairs with cluster id to target id
    """
    from collections import Counter
    res = []
    for c1 in np.unique(flat_preds):
        gt_i = flat_targets[flat_preds == c1]
        c2, _ = Counter(gt_i).most_common()[0]
        res.append((c1, c2))
    return res



def kmeans(features, features_val, labels_val, nmb_clusters, target_nub_clusters, use_faiss=True, 
            num_trials=1, whiten_feats=True, return_all_metrics=False, seed=None):
    """ Kmeans clustering to nmb_clusters of features
        output: centroid ids after hungarian matching with labels
    """
    from sklearn import metrics
    import scipy.cluster.vq as vq

    # make sure all features are in numpy format
    features = features.cpu().numpy()
    features_val = features_val.cpu().numpy()
    labels_val = labels_val.cpu().numpy()

    print(features.shape, "features shape")
    print(features_val.shape, "val features shape")
    print(labels_val.shape, "val labels shape")
    print(labels_val.max(),labels_val.min() )
  

    # l2-normalize whitened features
    if whiten_feats:
        features = vq.whiten(features)
        features_val = vq.whiten(features_val)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    features_val = features_val / np.linalg.norm(features_val, axis=1, keepdims=True)

    # keep track of clustering metrics: acc, nmi, ari
    acc_lst, nmi_lst, ari_lst = [], [], []
    for i in range(num_trials):
        if seed is None: curr_seed = i
        else: curr_seed = seed

        # run kmeans
        num_elems = labels_val.shape[0]
        print(use_faiss, "use faiss")
        I, centroids = run_kmeans(features, nmb_clusters=nmb_clusters, 
                max_points=int(features.shape[0]/nmb_clusters), 
                val_features=features_val, 
                use_faiss=use_faiss, verbose=False, seed=curr_seed)
        pred_labels = np.array(I)

        print(centroids.shape, "centroids shape")
        print(I.shape, ":predicted shape,", I[0],":first element of prediction")
        print(f"the first label of pred_labels should be {labels_val[0]}")

        if nmb_clusters == target_nub_clusters:
            # number of clusters equals number of classes C in dataset
            match = _hungarian_match(pred_labels, 
                                     labels_val, 
                                     nmb_clusters, 
                                     nmb_clusters)

        else:
            # number of clusters excedes number of classes C in dataset
            assert nmb_clusters > target_nub_clusters
            match = _majority_match(pred_labels, labels_val)

        reordered_preds = np.zeros(num_elems)
        reordered_centroids = np.zeros_like(centroids)

        for pred_i, target_i in match:
            reordered_preds[pred_labels == int(pred_i)] = int(target_i)
            reordered_centroids[int(target_i)] = centroids[int(pred_i)]
            

        # gather performance metrics)
        acc = int((reordered_preds == labels_val).sum()) / float(num_elems)
        if return_all_metrics:
            nmi_lst.append(metrics.normalized_mutual_info_score(labels_val, pred_labels))
            ari_lst.append(metrics.adjusted_rand_score(labels_val, pred_labels))
        print(colored(
            'Computed KMeans RUN {} with CLS ACC {:.2f}'.format(i, 100*acc), 'yellow'))
        acc_lst.append(acc)

    # return performance metrics
    if return_all_metrics:
        return {'ACC': 100*np.mean(acc_lst), 'NMI': 100*np.mean(nmi_lst), 
                'ARI': 100*np.mean(ari_lst)}, reordered_preds, reordered_centroids
    else:
        return {'ACC': 100*np.mean(acc_lst)}, reordered_preds, reordered_centroids

class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
    parser.add_argument('--batch_size_per_gpu', default=256, type=int, help='Per-GPU batch-size')
    parser.add_argument('--temperature', default=0.07, type=float,
        help='Temperature used in the voting coefficient')
    parser.add_argument('--use_dino_head', default=False, type=utils.bool_flag)
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--dump_features', default="Results/", 
        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load_features', default="Results/", help="""If the features have
        already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/home/xiaoyan/Documents/Data/miniimagenet_yu/', type=str)

    # k-means params
    parser.add_argument("--num_classes", type=int, default=64, help='dataset classes')
    parser.add_argument("--overcluster", type=int, default=1)
    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    if args.dump_features:
        args.dump_features = os.path.join(args.dump_features, f"{args.arch}{args.patch_size}_s{args.patch_size}") # stride = patch size TODO

    if args.load_features:
        train_features = torch.load(os.path.join(args.load_features, "trainfeat.pth"))
        test_features = torch.load(os.path.join(args.load_features, "testfeat.pth"))
        train_labels = torch.load(os.path.join(args.load_features, "trainlabels.pth"))
        test_labels = torch.load(os.path.join(args.load_features, "testlabels.pth"))
    else:
        # need to extract features !
        train_features, test_features, train_labels, test_labels = extract_feature_pipeline(args)

    if utils.get_rank() == 0:
        if args.use_cuda:
            train_features = train_features.cuda()
            test_features = test_features.cuda()
            train_labels = train_labels.cuda()
            test_labels = test_labels.cuda()
        
        print("Features are ready!\nStart the k-NN classification.")

        # cluster train features
        res, reordered_preds, reordered_centroids = kmeans(train_features, train_features, train_labels,
            nmb_clusters=args.num_classes*args.overcluster, target_nub_clusters= args.num_classes, use_faiss=False, num_trials=1, 
            whiten_feats=True, return_all_metrics=True, seed=1234)
    
        # save centroids
        if args.dump_features and dist.get_rank() == 0:
            torch.save(train_features.cpu(), os.path.join(args.dump_features, "train_centroids.pth"))
    dist.barrier()
