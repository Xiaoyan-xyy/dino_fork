'''
file to store plot functions
'''
import argparse
import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
# from sklearn import svm


def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: an tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
    is the number of tokens in x.
    :param y: a tensor of descriptors of shape Bx1x(t_y)xd' where d' is the dimensionality of the descriptors and t_y
    is the number of tokens in y.
    :return: cosine similarity between all descriptors in x and all descriptors in y. Has shape of Bx1x(t_x)x(t_y) """
    result_list = []
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)  # Bx1x1xd'
        result_list.append(torch.nn.CosineSimilarity(dim=3)(token, y))  # Bx1xt
    return torch.stack(result_list, dim=2)  # Bx1x(t_x)x(t_y)



#####################
# iterative plot
#####################

def show_similarity_interactive(image_path_a: str, image_path_b: str, centroid_a : torch.Tensor, centroid_b: torch.Tensor, 
                                load_size: int = 224, layer: int = 11,
                                list_facet: list = ['key','query','value', 'token'], 
                                bin: bool = False, stride: int = 4, model_type: str = 'dino_vits8',
                                num_sim_patches: int = 1):
    """
     finding similarity between a descriptor in one image to the all descriptors in the other image.
     :param image_path_a: path to first image.
     :param image_path_b: path to second image.
     :param load_size: size of the smaller edge of loaded images. If None, does not resize.
     :param layer: layer to extract descriptors from.
     :param list_facet: list of facets to extract descriptors from.
     :param bin: if True use a log-binning descriptor.
     :param stride: stride of the model.
     :param model_type: type of model to extract descriptors from.
     :param num_sim_patches: number of most similar patches from image_b to plot.
    """
    # color_lookup_table
    color_lookup_table = {
        'key':(1, 1, 0, 0.8),
        'query':(0.137, 1, 0, 0.8),
        'value':(0, 0.03, 1, 0.8),
        'token':(0.474, 0, 0.647, 0.8),
        'wkey':(1, 1, 0.5, 0.8),
        'wquery':(0.137, 1, 0.5, 0.8),
        'wvalue':(0.5, 0.03, 1, 0.8),
        'wtoken':(0.474, 0.5, 0.647, 0.8),
        'wfc1': (0.3,0.3,1,0.8),
        'wfc2': (0.5,0.5,1,0.8)
    }
    # extract descriptors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTExtractor(model_type, stride, device=device)
    patch_size = extractor.model.patch_embed.patch_size

    print(f"patch_size = {patch_size}, stride = {stride}")
    # image_batch_a, image_pil_a = extractor.preprocess(image_path_a, load_size)
    # image_batch_b, image_pil_b = extractor.preprocess(image_path_b, load_size)

    image_batch_a, image_pil_a = extractor.preprocess(image_path_a)
    image_batch_b, image_pil_b = extractor.preprocess(image_path_b)
    print(f"Aftre resize according to args.load_size, a.size={image_batch_a.shape}, b.size={image_batch_b.shape}.")
    
    list_simi = []
    for facet in list_facet:
        descs_a = extractor.extract_descriptors(image_batch_a.to(device), centroid_a.to(device),layer, facet, bin, include_cls=True)
        num_patches_a, load_size_a = extractor.num_patches, extractor.load_size
        descs_b = extractor.extract_descriptors(image_batch_b.to(device), centroid_b.to(device), layer, facet, bin, include_cls=True)
        num_patches_b, load_size_b = extractor.num_patches, extractor.load_size
        # calculate and plot similarity between image1 and image2 descriptors
        # TODO: implement a function to calculate similarity under one of two options: cosine similarity or exemplar SVM
        
        similarities = chunk_cosine_sim(descs_a, descs_b)
        # similarities = chunk_exemplar_svm_similarity(descs_a, descs_b)
        print(f"{facet} :descs_a.max(),descs_a.min(),similarities.max(),similarities.min() {descs_a.max(),descs_a.min(),similarities.max(),similarities.min()}.")
        print(torch.unique(similarities),f"{facet}:unique similarities")
        list_simi.append(similarities)
        


    # plot
    fig, axes = plt.subplots(1, 2+len(list_facet))
    [axi.set_axis_off() for axi in axes.ravel()]
    visible_patches = []
    radius = patch_size // 2
    # plot image_a and the chosen patch. if nothing marked chosen patch is cls patch.
    axes[0].imshow(image_pil_a)
    # plot image_b and the closest patch in it to the chosen patch in image_a
    axes[1+len(list_facet)].imshow(image_pil_b)
 
    for i, similarities in enumerate(list_simi):
        curr_similarities = similarities[0, 0, 0, 1:]  # similarity to all spatial descriptors, without cls token
        curr_similarities = curr_similarities.reshape(num_patches_b)
        axes[1+i].imshow(curr_similarities.cpu().numpy(), cmap='jet')
        axes[1+i].set_title(f"{list_facet[i]}")


        # add points
        sims, idxs = torch.topk(curr_similarities.cpu().flatten(), num_sim_patches)
        for idx, sim in zip(idxs, sims):
            y_descs_coor, x_descs_coor = idx // num_patches_b[1], idx % num_patches_b[1]
            # most of cases patch_size = stride
            # patch_size = model.patch_embed.patch_size
            # stride = .model.patch_embed.proj.stride
            center = ((x_descs_coor - 1) * stride + stride + patch_size // 2 - .5,
                    (y_descs_coor - 1) * stride + stride + patch_size // 2 - .5) 
            patch = plt.Circle(center, radius, color = color_lookup_table[list_facet[i]])
            axes[1+len(list_facet)].add_patch(patch)
            visible_patches.append(patch)
    plt.tight_layout()
    plt.draw()


    # start interactive loop
    # get input point from user
    fig.suptitle(f'Model {model_type} :Select a point on the left image. \n Right click to stop.', fontsize=12)
    plt.draw()
    pts = np.asarray(plt.ginput(1, timeout=-1, mouse_stop=plt.MouseButton.RIGHT, mouse_pop=None))
    while len(pts) == 1:
        y_coor, x_coor = int(pts[0, 1]), int(pts[0, 0])
        new_H = patch_size / stride * (load_size_a[0] // patch_size - 1) + 1
        new_W = patch_size / stride * (load_size_a[1] // patch_size - 1) + 1
        y_descs_coor = int(new_H / load_size_a[0] * y_coor)
        x_descs_coor = int(new_W / load_size_a[1] * x_coor)

        # print(pts, "pts")
        # print(new_H, new_W, "new h w")
        # print(y_descs_coor,"y_descs_coor",x_descs_coor,"x_descs_coor")

        # reset previous marks
        for patch in visible_patches:
            patch.remove()
            visible_patches = []

        # draw chosen point
        center = ((x_descs_coor - 1) * stride + stride + patch_size // 2 - .5,
                  (y_descs_coor - 1) * stride + stride + patch_size // 2 - .5)
        patch = plt.Circle(center, radius, color=(1, 0, 0, 0.75)) # fix color
        axes[0].add_patch(patch)
        visible_patches.append(patch)

        # get and draw current similarities
        raveled_desc_idx = num_patches_a[1] * y_descs_coor + x_descs_coor
        reveled_desc_idx_including_cls = raveled_desc_idx + 1

        for i, facet, similarities in zip(range(len(list_facet)), list_facet, list_simi):
            curr_similarities = similarities[0, 0, reveled_desc_idx_including_cls, 1:]
            curr_similarities = curr_similarities.reshape(num_patches_b) 
            axes[i+1].imshow(curr_similarities.cpu().numpy(), cmap='jet')
            axes[1+i].set_title(f"{facet}")

            # get and draw most similar points
            sims, idxs = torch.topk(curr_similarities.cpu().flatten(), num_sim_patches)
            for idx, sim in zip(idxs, sims):
                y_descs_coor, x_descs_coor = idx // num_patches_b[1], idx % num_patches_b[1]
                center = ((x_descs_coor - 1) * stride + stride + patch_size // 2 - .5,
                        (y_descs_coor - 1) * stride + stride + patch_size // 2 - .5)
                patch = plt.Circle(center, radius, color=color_lookup_table[facet])
                axes[1+len(list_facet)].add_patch(patch)
                visible_patches.append(patch)
        plt.tight_layout()
        plt.draw()

        # get input point from user
        fig.suptitle('Select a point on the left image', fontsize=16)
        plt.draw()
        pts = np.asarray(plt.ginput(1, timeout=-1, mouse_stop=plt.MouseButton.RIGHT, mouse_pop=None))
        
