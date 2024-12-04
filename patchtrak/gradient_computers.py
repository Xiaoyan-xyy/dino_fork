"""
Computing features for the TRAK algorithm involves computing (and projecting)
per-sample gradients. This module contains classes that compute these
per-sample gradients. The :code:`AbstractFeatureComputer` class defines the
interface for such gradient computers. Then, we provide two implementations:
- :class:`FunctionalFeatureComputer`: A fast implementation that uses
  :code:`torch.func` to vectorize the computation of per-sample gradients, and
  thus fully levereage parallelism.
- :class:`IterativeFeatureComputer`: A more naive implementation that only uses
  native pytorch operations (i.e. no :code:`torch.func`), and computes per-sample
  gradients in a for-loop. This is often much slower than the functional
  version, but it is useful if you cannot use :code:`torch.func`, e.g., if you
  have an old version of pytorch that does not support it, or if your application
  is not supported by :code:`torch.func`.

"""
from abc import ABC, abstractmethod
from typing import Iterable, Optional
from torch import Tensor
from .utils import get_num_params, parameters_to_vector
from .modelout_functions import AbstractModelOutput
import logging
import torch
from typing import Union, List, Tuple
from PIL import Image
from torchvision import transforms as pth_transforms
from pathlib import Path
ch = torch
import re, sys
import torch.nn.functional as F


class AbstractGradientComputer(ABC):
    """Implementations of the GradientComputer class should allow for
    per-sample gradients.  This is behavior is enabled with three methods:

    - the :meth:`.load_model_params` method, well, loads model parameters. It can
      be as simple as a :code:`self.model.load_state_dict(..)`

    - the :meth:`.compute_per_sample_grad` method computes per-sample gradients
      of the chosen model output function with respect to the model's parameters.

    - the :meth:`.compute_loss_grad` method computes the gradients of the loss
      function with respect to the model output (which should be a scalar) for
      every sample.

    """

    @abstractmethod
    def __init__(
        self,
        model: torch.nn.Module,
        task: AbstractModelOutput,
        grad_dim: Optional[int] = None,
        dtype: Optional[torch.dtype] = torch.float16,
        device: Optional[torch.device] = "cuda",
    ) -> None:
        """Initializes attributes, nothing too interesting happening.

        Args:
            model (torch.nn.Module):
                model
            task (AbstractModelOutput):
                task (model output function)
            grad_dim (int, optional):
                Size of the gradients (number of model parameters). Defaults to
                None.
            dtype (torch.dtype, optional):
                Torch dtype of the gradients. Defaults to torch.float16.
            device (torch.device, optional):
                Torch device where gradients will be stored. Defaults to 'cuda'.

        """
        self.model = model
        self.modelout_fn = task
        self.grad_dim = grad_dim
        self.dtype = dtype
        self.device = device


    @abstractmethod
    def load_model_params(self, model) -> None:
        ...

    @abstractmethod
    def compute_per_sample_grad(self, batch: Iterable[Tensor]) -> Tensor:
        ...

    @abstractmethod
    def compute_loss_grad(self, batch: Iterable[Tensor], batch_size: int) -> Tensor:
        ...

class PatchGradientComputer(AbstractGradientComputer):
    def __init__(
        self,
        model: torch.nn.Module,
        patch_size: int,
        stride: Tuple[int, int],
        task: AbstractModelOutput,
        centroids: torch.Tensor,
        grad_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        grad_wrt: Optional[Iterable[str]] = None, # grad_wrt contains the layer_idx information, e.g, wkey_11
        facet: Optional[Iterable[str]] = None, # ["wkey"]
        include_cls: bool = False,
    ) -> None:
        super().__init__(model, task, grad_dim, dtype, device)
        self.model = model
        self.patch_size = patch_size
        self.stride = stride
        # self.load_model_params(model) # TODO check if we need this 
        self.grad_wrt = grad_wrt
        # get layer info from grad_wrt
        assert len(self.grad_wrt)==1, "grad_wrt should be a list of length 1" # TODO check if we need this
        self.layers = []
        for string in self.grad_wrt:
            numbers = list(filter(lambda x: x.isdigit(), string.split(".")))
            self.layers += [int(s) for s in numbers]
        print("layers", self.layers)
        self.facet = facet
        self.include_cls = include_cls
        self.centroids = centroids.to(self.device)
 

        self.logger = logging.getLogger("GradientComputer")

       
 
        self.hook_handlers = [] #TODO check if we need this
        # self.total_num_patches = 0 #TODO check if we need this

        # register hook for the model
        assert len(self.layers)==1 and len(self.facet)==1, "currently only support one layer one facet."
        self._register_hooks(self.layers, self.facet[0]) #self.layers should be a list, self.facet[0] must be a string


    def finish_grads_computation(self):
        self._unregister_hooks()
    
    def load_model_params(self, model) -> None:
        '''
        we may not need this function
        '''
        self.model = model
        return None

 
    def compute_per_sample_grad(self, batch: Iterable[Tensor]) -> Tensor:
        self.total_num_patches = 0
        batch_grads=[]
        imgs, labels = batch
        imgs, labels = imgs.to(self.device), labels.to(self.device)

        for sample, label in zip(imgs, labels):
            sample, label = sample.unsqueeze(0), label.unsqueeze(0)
            sample_grad = self.compute_per_patch_grad(sample, label)
            if sample_grad.shape[0]==1:
                sample_grad = sample_grad.squeeze(0)

            if not self.include_cls:
                batch_grads.append(sample_grad[1:,:])
        grads = torch.cat(batch_grads,dim=0)
        return grads


    def compute_loss_grad(self, batch: Iterable[Tensor], batch_size: int) -> Tensor:
        return None
    
    def compute_per_patch_grad(self, sample: Tensor, label: Tensor) -> Tensor:

        # TODO check how to use toch.func.grad and torch.func.vmap
        self._extract_features(sample, self.centroids[label].unsqueeze(0), self.facet[0])
        return self._feats[0]



    def _extract_features(self, image: Tensor, centroid_feat: Tensor,  facet: str = 'key') -> List[torch.Tensor]:
        """
        extract features from the model
        :param batch: batch to extract features for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        :return : tensor of features.
                  if facet is 'key' | 'query' | 'value' has shape Bxhxtxd
                  if facet is 'attn' has shape Bxhxtxt
                  if facet is 'token' has shape Bxtxd
        """
        B, C, H, W = image.shape
        self._feats = []
        self.setting_dict = {}
        self._v = []
        self._attn = []
        self._input = []
        self._k = []
        self._q =[]
        
        num_patches = (1 + (H - self.patch_size) // self.stride[0], 1 + (W - self.patch_size) // self.stride[1])
        self.total_num_patches += num_patches[0] * num_patches[1]


        if facet[0]== 'w':
            # batch.requires_grad = True # turn on the requires_grad for the batch
            # self.model.train()
               
            logits = self.model(image) # [1, feature_dim]
            loss = self.dino_loss(logits, centroid_feat)
         
            self.model.zero_grad()
            loss.backward(retain_graph=True)

            self._weight_feats_calc(facet, self._grad)
            # print(f"get {facet} features with its dimension {self._feats[0].shape}")
        else:
            _ = self.model(image)
            # print(f"get {facet} features with its dimension {self._feats[0].shape}") #  B x num_heads x N x C // num_heads


        
        assert self._feats!=[], "no features are extracted"
        return self._feats


    def dino_loss(self, logits, centroid_feat):
        student_out = logits
        teacher_out = F.softmax((centroid_feat) , dim=-1)
        loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)
        return loss
    

    def _register_hooks(self, layers: List[int], facet: str) -> None:
        """
        register hook to extract features.
        :param layers: layers from which to extract features.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        """
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in layers:
                if facet == 'token':
                    self.hook_handlers.append(block.register_forward_hook(self._get_hook(facet)))
                elif facet == 'attn':
                    self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_hook(facet)))
                elif facet in ['key', 'query', 'value']:
                    self.hook_handlers.append(block.attn.register_forward_hook(self._get_hook(facet)))
                elif facet in ['wkey', 'wquery', 'wvalue']:
                    self.hook_handlers.append(block.attn.register_forward_hook(self._get_weight_hook(facet, 'forward')))
                    self.hook_handlers.append(block.attn.proj.register_backward_hook(self._get_weight_hook(facet, 'backward')))
                elif facet == 'wfc1':
                    self.hook_handlers.append(block.mlp.fc1.register_forward_hook(self._get_weight_hook(facet, 'forward')))
                    self.hook_handlers.append(block.mlp.fc1.register_backward_hook(self._get_weight_hook(facet, 'backward')))
                elif facet == 'wfc2':
                    self.hook_handlers.append(block.mlp.fc2.register_forward_hook(self._get_weight_hook(facet, 'forward')))
                    self.hook_handlers.append(block.mlp.fc2.register_backward_hook(self._get_weight_hook(facet, 'backward')))
                else:
                    raise TypeError(f"{facet} is not a supported facet.")


    def _get_weight_hook(self, facet: str, path: str):
        """
        generate a hook method for the weights in a specific block and facet.
        """
        print("register weight hook for the model----------------")
        if path == 'forward':
            '''
            foward hook to get the setting_dict and other necessary values
            '''
            if facet in ['wfc1', 'wfc2']:
                def _inner_hook(module, input, output):
                    self._input.append(input[0])
                return _inner_hook

            else:
                def _inner_hook(module, input, output):
                    input = input[0]
                    B, N, C = input.shape
                    qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)

                    q, k, v = qkv[0], qkv[1], qkv[2]
                    attn = (q @ k.transpose(-2, -1)) #* module.scale
                    # print(attn.shape, "attn.shape")  # shape should be B, nh, N, N

                    self.setting_dict ={
                        'B': B,
                        'N': N,
                        'C': C,
                        'num_heads': module.num_heads,
                        'scale': module.scale
                    }
                    if facet == 'wattn':
                        self._v.append(v)
                    
                    elif facet == 'wvalue':
                        self._attn.append(attn)
                        self._input.append(input)
                    elif facet == 'wkey':
                        self._attn.append(attn)
                        self._v.append(v)
                        self._input.append(input)
                        self._q = q  # B, num_heads, N, C // num_heads
                    elif facet == 'wquery':
                        self._attn.append(attn)
                        self._v.append(v)
                        self._input.append(input)
                        self._k = k # B, num_heads, N, C // num_heads
                    else:
                        raise TypeError(f"{facet} is not a supported facet.")
                return _inner_hook
            
        elif path == 'backward':
            if facet in ['wfc1', 'wfc2']:
                def _inner_hook(module, grad_input, grad_output):

                    self._grad = grad_input[0].unsqueeze(0) # N, C -> 1, N, C
                    

            else:
                def _inner_hook(module, grad_input, grad_output):
                    '''
                    first step: get the graddents of the output with respect to the attention layer
                    '''
                    # https://discuss.pytorch.org/t/what-does-gard-input-and-grad-output-mean/195577
                    # shows that we need grad_input

                    # grad_input[0] shape B, N, C
                    # TODO assign value B, num_heads, N, N
                    grad = grad_input[0].reshape(self.setting_dict["B"],  self.setting_dict["N"], self.setting_dict["num_heads"],
                                                self.setting_dict["C"]// self.setting_dict["num_heads"]) # B, N, C-> B, N, num_heads, C // num_heads
                    grad = grad.permute(0,2,1,3) # B, num_heads, N, C // num_heads
                
                    self._grad = grad
            

            return _inner_hook
        else:
            raise TypeError(f"{path} is not a supported path.")

    def _weight_feats_calc(self, facet: str, grad: torch.Tensor) :
        if facet == 'wattn': 
            grad = grad @ self._v[0].transpose(-2, -1) # B, num_heads, N, C // num_heads @ B, num_heads, C // num_heads, N -> B, num_heads, N, N
            grad_features =  grad.mean(dim=1) # B, N, N as we average over the heads

            print("calculate the gradients on the value matrix")
        elif facet == 'wvalue':
            attn = (self._attn[0]* self.setting_dict["scale"]).softmax(dim=-1)
            # TODO check if we need dropout func here

            # gradients on the value matrix
            grad = attn.transpose(-2, -1) @ grad # B, num_heads, N, N @ B, num_heads, N, C // num_heads -> B, num_heads, N, C // num_heads
            grad = grad.transpose(1,2) # B, N, num_heads, C // num_heads
            grad = grad.reshape(self.setting_dict["B"],self.setting_dict["N"], self.setting_dict["C"]) # B, N, C

            # gradients on the value weight
            grad_features = torch.zeros(self.setting_dict["B"], self.setting_dict["N"], self.setting_dict["C"], self.setting_dict["C"]).to(self.device)
            for i in range(self.setting_dict["N"]):
                # B, C, 1 @ B, 1, C -> B, C, C-> iterate to B, N, C, C 
                grad_features[:,i,:,:] = self._input[0][:,i, :].unsqueeze(-2).transpose(-2,-1)@grad[:,i,:].unsqueeze(-2) 
            grad_features = grad_features.reshape(self.setting_dict["B"], self.setting_dict["N"], self.setting_dict["C"]* self.setting_dict["C"])
        elif facet == 'wkey':
            # gradients on the attn
            grad = grad @ self._v[0].transpose(-2, -1) # B, num_heads, N, C // num_heads @ B, num_heads, C // num_heads, N -> B, num_heads, N, N
            # dropout layer scient pass and then softmax layer
            attn_output = torch.nn.Softmax(dim=-1)(self._attn[0])
            grad = torch.autograd.grad(attn_output, self._attn[0], grad, retain_graph=True)[0] # chek retain_graph; size B, num_heads, N, N
            grad = grad*self.setting_dict["scale"]
        
            # print(torch.unique(grad[:,:,1:,1:]), "grad[:,:,1:,1:] for wkey in the previous step")
            # gradients on the key matrix
            grad = self._q[0].transpose(-2,-1)@grad # B, num_heads, C // num_heads, N @ B, num_heads, N, N -> B, num_heads,  C // num_heads, N
            grad = grad.permute(0,3,1,2) # B, N, num_heads, C // num_heads
            # gradient on the key weight, before summing together on each of patches
            grad = grad.reshape(self.setting_dict["B"],self.setting_dict["N"], self.setting_dict["C"]) # B, N, C

            grad_features = torch.zeros(self.setting_dict["B"], self.setting_dict["N"], self.setting_dict["C"], self.setting_dict["C"]).to(self.device)
            for i in range(self.setting_dict["N"]):
                # B, C, 1 @ B, 1, C -> B, C, C-> iterate to B, N, C, C 
                grad_features[:,i,:,:] = self._input[0][:,i, :].unsqueeze(-2).transpose(-2,-1)@grad[:,i,:].unsqueeze(-2) 
            grad_features = grad_features.reshape(self.setting_dict["B"], self.setting_dict["N"], self.setting_dict["C"]* self.setting_dict["C"])

          
            # print("calculate the gradients on the key matrix")
        elif facet == 'wquery':
            # gradients on the attn
            grad = grad @ self._v[0].transpose(-2, -1) # B, num_heads, N, C // num_heads @ B, num_heads, C // num_heads, N -> B, num_heads, N, N
            # dropout layer scient pass and then softmax layer
            attn_output = torch.nn.Softmax(dim=-1)(self._attn[0])
            grad = torch.autograd.grad(attn_output, self._attn[0], grad, retain_graph=True)[0] # chek retain_graph; size B, num_heads, N, N
            grad = grad*self.setting_dict["scale"]

            print(torch.unique(grad[:,:,1:,1:]), "grad[:,:,1:,1:] for wquery in the previous step")

            # gradients on the query matrix
            grad = grad @ self._k[0] # B, num_heads, N, N  @ B, num_heads,  N, C // num_heads -> B, num_heads, N, C // num_heads
            grad = grad.permute(0,2,1,3) # B, N, num_heads, C // num_heads
            # gradient on the query weight, before summing together on each of patches
            grad = grad.reshape(self.setting_dict["B"],self.setting_dict["N"], self.setting_dict["C"]) # B, N, C

            print(torch.unique(grad[:,1:,:]), "grad[:,1:,:] for wquery")
           

            grad_features = torch.zeros(self.setting_dict["B"], self.setting_dict["N"], self.setting_dict["C"], self.setting_dict["C"]).to(self.device)
            for i in range(self.setting_dict["N"]):
                # B, C, 1 @ B, 1, C -> B, C, C-> iterate to B, N, C, C 
                grad_features[:,i,:,:] = self._input[0][:,i, :].unsqueeze(-2).transpose(-2,-1)@grad[:,i,:].unsqueeze(-2)
            grad_features = grad_features.reshape(self.setting_dict["B"], self.setting_dict["N"], self.setting_dict["C"]* self.setting_dict["C"])
         
            
            print("calculate the gradients on the query matrix")       
        
        elif facet in ['wfc1', 'wfc2']:
            B, N, C = self._input[0].shape
            C_out = grad.shape[-1]
            # grad_features = torch.zeros(B, N, C, C_out).to(self.device)

            # for i in range (N):
            #     grad_features[:,i,:,:] = self._input[0][:,i,:].unsqueeze(-2).transpose(-2,-1)@grad[:,i,:].unsqueeze(-2)
            # grad_features = grad_features.reshape(B, N, C*C_out)
            grad_features =  grad
        else:
            raise TypeError(f"{facet} is not a supported facet.")
        self._feats.append(grad_features.detach())                

    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []



class FunctionalGradientComputer(AbstractGradientComputer):
    def __init__(
        self,
        model: torch.nn.Module,
        task: AbstractModelOutput,
        grad_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        grad_wrt: Optional[Iterable[str]] = None,
    ) -> None:
        """Initializes attributes, and loads model parameters.

        Args:
            grad_wrt (list[str], optional):
                A list of parameter names for which to keep gradients.  If None,
                gradients are taken with respect to all model parameters.
                Defaults to None.
        """
        super().__init__(model, task, grad_dim, dtype, device)
        self.model = model
        self.num_params = get_num_params(self.model)
        self.load_model_params(model)
        self.grad_wrt = grad_wrt
        self.logger = logging.getLogger("GradientComputer")

    def load_model_params(self, model) -> None:
        """Given a a torch.nn.Module model, inits/updates the (functional)
        weights and buffers. See https://pytorch.org/docs/stable/func.html
        for more details on :code:`torch.func`'s functional models.

        Args:
            model (torch.nn.Module):
                model to load

        """
        self.func_weights = dict(model.named_parameters())
        self.func_buffers = dict(model.named_buffers())

    def compute_per_sample_grad(self, batch: Iterable[Tensor]) -> Tensor:
        """Uses functorch's :code:`vmap` (see
        https://pytorch.org/functorch/stable/generated/functorch.vmap.html#functorch.vmap
        for more details) to vectorize the computations of per-sample gradients.

        Doesn't use :code:`batch_size`; only added to follow the abstract method
        signature.

        Args:
            batch (Iterable[Tensor]):
                batch of data

        Returns:
            dict[Tensor]:
                A dictionary where each key is a parameter name and the value is
                the gradient tensor for that parameter.

        """
        # taking the gradient wrt weights (second argument of get_output, hence argnums=1)
        grads_loss = torch.func.grad(
            self.modelout_fn.get_output, has_aux=False, argnums=1
        )

        # map over batch dimensions (hence 0 for each batch dimension, and None for model params)
        grads = torch.func.vmap(
            grads_loss,
            in_dims=(None, None, None, *([0] * len(batch))),
            randomness="different",
        )(self.model, self.func_weights, self.func_buffers, *batch)

        if self.grad_wrt is not None:
            for param_name in list(grads.keys()):
                if param_name not in self.grad_wrt:
                    del grads[param_name]
        return grads

    def compute_loss_grad(self, batch: Iterable[Tensor]) -> Tensor:
        """Computes the gradient of the loss with respect to the model output

        .. math::

            \\partial \\ell / \\partial \\text{(model output)}

        Note: For all applications we considered, we analytically derived the
        out-to-loss gradient, thus avoiding the need to do any backward passes
        (let alone per-sample grads). If for your application this is not feasible,
        you'll need to subclass this and modify this method to have a structure
        similar to the one of :meth:`FunctionalGradientComputer:.get_output`,
        i.e. something like:

        .. code-block:: python

            grad_out_to_loss = grad(self.model_out_to_loss_grad, ...)
            grads = vmap(grad_out_to_loss, ...)
            ...

        Args:
            batch (Iterable[Tensor]):
                batch of data

        Returns:
            Tensor:
                The gradient of the loss with respect to the model output.
        """
        return self.modelout_fn.get_out_to_loss_grad(
            self.model, self.func_weights, self.func_buffers, batch
        )


class IterativeGradientComputer(AbstractGradientComputer):
    def __init__(
        self,
        model,
        task: AbstractModelOutput,
        grad_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        grad_wrt: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__(model, task, grad_dim, dtype, device)
        self.load_model_params(model)
        self.grad_wrt = grad_wrt
        self.logger = logging.getLogger("GradientComputer")
        if self.grad_wrt is not None:
            self.logger.warning(
                "IterativeGradientComputer: ignoring grad_wrt argument."
            )

    def load_model_params(self, model) -> Tensor:
        self.model = model
        self.model_params = list(self.model.parameters())

    def compute_per_sample_grad(self, batch: Iterable[Tensor]) -> Tensor:
        """Computes per-sample gradients of the model output function This
        method does not leverage vectorization (and is hence much slower than
        its equivalent in :class:`.FunctionalGradientComputer`). We recommend
        that you use this only if :code:`torch.func` is not available to you,
        e.g. if you have a (very) old version of pytorch.
        Args:
            batch (Iterable[Tensor]):
                batch of data
        Returns:
            Tensor:
                gradients of the model output function of each sample in the
                batch with respect to the model's parameters.
        """
        batch_size = batch[0].shape[0]
        grads = ch.zeros(batch_size, self.grad_dim).to(batch[0].device)

        margin = self.modelout_fn.get_output(self.model, None, None, *batch)
        for ind in range(batch_size):
            grads[ind] = parameters_to_vector(
                ch.autograd.grad(margin[ind], self.model_params, retain_graph=True)
            )
        return grads

    def compute_loss_grad(self, batch: Iterable[Tensor]) -> Tensor:
        """Computes the gradient of the loss with respect to the model output
        .. math::
            \\partial \\ell / \\partial \\text{(model output)}
        Note: For all applications we considered, we analytically derived the
        out-to-loss gradient, thus avoiding the need to do any backward passes
        (let alone per-sample grads). If for your application this is not feasible,
        you'll need to subclass this and modify this method to have a structure
        similar to the one of :meth:`.IterativeGradientComputer.get_output`,
        i.e. something like:
        .. code-block:: python
            out_to_loss = self.model_out_to_loss(...)
            for ind in range(batch_size):
                grads[ind] = torch.autograd.grad(out_to_loss[ind], ...)
            ...
        Args:
            batch (Iterable[Tensor]):
                batch of data

        Returns:
            Tensor:
                The gradient of the loss with respect to the model output.
        """
        return self.modelout_fn.get_out_to_loss_grad(self.model, None, None, batch)
