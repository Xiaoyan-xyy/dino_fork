'''
mimic trak.py to implement the patch trak
'''
from .modelout_functions import AbstractModelOutput, TASK_TO_MODELOUT
from .projectors import (
    ProjectionType,
    AbstractProjector,
    CudaProjector,
    BasicProjector,
    ChunkedCudaProjector,
)
from .gradient_computers import FunctionalGradientComputer, AbstractGradientComputer, PatchGradientComputer
from .score_computers import AbstractScoreComputer, BasicScoreComputer
from .savers import AbstractSaver, MmapSaver, ModelIDException
from .utils import get_num_params, get_parameter_chunk_sizes

from typing import Iterable, Optional, Union
from pathlib import Path
from tqdm import tqdm
from torch import Tensor

import logging
import numpy as np
import torch
import sys
ch = torch



class patchTRAKer:
    def __init__(
        self,
        model: torch.nn.Module,
        patch_size : int,
        stride : tuple,
        task: Union[AbstractModelOutput, str],
        train_set_size: int ,
        centroids: torch.Tensor,
        save_dir: str = "./trak_results",
        load_from_save_dir: bool = True,
        device: Union[str, torch.device] = "cuda",
        gradient_computer: AbstractGradientComputer = PatchGradientComputer,
        projector: Optional[AbstractProjector] = None,
        saver: Optional[AbstractSaver] = None,
        score_computer: Optional[AbstractScoreComputer] = None,
        proj_dim: int = 2048,
        logging_level=logging.INFO,
        use_half_precision: bool = True,
        proj_max_batch_size: int = 32,
        projector_seed: int = 0,
        grad_wrt: Optional[Iterable[str]] = None,
        facet : Optional[Iterable[str]] = None,
        include_cls: bool = False,
        lambda_reg: float = 0.0,
        total_num_patches : int = 0    
        ) -> None:
        
        self.model = model
        self.task = task
        self.train_set_size = train_set_size
        self.device = device
        self.dtype = ch.float16 if use_half_precision else ch.float32
        self.grad_wrt = grad_wrt
        self.facet = facet
        self.lambda_reg = lambda_reg
        self.total_num_patches = total_num_patches
        self.include_cls = include_cls
        print(f"total_num_patches: {total_num_patches}")
       

        logging.basicConfig()
        self.logger = logging.getLogger("patchTRAK")
        self.logger.setLevel(logging_level)

        # set which part of weights gradients should be calcualted
        # should be changed to layer index in the vision transformer
        self.num_params = get_num_params(self.model)
        if self.grad_wrt is not None:
            d = dict(self.model.named_parameters())
            self.num_params_for_grad = sum(
                [d[param_name].numel() for param_name in self.grad_wrt]
            )

            print(self.num_params_for_grad, f"before constranit by {facet}")
            assert len(self.facet)>=1 and len(self.facet)<=3, "facet should be a list of length 1, 2 or 3"
            # TODO currently only support attn.qkv.weight
            self.num_params_for_grad = self.num_params_for_grad // 3 *len(self.facet)
            print(self.num_params_for_grad, f"when constranit by {facet}")
        else:
            self.num_params_for_grad = self.num_params


        # inits self.projector
        self.proj_seed = projector_seed
        self.init_projector(
            projector=projector,
            proj_dim=proj_dim,
            proj_max_batch_size=proj_max_batch_size,
        )

        # normalize to make X^TX numerically stable
        # doing this instead of normalizing the projector matrix
        self.normalize_factor = ch.sqrt(
            ch.tensor(self.num_params_for_grad, dtype=ch.float32)
        )

        self.save_dir = Path(save_dir).resolve()
        self.load_from_save_dir = load_from_save_dir # TODO can be ignored or not

        if type(self.task) is str:
            self.task = TASK_TO_MODELOUT[self.task]()

        
        self.gradient_computer = gradient_computer(
            model=self.model,
            patch_size = patch_size,
            stride = stride,
            task=self.task,
            centroids=centroids,
            grad_dim=self.num_params_for_grad,
            dtype=self.dtype,
            device=self.device,
            grad_wrt=self.grad_wrt,
            facet = self.facet,
            include_cls=self.include_cls,
        )

        # Class to use for computing the final TRAK scores. If None, the class:`.BasicScoreComputer`
        # TODO we may not need to do any changes for it
        if score_computer is None:
            score_computer = BasicScoreComputer
        self.score_computer = score_computer(
            dtype=self.dtype,
            device=self.device,
            logging_level=logging_level,
            lambda_reg=self.lambda_reg,
        )

        metadata = {
            "JL dimension": self.proj_dim,
            "JL matrix type": self.projector.proj_type,
            "train set size": self.train_set_size,
        }

        if saver is None:
            saver = MmapSaver
        self.saver = saver(
            save_dir=self.save_dir,
            metadata=metadata,
            train_set_size=self.train_set_size,
            total_num_patches = self.total_num_patches,
            proj_dim=self.proj_dim,
            load_from_save_dir=self.load_from_save_dir,
            logging_level=logging_level,
            use_half_precision=use_half_precision,
        )

        
        # check if we need this TODO
        self.ckpt_loaded = "no ckpt loaded"


    # this part may be don't need any changes TODO check
    def init_projector(
        self,
        projector: Optional[AbstractProjector],
        proj_dim: int,
        proj_max_batch_size: int,
    ) -> None:
        """Initialize the projector for a traker class

        Args:
            projector (Optional[AbstractProjector]):
                JL projector to use. If None, a CudaProjector will be used (if
                possible).
            proj_dim (int):
                Dimension of the projected gradients and TRAK features.
            proj_max_batch_size (int):
                Batch size used by fast_jl if the CudaProjector is used. Must be
                a multiple of 8. The maximum batch size is 32 for A100 GPUs, 16
                for V100 GPUs, 40 for H100 GPUs.
        """

        self.projector = projector
        if projector is not None:
            self.proj_dim = self.projector.proj_dim
            if self.proj_dim == 0:  # using NoOpProjector
                self.proj_dim = self.num_params_for_grad

        else:
            using_cuda_projector = False
            self.proj_dim = proj_dim
            if self.device == "cpu":
                self.logger.info("Using BasicProjector since device is CPU")
                projector = BasicProjector
                # Sampling from bernoulli distribution is not supported for
                # dtype float16 on CPU; playing it safe here by defaulting to
                # normal projection, rather than rademacher
                proj_type = ProjectionType.normal
                self.logger.info("Using Normal projection")
            else:
                try:
                    import fast_jl

                    test_gradient = ch.ones(1, self.num_params_for_grad).cuda()
                    num_sms = ch.cuda.get_device_properties(
                        "cuda"
                    ).multi_processor_count
                    fast_jl.project_rademacher_8(
                        test_gradient, self.proj_dim, 0, num_sms
                    )
                    projector = CudaProjector
                    using_cuda_projector = True

                except (ImportError, RuntimeError, AttributeError) as e:
                    self.logger.error(f"Could not use CudaProjector.\nReason: {str(e)}")
                    self.logger.error("Defaulting to BasicProjector.")
                    projector = BasicProjector
                proj_type = ProjectionType.rademacher

            if using_cuda_projector:
                max_chunk_size, param_chunk_sizes = get_parameter_chunk_sizes(
                    self.model, proj_max_batch_size
                )
                self.logger.debug(
                    (
                        f"the max chunk size is {max_chunk_size}, ",
                        "while the model has the following chunk sizes",
                        f"{param_chunk_sizes}.",
                    )
                )

                if (
                    len(param_chunk_sizes) > 1
                ):  # we have to use the ChunkedCudaProjector
                    self.logger.info(
                        (
                            f"Using ChunkedCudaProjector with"
                            f"{len(param_chunk_sizes)} chunks of sizes"
                            f"{param_chunk_sizes}."
                        )
                    )
                    rng = np.random.default_rng(self.proj_seed)
                    seeds = rng.integers(
                        low=0,
                        high=500,
                        size=len(param_chunk_sizes),
                    )
                    projector_per_chunk = [
                        projector(
                            grad_dim=chunk_size,
                            proj_dim=self.proj_dim,
                            seed=seeds[i],
                            proj_type=ProjectionType.rademacher,
                            max_batch_size=proj_max_batch_size,
                            dtype=self.dtype,
                            device=self.device,
                        )
                        for i, chunk_size in enumerate(param_chunk_sizes)
                    ]
                    self.projector = ChunkedCudaProjector(
                        projector_per_chunk,
                        max_chunk_size,
                        param_chunk_sizes,
                        proj_max_batch_size,
                        self.device,
                        self.dtype,
                    )
                    return  # do not initialize projector below

            self.logger.debug(
                f"Initializing projector with grad_dim {self.num_params_for_grad}"
            )
            self.projector = projector(
                grad_dim=self.num_params_for_grad,
                proj_dim=self.proj_dim,
                seed=self.proj_seed,
                proj_type=proj_type,
                max_batch_size=proj_max_batch_size,
                dtype=self.dtype,
                device=self.device,
            )
            self.logger.debug(f"Initialized projector with proj_dim {self.proj_dim}")

    def load_checkpoint(
        self,
        checkpoint: Iterable[Tensor],
        model_id: int,
        _allow_featurizing_already_registered=False,
    ) -> None:
        """Loads state dictionary for the given checkpoint; initializes arrays
        to store TRAK features for that checkpoint, tied to the model ID.

        Args:
            checkpoint (Iterable[Tensor]):
                state_dict to load
            model_id (int):
                a unique ID for a checkpoint
            _allow_featurizing_already_registered (bool, optional):
                Only use if you want to override the default behaviour that
                :code:`featurize` is forbidden on already registered model IDs.
                Defaults to None.

        """
        print(self.saver.model_ids,"self.saver.model_ids in traker.load_checkpoint")
        if self.saver.model_ids.get(model_id) is None:
            self.saver.register_model_id(
                model_id, _allow_featurizing_already_registered
            )
        else:
            self.saver.load_current_store(model_id)
       
        # check if the following line is necessary as we already load the checkpoint before initializing the TRAKer
        # self.model.load_state_dict(checkpoint)
        self.model.eval()


        ############################################
        # initialize self.func_weights and self.func_buffers for self.gradient_computer
        # self.func_weights = dict(model.named_parameters())
        # self.func_buffers = dict(model.named_buffers())    
        # or
        # initialize self.model and self.model_params for self.gradient_computer
        # self.model = model
        # self.model_params = list(self.model.parameters())
        self.gradient_computer.load_model_params(self.model)
        ############################################

        self._last_ind = 0
        self.ckpt_loaded = model_id

    def featurize(
        self,
        batch: Iterable[Tensor],
        num_pacthes_in_batch: Optional[int] = None,
    ) -> None:
        """Creates TRAK features for the given batch by computing the gradient
        of the model output function and projecting it. In the notation of the
        paper, for an input pair :math:`z=(x,y)`, model parameters
        :math:`\\theta`, and JL projection matrix :math:`P`, this method
        computes :math:`P^\\top \\nabla_\\theta f(z_i, \\theta)`.
        Additionally, this method computes the gradient of the out-to-loss
        function (in the notation of the paper, the :math:`Q` term in Section
        3.4).

        Either :code:`inds` or :code:`num_samples` must be specified. Using
        :code:`num_samples` will write sequentially into the internal store of
        the :func:`TRAKer`.

        Args:
            batch (Iterable[Tensor]):
                input batch
         
            num_samples (Optional[int], optional):
                Number of samples in the batch. Defaults to None.

        """
        torch.cuda.empty_cache()
        assert (
            self.ckpt_loaded == self.saver.current_model_id
        ), "Load a checkpoint using traker.load_checkpoint before featurizing"
       
       
        inds = np.arange(self._last_ind, self._last_ind + num_pacthes_in_batch)
        self._last_ind += num_pacthes_in_batch
       

        # handle re-starting featurizing from a partially featurized model (some inds already featurized)
        _already_done = (self.saver.current_store["is_featurized"][inds] == 1).reshape(-1)

        # TODO set "is_featurized" to be counted by sample number not patch number?
        inds = inds[~_already_done]
        if len(inds) == 0:
            self.logger.debug("All samples in batch already featurized.")
            return 0

        
        grads = self.gradient_computer.compute_per_sample_grad(batch = batch)
       
      
        grads = self.projector.project(grads, model_id=self.saver.current_model_id)
        grads /= self.normalize_factor

        self.saver.current_store["grads"][inds] = (
            grads.to(self.dtype).cpu().clone().detach()
        )

        # we don't need "out_to_loss"
        # loss_grads = self.gradient_computer.compute_loss_grad(batch)
        # self.saver.current_store["out_to_loss"][inds] = (
        #     loss_grads.to(self.dtype).cpu().clone().detach()
        # )

        self.saver.current_store["is_featurized"][inds] = 1
        self.saver.serialize_current_model_id_metadata() 
        print(f'finishing one batch with {num_pacthes_in_batch} patches')

        

    def finalize_features(
        self, model_ids: Iterable[int] = None, del_grads: bool = False
    ) -> None:
        """For a set of checkpoints :math:`C` (specified by model IDs), and
        gradients :math:`\\{ \\Phi_c \\}_{c\\in C}`, this method computes
        :math:`\\Phi_c (\\Phi_c^\\top\\Phi_c)^{-1}` for all :math:`c\\in C`
        and stores the results in the internal store of the :func:`TRAKer`
        class.

        Args:
            model_ids (Iterable[int], optional): A list of model IDs for which
                features should be finalized. If None, features are finalized
                for all model IDs in the :code:`save_dir` of the :class:`.TRAKer`
                class. Defaults to None.

        """
        # unregister the model hooks
        self.gradient_computer.finish_grads_computation()
        # this method is memory-intensive, so we're freeing memory beforehand
        torch.cuda.empty_cache()
        self.projector.free_memory()

        if model_ids is None:
            model_ids = list(self.saver.model_ids.keys())

        self._last_ind = 0

        for model_id in tqdm(model_ids, desc="Finalizing features for all model IDs.."):
            if self.saver.model_ids.get(model_id) is None:
                raise ModelIDException(
                    f"Model ID {model_id} not registered, not ready for finalizing."
                )
            elif self.saver.model_ids[model_id]["is_featurized"] == 0:
                raise ModelIDException(
                    f"Model ID {model_id} not fully featurized, not ready for finalizing."
                )
            elif self.saver.model_ids[model_id]["is_finalized"] == 1:
                self.logger.warning(
                    f"Model ID {model_id} already finalized, skipping .finalize_features for it."
                )
                continue

            self.saver.load_current_store(model_id)

            
            print("Loading the gradients of trainig patches to cpu")
            g = ch.as_tensor(self.saver.current_store["grads"], device=torch.device("cpu"))
            print(g.shape, "g.shape")
            xtx = self.score_computer.get_xtx(g)
            print(xtx.shape, "xtx.shape")

            features = self.score_computer.get_x_xtx_inv(g, xtx) # already on cpu
            # self.saver.current_store["features"][:] = features.to(self.dtype).cpu()
            self.saver.current_store["features"][:] = features.to(self.dtype)
            if del_grads:
                self.saver.del_grads(model_id)

            self.saver.model_ids[self.saver.current_model_id]["is_finalized"] = 1
            self.saver.serialize_current_model_id_metadata()

    
    def start_scoring_checkpoint(
        self,
        exp_name: str,
        checkpoint: Iterable[Tensor],
        model_id: int,
        num_targets: int,
    ) -> None:
        """This method prepares the internal store of the :class:`.TRAKer` class
        to start computing scores for a set of targets.

        Args:
            exp_name (str):
                Here we will set the exp_name = target img name
                Experiment name. Each experiment should have a unique name, and
                it corresponds to a set of targets being scored. The experiment
                name is used as the name for saving the target features, as well
                as scores produced by this method in the :code:`save_dir` of the
                :class:`.TRAKer` class.
            checkpoint (Iterable[Tensor]):
                model checkpoint (state dict)
            model_id (int):
                a unique ID for a checkpoint
            num_targets (int):
                number of targets to score

        """
        print(exp_name, "exp_name", "initialize the experiments")
        self.saver.init_experiment(exp_name, num_targets, model_id)

        
        self.model.eval()
        self.gradient_computer.load_model_params(self.model)

        # TODO: make this exp_name-dependent
        # e.g. make it a value in self.saver.experiments[exp_name]
        self._last_ind_target = 0

    def score_one_sample(
        self,
        sample: Tensor,
        label: Tensor,
        num_patches_in_img: Optional[int] = None,
    ) -> None:
        """This method computes the (intermediate per-checkpoint) TRAK scores
        for a batch of targets and stores them in the internal store of the
        :class:`.TRAKer` class.

        Either :code:`inds` or :code:`num_samples` must be specified. Using
        :code:`num_samples` will write sequentially into the internal store of
        the :class:`.TRAKer`.

        Args:
            batch (Iterable[Tensor]):
                input batch
            inds (Optional[Iterable[int]], optional):
                Indices of the batch samples in the train set. Defaults to None.
            num_samples (Optional[int], optional):
                Number of samples in the batch. Defaults to None.

        """
       
        if self.saver.model_ids[self.saver.current_model_id]["is_finalized"] == 0:
            self.logger.error(
                f"Model ID {self.saver.current_model_id} not finalized, cannot score"
            )
            return None

      
        inds = np.arange(self._last_ind_target, self._last_ind_target + num_patches_in_img)

        # when calculate the gradient on one target image, initialize the total_num_patches to 0
        self.gradient_computer.total_num_patches = 0
        grads = self.gradient_computer.compute_per_patch_grad(sample.to(self.device), label.to(self.device))
        if grads.shape[0]==1:
            grads = grads.squeeze(0)
            
        # remove cls token
        grads = self.projector.project(grads[1:,:], model_id=self.saver.current_model_id)# always exclude the CLS token
        grads /= self.normalize_factor
        print(f"after projection, the shape of grads is {grads.shape}")

        exp_name = self.saver.current_experiment_name
        self.saver.current_store[f"{exp_name}_grads"][inds] = (
            grads.to(self.dtype).cpu().clone().detach()
        )




    def finalize_scores(
        self,
        exp_name: str,
        model_ids: Iterable[int] = None,
        allow_skip: bool = False,
    ) -> Tensor:
        """This method computes the final TRAK scores for the given targets,
        train samples, and model checkpoints (specified by model IDs).

        Args:
            exp_name (str):
                Experiment name. Each experiment should have a unique name, and
                it corresponds to a set of targets being scored. The experiment
                name is used as the name for saving the target features, as well
                as scores produced by this method in the :code:`save_dir` of the
                :class:`.TRAKer` class.
            model_ids (Iterable[int], optional):
                A list of model IDs for which
                scores should be finalized. If None, scores are computed
                for all model IDs in the :code:`save_dir` of the :class:`.TRAKer`
                class. Defaults to None.
            allow_skip (bool, optional):
                If True, raises only a warning, instead of an error, when target
                gradients are not computed for a given model ID. Defaults to
                False.

        Returns:
            Tensor: TRAK scores

        """
        # reset counter for inds used for scoring
        self._last_ind_target = 0

        if model_ids is None:
            model_ids = self.saver.model_ids
        else:
            model_ids = {
                model_id: self.saver.model_ids[model_id] for model_id in model_ids
            }
        assert len(model_ids) > 0, "No model IDs to finalize scores for"

        if self.saver.experiments.get(exp_name) is None:
            raise ValueError(
                f"Experiment {exp_name} does not exist. Create it\n\
                              and compute scores first before finalizing."
            )

        num_targets = self.saver.experiments[exp_name]["num_patch_in_targets"]
        _completed = [False] * len(model_ids)

        self.saver.load_current_store(list(model_ids.keys())[0], exp_name, num_targets)
        _scores_mmap = self.saver.current_store[f"{exp_name}_scores"]
        _scores_on_cpu = ch.zeros(*_scores_mmap.shape, device="cpu")
        if self.device != "cpu":
            _scores_on_cpu.pin_memory()

        # we don't need "out_to_loss"
        # _avg_out_to_losses = np.zeros(
        #     (self.saver.train_set_size, 1),
        #     dtype=np.float16 if self.dtype == ch.float16 else np.float32,
        # )

        for j, model_id in enumerate(
            tqdm(model_ids, desc="Finalizing scores for all model IDs..")
        ):
            self.saver.load_current_store(model_id)
            try:
                self.saver.load_current_store(model_id, exp_name, num_targets)
            except OSError as e:
                if allow_skip:
                    self.logger.warning(
                        f"Could not read target gradients for model ID {model_id}. Skipping."
                    )
                    continue
                else:
                    raise e

            if self.saver.model_ids[self.saver.current_model_id]["is_finalized"] == 0:
                self.logger.warning(
                    f"Model ID {self.saver.current_model_id} not finalized, cannot score"
                )
                continue

            # g = ch.as_tensor(self.saver.current_store["features"], device=self.device)
            g = ch.as_tensor(self.saver.current_store["features"], device=torch.device("cpu"))
            g_target = ch.as_tensor(
                self.saver.current_store[f"{exp_name}_grads"], device=g.device
            )
            if g.device == torch.device("cpu"):
                g , g_target = g.to(torch.float32), g_target.to(torch.float32)
            print(_scores_on_cpu.shape,"_scores_on_cpu.shape")
            self.score_computer.get_scores(g, g_target, accumulator=_scores_on_cpu)
            # .cpu().detach().numpy()

            # _avg_out_to_losses += self.saver.current_store["out_to_loss"]
            _completed[j] = True

        _num_models_used = float(sum(_completed))

        # only write to mmap (on disk) once at the end

        _scores_mmap[:] = (_scores_on_cpu.numpy() / _num_models_used) 

        # _scores_mmap[:] = (_scores_on_cpu.numpy() / _num_models_used) * (
        #     _avg_out_to_losses / _num_models_used
        # )

        self.logger.debug(f"Scores dtype is {_scores_mmap.dtype}")
        self.saver.save_scores(exp_name)
        self.scores = _scores_mmap

        return self.scores

