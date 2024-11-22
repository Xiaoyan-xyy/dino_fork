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
from .gradient_computers import FunctionalGradientComputer, AbstractGradientComputer
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

ch = torch



class patchTRAker:
    def __init__(
        self,
        model: torch.nn.Module,
        task: Union[AbstractModelOutput, str],
        # replace train_set_size: int with train_patch_numbers: int
        train_patch_numbers: int,
        save_dir: str = "./trak_results",
        load_from_save_dir: bool = True,
        device: Union[str, torch.device] = "cuda",
        gradient_computer: AbstractGradientComputer = FunctionalGradientComputer,
        projector: Optional[AbstractProjector] = None,
        saver: Optional[AbstractSaver] = None,
        score_computer: Optional[AbstractScoreComputer] = None,
        proj_dim: int = 2048,
        logging_level=logging.INFO,
        use_half_precision: bool = True,
        proj_max_batch_size: int = 32,
        projector_seed: int = 0,
        grad_wrt: Optional[Iterable[str]] = None,
        lambda_reg: float = 0.0,
    ) -> None:
        
        self.model = model
        self.task = task

        logging.basicConfig()
        self.logger = logging.getLogger("TRAK")
        self.logger.setLevel(logging_level)

        # set which part of weights gradients should be calcualted
        # should be changed to layer index in the vision transformer
        self.num_params = get_num_params(self.model)
        if self.grad_wrt is not None:
            d = dict(self.model.named_parameters())
            self.num_params_for_grad = sum(
                [d[param_name].numel() for param_name in self.grad_wrt]
            )
        else:
            self.num_params_for_grad = self.num_params

        self.normalize_factor = ch.sqrt(
            ch.tensor(self.num_params_for_grad, dtype=ch.float32)
        )

        self.save_dir = Path(save_dir).resolve()
        self.load_from_save_dir = load_from_save_dir # TODO can be ignored or not

        if type(self.task) is str:
            self.task = TASK_TO_MODELOUT[self.task]()

        
        self.gradient_computer = gradient_computer(
            model=self.model,
            task=self.task,
            grad_dim=self.num_params_for_grad,
            dtype=self.dtype,
            device=self.device,
            grad_wrt=self.grad_wrt,
        )

        # Class to use for computing the final TRAK scores. If None, the class:`.BasicScoreComputer`
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
            "train_patch_numbers": self.train_patch_numbers,
        }

        if saver is None:
            saver = MmapSaver
        self.saver = saver(
            save_dir=self.save_dir,
            metadata=metadata,
            train_set_size=self.train_set_size,
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
        # model_id: int,
        # _allow_featurizing_already_registered=False,
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
        # if self.saver.model_ids.get(model_id) is None:
        #     self.saver.register_model_id(
        #         model_id, _allow_featurizing_already_registered
        #     )
        # else:
        #     self.saver.load_current_store(model_id)

        # only have one model checkpoint
        model_id = 0
        self.saver.load_current_store(model_id)

        self.model.load_state_dict(checkpoint)
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
        image: Tensor,
        centroid_feat: Tensor,
        num_patches: int,
        # batch: Iterable[Tensor],
        # inds: Optional[Iterable[int]] = None,
        # num_samples: Optional[int] = None,
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
            inds (Optional[Iterable[int]], optional):
                Indices of the batch samples in the train set. Defaults to None.
            num_samples (Optional[int], optional):
                Number of samples in the batch. Defaults to None.

        """
        # we don't need inds or num_samples as we don't chunk the data into several batches and send them into the model
        # we only have one image
        # TODO calculate the number of batch in this image
        assert (
            self.ckpt_loaded == self.saver.current_model_id
        ), "Load a checkpoint using traker.load_checkpoint before featurizing"
       
        # if num_samples is not None:
        #     inds = np.arange(self._last_ind, self._last_ind + num_samples)
        #     self._last_ind += num_samples
        # else:
        #     num_samples = inds.reshape(-1).shape[0]

        assert image.shape[0]==1, "currently only support one image"
        if centroid_feat.dim() == 1:
            centrpoid_feat = centroid_feat.unsqueeze(0)
        assert image.shape[0] == centroid_feat.shape[0], "image and centroid_feat should have the same batch size"

        inds = np.arange(self._last_ind, self._last_ind + num_patches)
        self._last_ind += num_patches
        
        # # handle re-starting featurizing from a partially featurized model (some inds already featurized)
        # _already_done = (self.saver.current_store["is_featurized"][inds] == 1).reshape(
        #     -1
        # )
        # inds = inds[~_already_done]
        # if len(inds) == 0:
        #     self.logger.debug("All samples in batch already featurized.")
        #     return 0

        grads = self.gradient_computer.compute_per_sample_grad(image=image, centroid_feat=centroid_feat)
        grads = self.projector.project(grads, model_id=self.saver.current_model_id)
        grads /= self.normalize_factor
        self.saver.current_store["grads"][inds] = (
            grads.to(self.dtype).cpu().clone().detach()
        )

        # Warning: Currently since we set out=loss, loss_grad should be an Identity matrix
        loss_grads = self.gradient_computer.compute_loss_grad(image=image, centroid_feat=centroid_feat)
        self.saver.current_store["out_to_loss"][inds] = (
            loss_grads.to(self.dtype).cpu().clone().detach()
        )

        self.saver.current_store["is_featurized"][inds] = 1
        self.saver.serialize_current_model_id_metadata()

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

            g = ch.as_tensor(self.saver.current_store["grads"], device=self.device)
            xtx = self.score_computer.get_xtx(g)

            features = self.score_computer.get_x_xtx_inv(g, xtx)
            self.saver.current_store["features"][:] = features.to(self.dtype).cpu()
            if del_grads:
                self.saver.del_grads(model_id)

            self.saver.model_ids[self.saver.current_model_id]["is_finalized"] = 1
            self.saver.serialize_current_model_id_metadata()