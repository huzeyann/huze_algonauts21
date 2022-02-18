import logging
from typing import Any, Callable, Dict, Optional
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BaseFinetuning
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks.finetuning import multiplicative
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.nn import Module
from torch.optim.optimizer import Optimizer

log = logging.getLogger(__name__)


class ReduceAuxLossWeight(Callback):
    r"""
    Monitor a metric and reduce aux loss weight.

    Args:
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute
            change of less than `min_delta`, will count as no improvement.
        patience: number of checks with no improvement
            after which training will be stopped. Under the default configuration, one check happens after
            every training epoch. However, the frequency of validation can be modified by setting various parameters on
            the ``Trainer``, for example ``check_val_every_n_epoch`` and ``val_check_interval``.

            .. note::

                It must be noted that the patience parameter counts the number of validation checks with
                no improvement, and not the number of training epochs. Therefore, with parameters
                ``check_val_every_n_epoch=10`` and ``patience=3``, the trainer will perform at least 40 training
                epochs before being stopped.

        verbose: verbosity mode.
        mode: one of ``'min'``, ``'max'``. In ``'min'`` mode, training will stop when the quantity
            monitored has stopped decreasing and in ``'max'`` mode it will stop when the quantity
            monitored has stopped increasing.
        strict: whether to crash the training if `monitor` is not found in the validation metrics.
        check_finite: When set ``True``, stops training when the monitor becomes NaN or infinite.
        stopping_threshold: Stop training immediately once the monitored quantity reaches this threshold.
        divergence_threshold: Stop training as soon as the monitored quantity becomes worse than this threshold.
        check_on_train_epoch_end: whether to run early stopping at the end of the training epoch.
            If this is ``False``, then the check runs at the end of the validation epoch.

    Raises:
        MisconfigurationException:
            If ``mode`` is none of ``"min"`` or ``"max"``.
        RuntimeError:
            If the metric ``monitor`` is not available.

    Example::

        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import EarlyStopping
        >>> early_stopping = EarlyStopping('val_loss')
        >>> trainer = Trainer(callbacks=[early_stopping])
    """
    mode_dict = {
        'min': torch.lt,
        'max': torch.gt,
    }

    order_dict = {
        'min': "<",
        'max': ">",
    }

    def __init__(
            self,
            monitor: str = 'reduce_lw_on',
            aux_name: str = 'lw_on',
            reduce_ratio=0.01,
            min_delta: float = 0.0,
            patience: int = 3,
            reduce_max_counts: int = 2,
            verbose: bool = False,
            mode: str = 'min',
            strict: bool = True,
            check_finite: bool = True,
            stopping_threshold: Optional[float] = None,
            divergence_threshold: Optional[float] = None,
            check_on_train_epoch_end: bool = False,
    ):
        super().__init__()
        self.reduce_max_counts = reduce_max_counts
        self.reduce_ratio = reduce_ratio
        self.aux_name = aux_name
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.strict = strict
        self.check_finite = check_finite
        self.stopping_threshold = stopping_threshold
        self.divergence_threshold = divergence_threshold
        self.wait_count = 0
        self.reduced_count = 0
        self.reduced_epoch = 0
        self._check_on_train_epoch_end = check_on_train_epoch_end

        if self.mode not in self.mode_dict:
            raise MisconfigurationException(f"`mode` can be {', '.join(self.mode_dict.keys())}, got {self.mode}")

        self.min_delta *= 1 if self.monitor_op == torch.gt else -1
        torch_inf = torch.tensor(np.Inf)
        self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf

    def reset_score(self):
        torch_inf = torch.tensor(np.Inf)
        self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf

    def _validate_condition_metric(self, logs):
        monitor_val = logs.get(self.monitor)

        error_msg = (
            f'Early stopping conditioned on metric `{self.monitor}` which is not available.'
            ' Pass in or modify your `EarlyStopping` callback to use any of the following:'
            f' `{"`, `".join(list(logs.keys()))}`'
        )

        if monitor_val is None:
            if self.strict:
                raise RuntimeError(error_msg)
            if self.verbose > 0:
                rank_zero_warn(error_msg, RuntimeWarning)

            return False

        return True

    @property
    def monitor_op(self) -> Callable:
        return self.mode_dict[self.mode]

    def on_save_checkpoint(self, trainer, pl_module, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'wait_count': self.wait_count,
            'reduced_epoch': self.reduced_epoch,
            'best_score': self.best_score,
            'patience': self.patience
        }

    def on_load_checkpoint(self, callback_state: Dict[str, Any]) -> None:
        self.wait_count = callback_state['wait_count']
        self.reduced_epoch = callback_state['reduced_epoch']
        self.best_score = callback_state['best_score']
        self.patience = callback_state['patience']

    def _should_skip_check(self, trainer) -> bool:
        from pytorch_lightning.trainer.states import TrainerFn
        return trainer.state.fn != TrainerFn.FITTING or trainer.sanity_checking

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if not self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_early_stopping_check(trainer, pl_module)

    def on_validation_end(self, trainer, pl_module) -> None:
        if self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return

        self._run_early_stopping_check(trainer, pl_module)

    def _run_early_stopping_check(self, trainer, pl_module) -> None:
        """
        Checks whether the early stopping condition is met
        and if so tells the trainer to stop the training.
        """
        logs = trainer.callback_metrics

        if (
                trainer.fast_dev_run  # disable early_stopping with fast_dev_run
                or not self._validate_condition_metric(logs)  # short circuit if metric not present
        ):
            return

        current = logs.get(self.monitor)

        # when in dev debugging
        trainer.dev_debugger.track_early_stopping_history(self, current)

        should_stop, reason = self._evalute_stopping_criteria(current, trainer)

        # stop every ddp process if any world process decides to stop
        should_stop = trainer.training_type_plugin.reduce_boolean_decision(should_stop)
        if should_stop:
            if self.reduced_count > self.reduce_max_counts:
                pl_module.aux_loss_weights[self.aux_name] = 0
            else:
                pl_module.aux_loss_weights[self.aux_name] = pl_module.aux_loss_weights[self.aux_name] * self.reduce_ratio
            # print(self.aux_name, "should stop")
        # else:
            # print(self.aux_name, "should not stop")
        if should_stop:
            self.reduced_epoch = trainer.current_epoch
        if reason and self.verbose:
            self._log_info(trainer, reason)

    def _evalute_stopping_criteria(self, current: torch.Tensor, trainer: 'pl.Trainer') -> Tuple[bool, str]:
        should_stop = False
        reason = None
        if self.check_finite and not torch.isfinite(current):
            should_stop = True
            reason = (
                f"Monitored metric {self.monitor} = {current} is not finite."
                f" Previous best value was {self.best_score:.3f}. Signaling Trainer to stop."
            )
        elif self.stopping_threshold is not None and self.monitor_op(current, self.stopping_threshold):
            should_stop = True
            reason = (
                "Stopping threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.stopping_threshold}."
                " Signaling Trainer to stop."
            )
        elif self.divergence_threshold is not None and self.monitor_op(-current, -self.divergence_threshold):
            should_stop = True
            reason = (
                "Divergence threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.divergence_threshold}."
                " Signaling Trainer to stop."
            )
        elif self.monitor_op(current - self.min_delta, self.best_score.to(trainer.lightning_module.device)):
            should_stop = False
            reason = self._improvement_message(current)
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                should_stop = True
                reason = (
                    f"Monitored metric {self.monitor} did not improve in the last {self.wait_count} records."
                    f" Best score: {self.best_score:.3f}. Signaling Trainer to stop."
                )
                self.wait_count = 0
                self.reduced_count += 1
                self.min_delta *= self.reduce_ratio
                self.patience *= 2
                self.reset_score()

        return should_stop, reason

    def _improvement_message(self, current: torch.Tensor) -> str:
        """ Formats a log message that informs the user about an improvement in the monitored score. """
        if torch.isfinite(self.best_score):
            msg = (
                f"Metric {self.monitor} improved by {abs(self.best_score - current):.3f} >="
                f" min_delta = {abs(self.min_delta)}. New best score: {current:.3f}"
            )
        else:
            msg = f"Metric {self.monitor} improved. New best score: {current:.3f}"
        return msg

    @staticmethod
    def _log_info(trainer: Optional["pl.Trainer"], message: str) -> None:
        if trainer is not None and trainer.world_size > 1:
            log.info(f"[rank: {trainer.global_rank}] {message}")
        else:
            log.info(message)



class HalfScoreFinetuning(BaseFinetuning):
    r"""

    Finetune a backbone model based on a learning rate user-defined scheduling.
    When the backbone learning rate reaches the current model learning rate
    and ``should_align`` is set to True, it will align with it for the rest of the training.

    Args:

        unfreeze_backbone_at_epoch: Epoch at which the backbone will be unfreezed.

        lambda_func: Scheduling function for increasing backbone learning rate.

        backbone_initial_ratio_lr:
            Used to scale down the backbone learning rate compared to rest of model

        backbone_initial_lr: Optional, Inital learning rate for the backbone.
            By default, we will use current_learning /  backbone_initial_ratio_lr

        should_align: Wheter to align with current learning rate when backbone learning
            reaches it.

        initial_denom_lr: When unfreezing the backbone, the intial learning rate will
            current_learning_rate /  initial_denom_lr.

        train_bn: Wheter to make Batch Normalization trainable.

        verbose: Display current learning rate for model and backbone

        round: Precision for displaying learning rate

    Example::

        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import BackboneFinetuning
        >>> multiplicative = lambda epoch: 1.5
        >>> backbone_finetuning = BackboneFinetuning(200, multiplicative)
        >>> trainer = Trainer(callbacks=[backbone_finetuning])

    """

    def __init__(
        self,
        unfreeze_backbone_at_val_score: float = 0.,
        lambda_func: Callable = multiplicative,
        backbone_initial_ratio_lr: float = 10e-2,
        backbone_initial_lr: Optional[float] = None,
        should_align: bool = True,
        initial_denom_lr: float = 10.,
        train_bn: bool = True,
        verbose: bool = False,
        round: int = 12,
    ):
        super().__init__()

        self.unfreeze_backbone_at_val_score = unfreeze_backbone_at_val_score
        self.backbone_initial_lr = backbone_initial_lr
        self.lambda_func = lambda_func
        self.backbone_initial_ratio_lr = backbone_initial_ratio_lr
        self.should_align = should_align
        self.initial_denom_lr = initial_denom_lr
        self.train_bn = train_bn
        self.round = round
        self.verbose = verbose

    def on_fit_start(self, trainer, pl_module):
        """
        Raises:
            MisconfigurationException:
                If LightningModule has no nn.Module `backbone` attribute.
        """
        if hasattr(pl_module, "backbone") and isinstance(pl_module.backbone, Module):
            return
        raise MisconfigurationException("The LightningModule should have a nn.Module `backbone` attribute")

    def freeze_before_training(self, pl_module: 'pl.LightningModule'):
        self.freeze(pl_module.backbone)

    def finetune_function(self, pl_module: 'pl.LightningModule', epoch: int, optimizer: Optimizer, opt_idx: int):
        """Called when the epoch begins."""
        if pl_module.current_val_score >= self.unfreeze_backbone_at_val_score:
            current_lr = optimizer.param_groups[0]['lr']
            initial_backbone_lr = self.backbone_initial_lr if self.backbone_initial_lr is not None \
                else current_lr * self.backbone_initial_ratio_lr
            self.previous_backbone_lr = initial_backbone_lr
            self.unfreeze_and_add_param_group(
                pl_module.backbone,
                optimizer,
                initial_backbone_lr,
                train_bn=self.train_bn,
                initial_denom_lr=self.initial_denom_lr
            )
            if self.verbose:
                log.info(
                    f"Current lr: {round(current_lr, self.round)}, "
                    f"Backbone lr: {round(initial_backbone_lr, self.round)}"
                )

        elif pl_module.current_val_score > self.unfreeze_backbone_at_val_score:
            current_lr = optimizer.param_groups[0]['lr']
            next_current_backbone_lr = self.lambda_func(epoch + 1) * self.previous_backbone_lr
            next_current_backbone_lr = current_lr if (self.should_align and next_current_backbone_lr > current_lr) \
                else next_current_backbone_lr
            optimizer.param_groups[-1]["lr"] = next_current_backbone_lr
            self.previous_backbone_lr = next_current_backbone_lr
            if self.verbose:
                log.info(
                    f"Current lr: {round(current_lr, self.round)}, "
                    f"Backbone lr: {round(next_current_backbone_lr, self.round)}"
                )