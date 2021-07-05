from argparse import ArgumentParser

import torch
from adabelief_pytorch import AdaBelief
from pytorch_lightning import LightningDataModule
from pytorch_lightning.plugins import DDPPlugin, DataParallelPlugin
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import SGD
from typing import List, Optional
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError

from dataloading import AlgonautsMINIDataModule
from pytorch_lightning.callbacks import BackboneFinetuning
from model_i3d import *
from utils import *
import kornia as K

from torchmetrics.regression import PearsonCorrcoef

from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.metric import Metric
from torch import Tensor
from typing import Any, Optional
from typing import Tuple


# from torchmetrics.utilities import rank_zero_warn
#
#
# def _pearson_corrcoef_update(
#         preds: Tensor,
#         target: Tensor,
#         *_,
# ) -> Tuple[Tensor, Tensor]:
#     """ updates current estimates of the mean, cov and n_obs with new data for calculating pearsons correlation """
#     # Data checking
#     _check_same_shape(preds, target)
#     preds = preds.squeeze()
#     target = target.squeeze()
#     if preds.ndim > 2 or target.ndim > 2:
#         raise ValueError('Expected both predictions and target to be 2 dimensional tensors.')
#
#     return preds, target
#
#
def _pearson_corrcoef_compute(preds: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    """ computes the final pearson correlation based on covariance matrix and number of observatiosn """
    dim = 1
    preds_diff = preds - preds.mean(dim)
    target_diff = target - target.mean(dim)

    cov = (preds_diff * target_diff).mean(dim)
    preds_std = torch.sqrt((preds_diff * preds_diff).mean(dim))
    target_std = torch.sqrt((target_diff * target_diff).mean(dim))

    denom = preds_std * target_std
    # prevent division by zero
    if denom == 0:
        denom += eps

    corrcoef = cov / denom
    return torch.clamp(corrcoef, -1.0, 1.0)


def vectorized_correlation(x, y):
    dim = 0

    centered_x = x - x.mean(dim, keepdims=True)
    centered_y = y - y.mean(dim, keepdims=True)

    covariance = (centered_x * centered_y).sum(dim, keepdims=True)

    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)

    x_std = x.std(dim, keepdims=True) + 1e-8
    y_std = y.std(dim, keepdims=True) + 1e-8

    corr = bessel_corrected_covariance / (x_std * y_std)

    return corr.ravel()


#
# class MulPearsonCorrcoef(Metric):
#     r"""
#     Forward accepts
#
#     - ``preds`` (float tensor): ``(B, N,)``
#     - ``target``(float tensor): ``(B, N,)``
#
#     Args:
#         compute_on_step:
#             Forward only calls ``update()`` and return None if this is set to False. default: True
#         dist_sync_on_step:
#             Synchronize metric state across processes at each ``forward()``
#             before returning the value at the step. default: False
#         process_group:
#             Specify the process group on which synchronization is called. default: None (which selects the entire world)
#
#     """
#
#     def __init__(
#         self,
#         compute_on_step: bool = False,
#         dist_sync_on_step: bool = False,
#         process_group: Optional[Any] = None,
#     ) -> None:
#         super().__init__(
#             compute_on_step=compute_on_step,
#             dist_sync_on_step=dist_sync_on_step,
#             process_group=process_group,
#         )
#
#         rank_zero_warn(
#             'Metric `MulPearsonCorrcoef` will save all targets and predictions in buffer.'
#             ' For large datasets this may lead to large memory footprint.'
#         )
#
#         self.add_state("preds", default=[], dist_reduce_fx="cat")
#         self.add_state("target", default=[], dist_reduce_fx="cat")
#
#     def update(self, preds: Tensor, target: Tensor) -> None:
#         """
#         Update state with predictions and targets.
#
#         Args:
#             preds: Predictions from model
#             target: Ground truth values
#         """
#         preds, target = _pearson_corrcoef_update(preds, target)
#         self.preds.append(preds)
#         self.target.append(target)
#
#     def compute(self) -> Tensor:
#         """
#         Computes pearson correlation coefficient over state.
#         """
#         preds = dim_zero_cat(self.preds)
#         target = dim_zero_cat(self.target)
#         return _pearson_corrcoef_compute(preds, target)
#
#     @property
#     def is_differentiable(self) -> bool:
#         return False


class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self) -> None:
        super().__init__()

        self.transforms = nn.Sequential(
            K.augmentation.RandomHorizontalFlip3D(p=0.5),
            K.augmentation.RandomRotation3D(degrees=15, p=0.5),
            # K.augmentation.RandomAffine3D(p=0.5,
            #                               degrees=15,
            #                               translate=(0.1, 0.1, 0),
            #                               )
        )

    @torch.no_grad()  # disable gradients for efficiency
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = self.transforms(x)  # BxCxDxHxW
        return x_out


class LitI3DFC(LightningModule):

    def __init__(self, backbone, hparams: dict, *args, **kwargs):
        super(LitI3DFC, self).__init__()
        self.save_hyperparameters(hparams)
        self.lr = self.hparams.learning_rate

        # self.train_transform = DataAugmentation()

        self.backbone = backbone
        self.conv31 = nn.Conv3d(1024, hparams['conv_size'], kernel_size=1, stride=1)
        input_dim = hparams['conv_size'] * int(hparams['video_frames'] / 8) * \
                    int(hparams['video_size'] / 16) * int(hparams['video_size'] / 16)
        self.fc = build_fc(hparams, input_dim, hparams['output_size'])

        # self.val_corr = MeanSquaredError(compute_on_step=False)

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument_group("LitModel")
        parser.add_argument('--conv_size', type=int, default=256)
        parser.add_argument('--num_layers', type=int, default=2)
        parser.add_argument('--activation', type=str, default='elu')
        parser.add_argument('--layer_hidden', type=int, default=2048)
        parser.add_argument('--dropout_rate', type=float, default=0.0)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--learning_rate', type=float, default=1e-2)
        return parser

    def forward(self, x):
        x3 = self.backbone(x)
        x3 = self.conv31(x3)
        out = self.fc(x3.reshape(x3.shape[0], -1))
        return out

    def _shared_train_val(self, batch, batch_idx, prefix):
        x, y = batch
        out = self(x)
        loss = F.mse_loss(out, y)
        self.log(f'{prefix}_mse_loss', loss,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return out, loss

    def training_step(self, batch, batch_idx):
        # x, y = batch
        # x = self.train_transform(x)
        # batch = (x, y)

        out, loss = self._shared_train_val(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        out, loss = self._shared_train_val(batch, batch_idx, 'val')
        y = batch[-1]
        # self.val_corr(out[:, 0], y[:, 0])
        return {'out': out, 'y': y}

    def validation_epoch_end(self, val_step_outputs) -> None:
        # print("hello there")
        # val_corr = self.val_corr.compute()
        # self.log('val_corr', val_corr, prog_bar=True, logger=True)
        # val_outs = {k: torch.cat(v, 0) for k, v in val_step_outputs.items()}
        val_outs = torch.cat([out['out'] for out in val_step_outputs], 0)
        val_ys = torch.cat([out['y'] for out in val_step_outputs], 0)
        val_corr = vectorized_correlation(val_outs, val_ys).mean()
        self.log('val_corr', val_corr, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "BatchNorm3D.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        # optimizer_grouped_parameters = [
        #     {
        #         "params": [p for p in self.backbone.parameters()],
        #         "lr": 1e-4,
        #     },
        #     {
        #         "params": [p for p in self.conv31.parameters()],
        #         "lr": 3e-4,
        #     },
        #     {
        #         "params": [p for p in self.fc.parameters()],
        #         "lr": 3e-4,
        #     },
        # ]
        # optimizer = AdaBelief(optimizer_grouped_parameters, lr=self.hparams.learning_rate)
        optimizer = SGD(optimizer_grouped_parameters, lr=self.lr, momentum=0.9)
        sch = CosineAnnealingLR(optimizer, self.hparams.max_epochs)

        return [optimizer], [sch]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video_frames', type=int, default=16)
    parser.add_argument('--video_size', type=int, default=288)
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--max_epochs', type=int, default=300)
    parser.add_argument('--backbone_freeze_epochs', type=int, default=100)
    parser.add_argument('--gpus', type=str, default='-1')
    parser.add_argument('--cached', type=bool, default=True)  #fix this

    parser = LitI3DFC.add_model_specific_args(parser)
    args = parser.parse_args()
    hparams = vars(args)

    dm = AlgonautsMINIDataModule(batch_size=args.batch_size, datasets_dir='datasets/',
                                 num_frames=hparams['video_frames'], resolution=hparams['video_size'],
                                 cached=args.cached)
    dm.setup()
    hparams['output_size'] = dm.num_voxels

    trainer = pl.Trainer(
        precision=16,
        gpus=args.gpus,
        # accelerator='ddp',
        plugins=DDPPlugin(find_unused_parameters=False),
        # plugins=DataParallelPlugin([0,1]),
        # limit_train_batches=0.1,
        # limit_val_batches=0.2,
        # limit_test_batches=0.3,
        max_epochs=args.max_epochs,
        checkpoint_callback=False,
        val_check_interval=1.0,
        callbacks=[BackboneFinetuning(args.backbone_freeze_epochs)],
        # auto_lr_find=True,
    )

    backbone = modify_resnets_patrial_x3(multi_resnet3d50())

    plmodel = LitI3DFC(backbone, hparams)

    # trainer.tune(plmodel, datamodule=dm)
    trainer.fit(plmodel, dm)

    pass
