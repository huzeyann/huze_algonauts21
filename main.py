from argparse import ArgumentParser
from typing import Any, Optional

import kornia as K
from kornia.augmentation import RandomCrop3D, CenterCrop3D
import pytorch_lightning as pl
from adabelief_pytorch import AdaBelief
from pytorch_lightning.callbacks import BackboneFinetuning, ModelCheckpoint, EarlyStopping, StochasticWeightAveraging
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.plugins import DDPPlugin
from torch import Tensor
from torch.nn import SyncBatchNorm
from torch.optim.lr_scheduler import MultiStepLR, StepLR

from callbacks import ReduceAuxLossWeight

from bdcn import load_bdcn
from bdcn_neck import BDCNNeck
from dataloading import AlgonautsDataModule
from i3d_flow import load_i3d_flow
from model_i3d import *
from sam import SAM
from utils import *
from pyramidpooling3d import *
import pandas as pd

from clearml import Task, Logger

PROJECT_NAME = 'Algonauts separate layers edge'

task = Task.init(
    project_name=PROJECT_NAME,
    task_name='task template',
    tags=None,
    reuse_last_task_id=False,
    continue_last_task=False,
    output_uri=None,
    auto_connect_arg_parser=True,
    auto_connect_frameworks=True,
    auto_resource_monitoring=True,
    auto_connect_streams=True,
)


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


class LitModel(LightningModule):

    def __init__(self, backbone, hparams: dict, *args, **kwargs):
        super(LitModel, self).__init__()
        self.save_hyperparameters(hparams)
        # self.hparams = hparams
        self.lr = self.hparams.learning_rate
        self.rois = [self.hparams.rois] if not self.hparams.separate_rois else self.hparams.rois.split(',')

        # self.automatic_optimization = False

        if self.hparams.crop_size > 0:
            if self.hparams.random_crop:
                self.train_transform = RandomCrop3D(
                    (self.hparams.video_frames, self.hparams.crop_size, self.hparams.crop_size))
            else:
                self.train_transform = TensorCenterCrop(self.hparams.crop_size)
            self.test_transform = TensorCenterCrop(self.hparams.crop_size)
        else:
            self.train_transform = None
            self.test_transform = None

        self.backbone = backbone

        # self.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(backbone) # slooooow

        if self.hparams.backbone_type == 'i3d_rgb' or self.hparams.backbone_type == 'i3d_flow':
            self.neck = I3d_neck(self.hparams)
        elif self.hparams.backbone_type == 'bdcn_edge':
            self.neck = BDCNNeck(self.hparams)
        else:
            NotImplementedError()

        if self.hparams.track == 'full_track' and not self.hparams.no_convtrans:
            # voxel mask
            subs = [f'sub{i + 1:02d}' for i in range(10)] if self.hparams.subs == 'all' else self.hparams.subs
            voxel_masks = []
            for sub in subs:
                voxel_mask = np.load(os.path.join(hparams['datasets_dir'], 'fmris', f'{sub}_voxel_mask.npy'))
                voxel_mask = torch.tensor(voxel_mask, device=self.device)
                voxel_mask = F.pad(voxel_mask, (4, 4, 1, 1, 0, 1))
                voxel_masks.append(voxel_mask)
            print('voxel_mask in ', self.device)
            self.voxel_masks = torch.stack(voxel_masks, 0)

        # aux reduction
        l = len(self.hparams.pyramid_layers.split(',')) * len(self.hparams.pathways.split(','))

        self.aux_loss_weights = {}
        for roi in self.rois:
            for x_i in self.hparams.pyramid_layers.split(','):
                for pathway in self.hparams.pathways.split(','):
                    if self.hparams[f'aux_loss_weight_{x_i}'] >= 0:
                        self.aux_loss_weights[f'{roi}_{pathway}_{x_i}'] = self.hparams[f'aux_loss_weight_{x_i}']
                    else:
                        self.aux_loss_weights[f'{roi}_{pathway}_{x_i}'] = self.hparams.aux_loss_weight

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument_group("LitModel")
        parser.add_argument('--conv_size', type=int, default=256)
        parser.add_argument('--num_layers', type=int, default=2)
        parser.add_argument('--activation', type=str, default='elu')
        parser.add_argument('--layer_hidden', type=int, default=2048)
        parser.add_argument('--first_layer_hidden', type=int, default=256)
        parser.add_argument('--dropout_rate', type=float, default=0.0)
        parser.add_argument('--weight_decay', type=float, default=1e-2)
        parser.add_argument('--learning_rate', type=float, default=3e-4)
        parser.add_argument('--backbone_lr_ratio', type=float, default=0.1)
        parser.add_argument('--pooling_mode', type=str, default='avg')
        parser.add_argument('--spp', default=False, action="store_true")
        parser.add_argument('--pooling_size', type=int, default=5)
        parser.add_argument('--pooling_size_t', type=int, default=1)
        parser.add_argument('--spp_size', type=int, nargs='+', help='SPP')
        parser.add_argument('--spp_size_t', type=int, nargs='+', help='SPP')
        parser.add_argument('--x1_pooling_mode', type=str, default='spp')
        parser.add_argument('--x2_pooling_mode', type=str, default='spp')
        parser.add_argument('--x3_pooling_mode', type=str, default='spp')
        parser.add_argument('--x4_pooling_mode', type=str, default='spp')
        parser.add_argument('--pooling_size_x1', type=int, default=5)
        parser.add_argument('--pooling_size_x2', type=int, default=5)
        parser.add_argument('--pooling_size_x3', type=int, default=5)
        parser.add_argument('--pooling_size_x4', type=int, default=5)
        parser.add_argument('--pooling_size_t_x1', type=int, default=4)
        parser.add_argument('--pooling_size_t_x2', type=int, default=4)
        parser.add_argument('--pooling_size_t_x3', type=int, default=2)
        parser.add_argument('--pooling_size_t_x4', type=int, default=1)
        parser.add_argument('--spp_size_x1', type=int, nargs='+', help='SPP')
        parser.add_argument('--spp_size_x2', type=int, nargs='+', help='SPP')
        parser.add_argument('--spp_size_x3', type=int, nargs='+', help='SPP')
        parser.add_argument('--spp_size_x4', type=int, nargs='+', help='SPP')
        parser.add_argument('--spp_size_t_x1', type=int, nargs='+', help='SPP', default=[1, 2, 2])
        parser.add_argument('--spp_size_t_x2', type=int, nargs='+', help='SPP', default=[1, 2, 2])
        parser.add_argument('--spp_size_t_x3', type=int, nargs='+', help='SPP', default=[1, 2, 2])
        parser.add_argument('--spp_size_t_x4', type=int, nargs='+', help='SPP', default=[1, 1, 1])
        parser.add_argument('--pyramid_layers', type=str, default='x1,x2,x3,x4')
        parser.add_argument('--final_fusion', type=str, default='conv')
        parser.add_argument('--old_mix', default=False, action="store_true")
        parser.add_argument('--lstm_layers', type=int, default=1)
        parser.add_argument('--pathways', type=str, default='topdown,bottomup', help="none or topdown,bottomup")
        parser.add_argument('--aux_loss_weight', type=float, default=0.0)
        parser.add_argument('--aux_loss_weight_x1', type=float, default=0.0)
        parser.add_argument('--aux_loss_weight_x2', type=float, default=-1)
        parser.add_argument('--aux_loss_weight_x3', type=float, default=-1)
        parser.add_argument('--aux_loss_weight_x4', type=float, default=-1)
        parser.add_argument('--aux_loss_weight_x5', type=float, default=-1)
        parser.add_argument('--reduce_aux_loss_ratio', type=float, default=-1)
        parser.add_argument('--reduce_aux_min_delta', type=float, default=0.0)
        parser.add_argument('--reduce_aux_patience', type=int, default=2)
        parser.add_argument('--detach_aux', default=False, action="store_true")
        parser.add_argument('--sample_voxels', default=False, action="store_true")
        parser.add_argument('--sample_num_voxels', type=int, default=1000)
        parser.add_argument('--freeze_bn', default=False, action="store_true")
        parser.add_argument('--convtrans_bn', default=False, action="store_true")
        parser.add_argument('--no_convtrans', default=False, action="store_true")
        parser.add_argument('--separate_rois', default=False, action="store_true")
        # legacy
        parser.add_argument('--fc_batch_norm', default=False, action="store_true")
        parser.add_argument('--global_pooling', default=False, action="store_true")
        return parser

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def forward(self, x):
        if not self.hparams.load_from_np:
            x_vid = x['video']
            # x_add = {k: v for k, v in x.items() if k != 'video'}

            if self.hparams.backbone_type == 'bdcn_edge':
                # if self.training == False:
                #     self.logger.experiment.add_image('original', x_vid[0, :, -1, :, :], global_step=self.global_step, dataformats='CHW')
                # vid to img
                x_vid = x_vid.permute(0, 2, 1, 3, 4)
                s = x_vid.shape
                x_vid = x_vid.reshape(s[0] * s[1], *s[2:])

            out_vid = self.backbone(x_vid)

            if self.hparams.backbone_type == 'bdcn_edge':
                # img to vid
                out_vid = out_vid.reshape(s[0], s[1], s[3], s[4])
                if self.training == False:
                    self.logger.experiment[0].add_image('edges', F.sigmoid(out_vid[0, -1, :, :]),
                                                        global_step=self.global_step, dataformats='HW')
                    # self.logger.experiment.add_scalar(f'edges/max', F.sigmoid(out_vid[-1][0, -1, :, :]).max(), global_step=self.global_step)
        else:
            out_vid = x

        out = self.neck(out_vid)
        return out

    def _shared_train_val(self, batch, batch_idx, prefix, is_log=True):
        x, y = batch
        if self.hparams['track'] == 'mini_track':
            out, out_aux = self(x)

            loss_all = 0
            if self.hparams.separate_rois:
                for roi, yy in zip(self.rois, dokodemo_hsplit(y, self.hparams.idx_ends)):

                    loss = F.mse_loss(out[roi], yy)
                    if is_log:
                        self.log(f'{prefix}_mse_loss/{roi}_final', loss,
                                 on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

                    if out_aux is not None:
                        aux_losses = []
                        for k, oa in out_aux.items():
                            if roi not in k: continue

                            loss_aux = F.mse_loss(oa, yy)
                            if is_log:
                                self.log(f'{prefix}_mse_loss/{k}', loss_aux,
                                         on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
                            loss_aux *= self.aux_loss_weights[k]
                            aux_losses.append(loss_aux)
                        loss_aux = torch.stack(aux_losses).mean()
                    else:
                        loss_aux = 0

                    loss_all = loss_all + loss + loss_aux
            else:
                loss = F.mse_loss(out[self.hparams.rois], y)
                if is_log:
                    self.log(f'{prefix}_mse_loss/{self.hparams.rois}_final', loss,
                             on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                if out_aux is not None:
                    aux_losses = []
                    for k, oa in out_aux.items():
                        loss_aux = F.mse_loss(oa, y)
                        if is_log:
                            self.log(f'{prefix}_mse_loss/{k}', loss_aux,
                                     on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
                        loss_aux *= self.aux_loss_weights[k]
                        aux_losses.append(loss_aux)
                    loss_aux = torch.stack(aux_losses).mean()
                else:
                    loss_aux = 0

                loss_all = loss_all + loss + loss_aux

            return out, loss_all, out_aux



        elif self.hparams.track == 'full_track':
            out = self(x)
            if not self.hparams.no_convtrans:
                out_voxels = out[self.voxel_masks.unsqueeze(0).expand(out.size()) == 1].reshape(out.shape[0], -1)
            else:
                out_voxels = out
            if self.hparams.sample_voxels:
                mask = torch.rand(out_voxels.shape[1]).unsqueeze(0).expand(out_voxels.size())
                th = self.hparams.sample_num_voxels / mask.shape[1]
                masked_out_voxels = out_voxels[mask < th]
                masked_y = y[mask < th]
                loss = F.mse_loss(masked_out_voxels, masked_y)
            else:
                loss = F.mse_loss(out_voxels, y)
            if is_log:
                self.log(f'{prefix}_mse_loss/final', loss,
                         on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
            return out_voxels, loss, None

    def training_step(self, batch, batch_idx):
        # print(self.neck.first_fcs['none_x3'][0].weight[0, 0])
        if self.hparams.freeze_bn:
            self.backbone.apply(disable_bn)
        x, y = batch
        if 'video' in x.keys():
            x['video'] = self.train_transform(x['video']) if self.train_transform is not None else x['video']
        batch = (x, y)

        out, loss, _ = self._shared_train_val(batch, batch_idx, 'train')
        return loss

    # def training_step(self, batch, batch_idx):
    #     self.train()
    #     if self.hparams.freeze_bn:
    #         self.backbone.apply(disable_bn)
    #     x, y = batch
    #     x = self.train_transform(x) if self.train_transform is not None else x
    #     batch = (x, y)
    #
    #     out, loss, _ = self._shared_train_val(batch, batch_idx, 'train')
    #     self.manual_backward(loss)  # take care fp16
    #
    #     optimizer = self.optimizers()
    #
    #     if self.hparams.asm:
    #         optimizer.first_step(zero_grad=True)
    #         self.backbone.apply(disable_bn)
    #         out, loss = self._shared_train_val(batch, batch_idx, 'train', is_log=False)
    #         self.manual_backward(loss)  # take care fp16
    #         optimizer.second_step(zero_grad=True)
    #     else:
    #         optimizer.step()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if 'video' in x.keys():
            x['video'] = self.test_transform(x['video']) if self.test_transform is not None else x['video']
        batch = (x, y)
        out, loss, out_aux = self._shared_train_val(batch, batch_idx, 'val')
        y = batch[-1]
        # self.val_corr(out[:, 0], y[:, 0])
        return {'out': out, 'y': y, 'out_aux': out_aux}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        x = batch
        if 'video' in x.keys():
            x['video'] = self.test_transform(x['video']) if self.test_transform is not None else x['video']
        return self(x)

    def validation_epoch_end(self, val_step_outputs) -> None:
        val_ys = torch.cat([out['y'] for out in val_step_outputs], 0)
        avg_val_corr = []
        if self.hparams.separate_rois:
            for roi, yy in zip(self.rois, dokodemo_hsplit(val_ys, self.hparams.idx_ends)):
                val_outs = torch.cat([out['out'][roi] for out in val_step_outputs], 0)
                val_corr = vectorized_correlation(val_outs, yy).mean().item()
                avg_val_corr.append(val_corr)
                # print(val_corr.mean())
                self.log(f'val_corr/{roi}_final', val_corr, prog_bar=True, logger=True, sync_dist=False)

                # aux heads
                if val_step_outputs[0]['out_aux'] is not None:
                    keys = val_step_outputs[0]['out_aux'].keys()
                    for k in keys:
                        if roi not in k: continue
                        outs = []
                        for i, val_step_output in enumerate(val_step_outputs):
                            out_aux = val_step_output['out_aux']
                            outs.append(out_aux[k])
                        outs = torch.cat(outs, 0)
                        corr = vectorized_correlation(outs, yy).mean().item()

                        self.log(f'aux_lw/{k}', self.aux_loss_weights[k], prog_bar=False, logger=True, sync_dist=True)
                        self.log(f'val_corr/{k}', corr, logger=True, sync_dist=False)
        else:
            # Logger.current_logger().report_scalar(
            #     "validation", "correlation", iteration=self.global_step, value=val_corr)
            def roi_correlation(x, y, roi_lens, roi_names):
                xx = dokodemo_hsplit(x, roi_lens)
                yy = dokodemo_hsplit(y, roi_lens)

                corrs_dict = {}
                i = 0
                for roi in roi_names:
                    a, b = xx[i], yy[i]
                    corr = vectorized_correlation(a, b).mean().item()
                    corrs_dict[roi] = corr
                    i += 1

                return corrs_dict

            val_outs = torch.cat([out['out'][self.hparams.rois] for out in val_step_outputs], 0)
            for roi, corr in roi_correlation(val_outs, val_ys, self.hparams.idx_ends,
                                             self.hparams.rois.split(',')).items():
                self.log(f'val_corr/{roi}_final', corr, logger=True, sync_dist=False)
                avg_val_corr.append(corr)

        avg_val_corr = np.mean(avg_val_corr)
        self.log(f'val_corr/final', avg_val_corr, prog_bar=True, logger=True, sync_dist=False)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "BatchNorm3D.weight", "BatchNorm1D.weight", "BatchNorm2D.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.backbone.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
                'lr': self.hparams.learning_rate * self.hparams.backbone_lr_ratio,
            },
            {
                "params": [p for n, p in self.backbone.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                'lr': self.hparams.learning_rate * self.hparams.backbone_lr_ratio,
            },
            {
                "params": [p for n, p in self.neck.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
                'lr': self.hparams.learning_rate,
            },
            {
                "params": [p for n, p in self.neck.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                'lr': self.hparams.learning_rate,
            },
        ]
        if not self.hparams.asm:
            optimizer = AdaBelief(optimizer_grouped_parameters)
            # optimizer = SGD(optimizer_grouped_parameters, lr=self.lr, momentum=0.9, weight_decay=self.hparams.weight_decay)
            # sch = CosineAnnealingLR(optimizer, self.hparams.max_epochs)
        else:
            optimizer = SAM(optimizer_grouped_parameters, AdaBelief, adaptive=True, rho=0.5)

        if self.hparams.step_lr_ratio < 1.0:
            scheduler = MultiStepLR(optimizer, milestones=self.hparams.step_lr_epochs, gamma=self.hparams.step_lr_ratio)
            # scheduler = StepLR(optimizer, step_size=self.hparams.backbone_freeze_epochs, gamma=0.5)
            tuple_of_dicts = (
                {"optimizer": optimizer, "lr_scheduler": scheduler},
            )
        else:
            tuple_of_dicts = (
                {"optimizer": optimizer},
            )

        return tuple_of_dicts


def train(args):
    hparams = vars(args)

    if args.backbone_type == 'i3d_flow':
        assert args.load_from_np

    dm = AlgonautsDataModule(batch_size=args.batch_size, datasets_dir=args.datasets_dir, rois=args.rois,
                             num_frames=args.video_frames, resolution=args.video_size, track=args.track,
                             cached=args.cached, val_ratio=args.val_ratio,
                             random_split=args.val_random_split,
                             use_cv=args.use_cv, num_split=int(1 / args.val_ratio), fold=args.fold,
                             additional_features_dir=args.additional_features_dir,
                             additional_features=args.additional_features,
                             preprocessing_type=args.preprocessing_type,
                             load_from_np=args.load_from_np)
    dm.setup('fit')

    callbacks = []

    early_stop_callback = EarlyStopping(
        monitor='val_corr/final',
        min_delta=0.00,
        patience=int(args.early_stop_epochs / args.val_check_interval),
        verbose=False,
        mode='max'
    )
    callbacks.append(early_stop_callback)

    if args.backbone_freeze_epochs > 0:
        finetune_callback = BackboneFinetuning(
            args.backbone_freeze_epochs if not args.debug else 1
        )
        callbacks.append(finetune_callback)

    rois = args.rois.split(',') if args.separate_rois else [args.rois]
    if args.reduce_aux_loss_ratio >= 0:
        for roi in rois:
            for x_i in args.pyramid_layers.split(','):
                for pathway in args.pathways.split(','):
                    k = f'{roi}_{pathway}_{x_i}'
                    callback = ReduceAuxLossWeight(
                        monitor=f'val_corr/{k}',
                        aux_name=k,
                        reduce_ratio=args.reduce_aux_loss_ratio,
                        reduce_max_counts=int(math.log(0.001, args.reduce_aux_loss_ratio) + 1),
                        mode='max',
                        min_delta=args.reduce_aux_min_delta,
                        patience=args.reduce_aux_patience,
                        verbose=False,
                    )
                    callbacks.append(callback)

    if args.save_checkpoints:
        checkpoint_callback = ModelCheckpoint(
            monitor='val_corr/final',
            dirpath=os.path.join(args.checkpoints_dir, task.id),
            filename='{epoch:02d}-{val_corr/final:.6f}',
            save_weights_only=True,
            save_top_k=1,
            mode='max',
        )
        callbacks.append(checkpoint_callback)

    if args.debug:
        torch.set_printoptions(10)

    if args.backbone_type == 'i3d_rgb':
        backbone = modify_resnets_patrial_x_all(multi_resnet3d50(cache_dir=args.i3d_rgb_dir))
    elif args.backbone_type == 'bdcn_edge':
        backbone = load_bdcn(args.bdcn_path)
    elif args.backbone_type == 'i3d_flow':
        backbone = load_i3d_flow(args.i3d_flow_path)
        # backbone = None
    else:
        NotImplementedError()

    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.logs_dir, 'lightning_logs', task.id))
    csv_logger = pl_loggers.CSVLogger(os.path.join(args.logs_dir, 'csv_logs', task.id))
    loggers = [tb_logger, csv_logger]

    trainer = pl.Trainer(
        precision=16 if args.fp16 else 32,
        gpus=args.gpus,
        accumulate_grad_batches=args.accumulate_grad_batches,
        # accelerator='ddp',
        # plugins=DDPPlugin(find_unused_parameters=False),
        limit_train_batches=1.0 if not args.debug else 0.2,
        limit_val_batches=1.0 if not args.debug else 0.5,
        # limit_test_batches=0.3,
        max_epochs=args.max_epochs if not args.debug else 2,
        checkpoint_callback=args.save_checkpoints,
        val_check_interval=args.val_check_interval if not args.debug else 1.0,
        callbacks=callbacks,
        logger=loggers,
        # auto_lr_find=True,
        # auto_scale_batch_size='binsearch'  # useful?
        # track_grad_norm=2,
    )

    hparams['output_size'] = dm.num_voxels
    hparams['idx_ends'] = dm.idx_ends
    z = np.array(dm.idx_ends).copy()
    z[1:] -= z[:-1].copy()
    hparams['roi_lens'] = z.tolist()

    if args.predictions_dir:
        prediction_dir = os.path.join(args.predictions_dir, task.id)
        if not os.path.exists(prediction_dir):
            os.system(f'mkdir {prediction_dir}')

    plmodel = LitModel(backbone, hparams)

    trainer.fit(plmodel, datamodule=dm)

    dm.teardown()

    if args.save_checkpoints:
        dm.setup('test')
        plmodel = LitModel.load_from_checkpoint(checkpoint_callback.best_model_path, backbone=backbone,
                                                hparams=hparams)
        predictions = trainer.predict(plmodel, datamodule=dm)
        for roi in rois:  # roi maybe multiple
            prediction = torch.cat([p[0][roi] for p in predictions], 0).cpu()
            if (not hparams['separate_rois']) and (len(hparams['rois'].split(',')) > 1):
                for rroi, pred in zip(hparams['rois'].split(','),
                                      dokodemo_hsplit(prediction, hparams['idx_ends'])):  # roi is single
                    # print(pred.shape)
                    torch.save(pred, os.path.join(prediction_dir, f'{rroi}.pt'))
            else:
                torch.save(prediction, os.path.join(prediction_dir, f'{roi}.pt'))
        os.remove(checkpoint_callback.best_model_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video_frames', type=int, default=16)
    parser.add_argument('--video_size', type=int, default=288)
    parser.add_argument('--crop_size', type=int, default=0)
    parser.add_argument('--random_crop', default=False, action="store_true")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=300)
    parser.add_argument('--datasets_dir', type=str, default='/home/huze/algonauts_datasets/')
    parser.add_argument('--bdcn_path', type=str,
                        default='/home/huze/algonauts_datasets/models/bdcn_pretrained_on_bsds500.pth')
    parser.add_argument('--i3d_flow_path', type=str,
                        default='/data_smr/huze/projects/my_algonauts/video_features/models/i3d/checkpoints/i3d_flow.pt')
    parser.add_argument('--additional_features', type=str, default='')
    parser.add_argument('--additional_features_dir', type=str, default='/data_smr/huze/projects/my_algonauts/features/')
    parser.add_argument('--track', type=str, default='mini_track')
    parser.add_argument('--backbone_type', type=str, default='i3d_rgb', help='i3d_rgb, bdcn_edge, i3d_flow')
    parser.add_argument('--rois', type=str, default="EBA")
    parser.add_argument('--subs', type=str, default="all")
    parser.add_argument('--num_subs', type=int, default=10)
    parser.add_argument('--backbone_freeze_epochs', type=int, default=0)
    parser.add_argument('--step_lr_epochs', type=int, nargs='+', default=[4, 12])
    parser.add_argument('--step_lr_ratio', type=float, default=1.0)
    parser.add_argument('--gpus', type=str, default='1')
    parser.add_argument('--val_check_interval', type=float, default=1.0)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--val_random_split', default=False, action="store_true")
    parser.add_argument('--load_from_np', default=False, action="store_true")
    parser.add_argument('--save_checkpoints', default=False, action="store_true")
    parser.add_argument('--use_cv', default=False, action="store_true")
    parser.add_argument('--fold', type=int, default=-1)
    parser.add_argument('--preprocessing_type', type=str, default='mmit', help='mmit, bdcn, i3d_flow')
    parser.add_argument('--early_stop_epochs', type=int, default=10)
    parser.add_argument('--cached', default=False, action="store_true")
    parser.add_argument("--fp16", default=False, action="store_true")
    parser.add_argument("--asm", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument('--predictions_dir', type=str, default='/data_smr/huze/projects/my_algonauts/predictions/')
    parser.add_argument('--checkpoints_dir', type=str, default='/home/huze/checkpoints/')
    parser.add_argument('--logs_dir', type=str, default='/data_smr/huze/projects/my_algonauts/')
    parser.add_argument('--i3d_rgb_dir', type=str, default='/home/huze/.cache/')

    parser = LitModel.add_model_specific_args(parser)
    args = parser.parse_args()

    train(args)
