from argparse import ArgumentParser

import kornia as K
import pytorch_lightning as pl
from adabelief_pytorch import AdaBelief
from pytorch_lightning.callbacks import BackboneFinetuning, ModelCheckpoint, EarlyStopping, StochasticWeightAveraging
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.plugins import DDPPlugin
from torch import Tensor
from torch.nn import SyncBatchNorm
from callbacks import ReduceAuxLossWeight

from bdcn import load_bdcn
from bdcn_edge import BDCNNeck
from dataloading import AlgonautsDataModule
from model_i3d import *
from sam import SAM
from utils import *
from pyramidpooling import *

from clearml import Task

task = Task.init(
    project_name='debug',
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

        # self.automatic_optimization = False

        # self.train_transform = DataAugmentation()
        self.train_transform = None

        self.backbone = backbone

        # self.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(backbone) # slooooow

        if self.hparams.backbone_type == 'i3d_rgb':
            self.neck = I3d_rgb(self.hparams)
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
        for x_i in self.hparams.pyramid_layers.split(','):
            for pathway in self.hparams.pathways.split(','):
                self.aux_loss_weights[f'{pathway}_{x_i}'] = self.hparams.aux_loss_weight

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
        parser.add_argument('--pooling_size', type=int, default=5)
        parser.add_argument('--x1_pooling_mode', type=str, default='spp')
        parser.add_argument('--x2_pooling_mode', type=str, default='spp')
        parser.add_argument('--x3_pooling_mode', type=str, default='spp')
        parser.add_argument('--x4_pooling_mode', type=str, default='spp')
        parser.add_argument('--pooling_size_x1', type=int, default=5)
        parser.add_argument('--pooling_size_x2', type=int, default=5)
        parser.add_argument('--pooling_size_x3', type=int, default=5)
        parser.add_argument('--pooling_size_x4', type=int, default=5)
        parser.add_argument('--final_fusion', type=str, default='concat')
        parser.add_argument('--pyramid_layers', type=str, default='x1,x2,x3,x4')
        parser.add_argument('--bdcn_outputs', type=str, default='-1')
        parser.add_argument('--bdcn_pool_size', type=int, default=1)
        parser.add_argument('--lstm_layers', type=int, default=1)
        parser.add_argument('--pathways', type=str, default='topdown,bottomup', help="none or topdown,bottomup")
        parser.add_argument('--aux_loss_weight', type=float, default=0.0)
        parser.add_argument('--reduce_aux_loss_ratio', type=float, default=-1)
        parser.add_argument('--detach_aux', default=False, action="store_true")
        parser.add_argument('--sample_voxels', default=False, action="store_true")
        parser.add_argument('--sample_num_voxels', type=int, default=1000)
        parser.add_argument('--freeze_bn', default=False, action="store_true")
        parser.add_argument('--convtrans_bn', default=False, action="store_true")
        parser.add_argument('--no_convtrans', default=False, action="store_true")
        # legacy
        parser.add_argument('--softpool', default=False, action="store_true")
        parser.add_argument('--fc_batch_norm', default=False, action="store_true")
        parser.add_argument('--global_pooling', default=False, action="store_true")
        return parser

    def forward(self, x):
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
            out_vid = [x.reshape(s[0], s[1], s[3], s[4]) for x in out_vid]
            if self.training == False:
                self.logger.experiment.add_image('edges', F.sigmoid(out_vid[-1][0, -1, :, :]),
                                                 global_step=self.global_step, dataformats='HW')
                # self.logger.experiment.add_scalar(f'edges/max', F.sigmoid(out_vid[-1][0, -1, :, :]).max(), global_step=self.global_step)

        # out = self.neck(out_vid, x_add)
        out = self.neck(out_vid)
        return out

    def _shared_train_val(self, batch, batch_idx, prefix, is_log=True):
        x, y = batch
        if self.hparams['track'] == 'mini_track':
            out, out_aux = self(x)
            loss = F.mse_loss(out, y)
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
            if is_log:
                self.log(f'{prefix}_mse_loss/final', loss,
                         on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            all_loss = loss + loss_aux
            return out, all_loss, out_aux
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
        x = self.train_transform(x) if self.train_transform is not None else x
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
        out, loss, out_aux = self._shared_train_val(batch, batch_idx, 'val')
        y = batch[-1]
        # self.val_corr(out[:, 0], y[:, 0])
        return {'out': out, 'y': y, 'out_aux': out_aux}

    def validation_epoch_end(self, val_step_outputs) -> None:
        val_outs = torch.cat([out['out'] for out in val_step_outputs], 0)
        val_ys = torch.cat([out['y'] for out in val_step_outputs], 0)
        val_corr = vectorized_correlation(val_outs, val_ys)
        # print(val_corr.mean())
        self.log('val_corr/final', val_corr.mean(), prog_bar=True, logger=True, sync_dist=False)

        def roi_correlation(x, y, roi_lens, roi_names):
            xx = torch.hsplit(x, roi_lens)[:-1]
            yy = torch.hsplit(y, roi_lens)[:-1]

            corrs_dict = {}
            i = 0
            for roi in roi_names:
                a, b = xx[i], yy[i]
                corr = vectorized_correlation(a, b).mean().item()
                corrs_dict[roi] = corr
                i += 1

            return corrs_dict

        for roi, corr in roi_correlation(val_outs, val_ys, self.trainer.datamodule.roi_lens,
                                         self.hparams.rois.split(',')).items():
            self.log(f'val_corr/{roi}', corr, logger=True, sync_dist=False)

        # aux heads
        if val_step_outputs[0]['out_aux'] is not None:
            keys = val_step_outputs[0]['out_aux'].keys()
            for k in keys:
                outs = []
                for i, val_step_output in enumerate(val_step_outputs):
                    out_aux = val_step_output['out_aux']
                    outs.append(out_aux[k])
                outs = torch.cat(outs, 0)
                aux_val_corr = vectorized_correlation(outs, val_ys).mean()
                self.log(f'val_corr/{k}', aux_val_corr, prog_bar=False, logger=True, sync_dist=True)
                self.log(f'aux_lw/{k}', self.aux_loss_weights[k], prog_bar=False, logger=True, sync_dist=True)

        # print(self.aux_loss_weights)

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

        return [optimizer], []


def train(args):
    hparams = vars(args)

    dm = AlgonautsDataModule(batch_size=args.batch_size, datasets_dir=args.datasets_dir, rois=args.rois,
                             num_frames=args.video_frames, resolution=args.video_size, track=args.track,
                             cached=args.cached, val_ratio=args.val_ratio,
                             random_split=args.val_random_split,
                             use_cv=args.use_cv, num_split=int(1 / args.val_ratio), fold=args.fold,
                             additional_features_dir=args.additional_features_dir,
                             additional_features=args.additional_features,
                             preprocessing_type=args.preprocessing_type)
    dm.setup('fit')

    checkpoint_callback = ModelCheckpoint(
        monitor='val_corr/final',
        dirpath=f'/home/huze/.cache/checkpoints/{task.id}',
        filename='{epoch:02d}-{val_corr/final:.6f}',
        save_weights_only=True,
        save_top_k=1,
        mode='max',
    )

    early_stop_callback = EarlyStopping(
        monitor='val_corr/final',
        min_delta=0.001,
        patience=int(args.early_stop_epochs / args.val_check_interval),
        verbose=False,
        mode='max'
    )

    finetune_callback = BackboneFinetuning(
        args.backbone_freeze_epochs if not args.debug else 1
    )

    callbacks = [early_stop_callback, finetune_callback]

    if args.reduce_aux_loss_ratio >= 0:
        for x_i in args.pyramid_layers.split(','):
            for pathway in args.pathways.split(','):
                k = f'{pathway}_{x_i}'
                callback = ReduceAuxLossWeight(
                    monitor=f'val_corr/{k}',
                    aux_name=k,
                    reduce_ratio=args.reduce_aux_loss_ratio,
                    reduce_max_counts=int(math.log(0.001, args.reduce_aux_loss_ratio) + 1),
                    mode='max',
                    min_delta=0.01,
                    patience=2,
                    verbose=False,
                )
                callbacks.append(callback)

    if args.save_checkpoints:
        callbacks.append(checkpoint_callback)

    if args.debug:
        torch.set_printoptions(10)

    if args.backbone_type == 'i3d_rgb':
        backbone = modify_resnets_patrial_x_all(multi_resnet3d50(cache_dir=args.cache_dir))
    elif args.backbone_type == 'bdcn_edge':
        backbone = load_bdcn(args.bdcn_path)
    else:
        NotImplementedError()

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
        # auto_lr_find=True,
        # auto_scale_batch_size='binsearch'  # useful?
        # track_grad_norm=2,
    )

    hparams['output_size'] = dm.num_voxels

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
        prediction = trainer.predict(plmodel, datamodule=dm)
        prediction = torch.cat([p[0] for p in prediction], 0)
        for roi, pred in zip(args.rois.split(','), torch.hsplit(prediction, dm.roi_lens)[:-1]):
            torch.save(pred, os.path.join(prediction_dir, f'{roi}.pt'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video_frames', type=int, default=16)
    parser.add_argument('--video_size', type=int, default=288)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=300)
    parser.add_argument('--datasets_dir', type=str, default='/home/huze/algonauts_datasets/')
    parser.add_argument('--bdcn_path', type=str,
                        default='/home/huze/my_algonauts/bdcn-final-model/bdcn_pretrained_on_bsds500.pth')
    parser.add_argument('--additional_features', type=str, default='')
    parser.add_argument('--additional_features_dir', type=str, default='/data_smr/huze/projects/my_algonauts/features/')
    parser.add_argument('--track', type=str, default='mini_track')
    parser.add_argument('--backbone_type', type=str, default='i3d_rgb', help='i3d_rgb, bdcn_edge')
    parser.add_argument('--rois', type=str, default="EBA")
    parser.add_argument('--subs', type=str, default="all")
    parser.add_argument('--num_subs', type=int, default=10)
    parser.add_argument('--backbone_freeze_epochs', type=int, default=100)
    parser.add_argument('--gpus', type=str, default='1')
    parser.add_argument('--val_check_interval', type=float, default=1.0)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--val_random_split', default=False, action="store_true")
    parser.add_argument('--save_checkpoints', default=False, action="store_true")
    parser.add_argument('--use_cv', default=False, action="store_true")
    parser.add_argument('--fold', type=int, default=-1)
    parser.add_argument('--preprocessing_type', type=str, default='mmit', help='mmit, bdcn')
    parser.add_argument('--early_stop_epochs', type=int, default=10)
    parser.add_argument('--cached', default=False, action="store_true")
    parser.add_argument("--fp16", default=False, action="store_true")
    parser.add_argument("--asm", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument('--predictions_dir', type=str, default='/data_smr/huze/projects/my_algonauts/predictions/')
    parser.add_argument('--cache_dir', type=str, default='/home/huze/.cache/')

    parser = LitModel.add_model_specific_args(parser)
    args = parser.parse_args()

    train(args)
