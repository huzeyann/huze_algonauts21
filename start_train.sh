python main.py --cached --use_cv --save_checkpoints --fp16 --old_mix --final_fusion concat --debug

python main.py --cached --use_cv --fp16 --old_mix --final_fusion concat \
--i3d_flow_path /home/huze/i3d_flow.pt \
--backbone_type i3d_flow \
--video_frames 64 \
--video_size 256 \
--crop_size 224 \
--cached \
--preprocessing_type i3d_flow \
--predictions_dir /tmp \
--track mini_track \
--rois V1,V2,V3,V4 \
--backbone_lr_ratio 0.5 \
--backbone_freeze_epochs 4 \
--pyramid_layers x3 \
--x3_pooling_mode no \
--pretrained \
--debug


python main.py --cached --use_cv --old_mix --final_fusion concat \
--backbone_type bdcn_edge \
--preprocessing_type bdcn \
--video_size 64 \
--video_frames 4 \
--predictions_dir /tmp \
--track full_track \
--rois WB \
--backbone_lr_ratio 0.25 \
--backbone_freeze_epochs 12 \
--lstm_layers 0 \
--spp \
--spp_size 3 6 9 \
--debug


python main.py --cached --use_cv --old_mix --final_fusion concat \
--backbone_type bdcn_edge \
--preprocessing_type bdcn \
--video_size 64 \
--video_frames 4 \
--predictions_dir /tmp \
--track mini_track \
--rois V1 \
--backbone_lr_ratio 0.25 \
--backbone_freeze_epochs 12 \
--spp \
--spp_size 3 6 9 \
--debug

python main.py --cached --fp16 --use_cv --old_mix --final_fusion concat \
--gpus 1 \
--learning_rate 1e-4 \
--pretrained \
--backbone_type i3d_rgb \
--preprocessing_type mmit \
--video_size 288 \
--video_frames 16 \
--predictions_dir /data_smr/huze/projects/my_algonauts/predictions/ \
--track full_track \
--rois WB \
--backbone_lr_ratio 0.5 \
--backbone_freeze_epochs 4 \
--early_stop_epochs 5 \
--max_epochs 100 \
--spp \
--spp_size 6 \
--pyramid_layers x3 \
--accumulate_grad_batches 4 \
--batch_size 8 \
--freeze_bn \
--old_mix \
--no_convtrans \
--final_fusion concat \
--voxel_index_file /data_smr/huze/projects/my_algonauts/tmp/C5_voxel_indexs.pt \
--save_checkpoints \
--rm_checkpoints \
--checkpoints_dir /mnt/huze/ckpts_mkii/ \

python main.py --cached --fp16 --use_cv --old_mix --final_fusion concat \
--gpus 1 \
--learning_rate 3e-4 \
--pretrained \
--backbone_type i3d_rgb \
--preprocessing_type mmit \
--video_size 288 \
--video_frames 16 \
--track full_track \
--rois WB \
--backbone_lr_ratio 0.1 \
--backbone_freeze_score 0.168 \
--early_stop_epochs 5 \
--max_epochs 100 \
--spp \
--spp_size 9 \
--pyramid_layers x3 \
--accumulate_grad_batches 4 \
--batch_size 8 \
--freeze_bn \
--old_mix \
--no_convtrans \
--final_fusion concat \
--rm_checkpoints \
--tag test \
--debug

clearml-agent daemon --queue 16-0 --detached
clearml-agent daemon --queue 16-1 --detached
clearml-agent daemon --stop