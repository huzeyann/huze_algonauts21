python main.py --cached --use_cv --save_checkpoints --fp16 --old_mix --final_fusion concat --debug

python main.py --cached --use_cv --fp16 --old_mix --final_fusion concat \
--i3d_flow_path /home/huze/i3d_flow.pt \
--backbone_type i3d_flow \
--preprocessing_type i3d_flow \
--predictions_dir /tmp \
--track full_track \
--rois WB \
--backbone_lr_ratio 0.5 \
--backbone_freeze_epochs 4 \
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

python main.py --cached --use_cv --old_mix --final_fusion concat \
--backbone_type i3d_rgb \
--preprocessing_type mmit \
--video_size 288 \
--video_frames 16 \
--predictions_dir /tmp \
--track mini_track \
--rois V1 \
--backbone_lr_ratio 0.5 \
--backbone_freeze_epochs 4 \
--spp \
--spp_size 3 6 9 \
--debug