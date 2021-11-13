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