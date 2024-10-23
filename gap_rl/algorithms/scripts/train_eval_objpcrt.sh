eval_datasets="ycb_train ycb_eval acronym_eval"
dynamic_modes="bezier2d line circular random2d"
timestamp=`date "+%Y%m%d_%H%M%S"`

### baselines
## obj points rt
echo "=========== TRAINING ==========="
python sac_train.py --config-name objpointsrt_ur85_bezier2d_goalaux --exp-suffix 0 --timestamp "$timestamp"
echo "=========== EVALUATION ==========="
python sac_baselines_dynamic_eval.py \
     --obj-test-num 20 \
     --max-steps 100 \
     --cam-mode "hand_realsense" \
     --gen-traj-modes $dynamic_modes \
     --pc-mode "rt" \
     --obs-mode "state_objpoints_rt" \
     --eval-datasets $eval_datasets \
     --timestamp $timestamp \
     --seeds "1029" \
     --save-video
