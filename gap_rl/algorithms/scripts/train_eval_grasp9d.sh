eval_datasets="ycb_train ycb_eval acronym_eval"
dynamic_modes="bezier2d line circular random2d"
timestamp=`date "+%Y%m%d_%H%M%S"`

### baselines
## grasps 6d
echo "=========== TRAINING ==========="
python sac_train.py --config-name grasp9d_ur85_bezier2d --exp-suffix 0 --timestamp "$timestamp"
echo "=========== EVALUATION ==========="
python sac_LoG_dynamic_eval.py \
      --gen-traj-modes $dynamic_modes \
      --eval-datasets $eval_datasets \
      --obs-mode "state_grasp9d_rt" \
      --timestamp $timestamp \
      --seed "1029" \
      --save-video
