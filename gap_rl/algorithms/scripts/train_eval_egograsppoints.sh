eval_datasets="ycb_train ycb_eval acronym_eval"
dynamic_modes="bezier2d line circular random2d"
timestamp=`date "+%Y%m%d_%H%M%S"`

### ours
## egopoints
echo "=========== TRAINING ==========="
python sac_train.py --config-name egopoints_ur85_bezier2d_goalaux --exp-suffix GraspPointAppGroup_EARL_5M --timestamp "$timestamp"
echo "=========== EVALUATION ==========="
python sac_LoG_dynamic_eval.py \
      --gen-traj-modes $dynamic_modes \
      --eval-datasets $eval_datasets \
      --obs-mode "state_egopoints_rt" \
      --timestamp $timestamp \
      --seeds "1029" \
      --save-video
