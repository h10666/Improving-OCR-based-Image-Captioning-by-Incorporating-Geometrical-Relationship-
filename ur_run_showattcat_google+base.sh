result_dir="/localdisk2/jwang195/results/base+spatial/1111-1"
if [ ! -d $result_dir ]; then
  mkdir -p $result_dir
fi
id="show_attend_tell_cat"
ckpt_path=$result_dir"/log_"$id
if [ ! -d $ckpt_path ]; then
  mkdir $ckpt_path
fi
if [ ! -f $ckpt_path"/infos_"$id"-best.pkl" ]; then
start_from=""
else
start_from="--start_from "$ckpt_path
fi
eval_path=$result_dir"/eval_results"
if [ ! -d $eval_path ]; then
  mkdir $eval_path
fi

CUDA_VISIBLE_DEVICES=4,7 python train.py --id $id \
  --caption_model show_attend_tell_cat \
  --input_json ../../data/step5_3_eccvrel+mmrel_google+base/TC.pkl \
  --input_label_h5 ../../data/step5_3_eccvrel+mmrel_google+base/TC_label.h5 \
  --input_fc_dir ../../data/open_images/resnet152 \
  --input_att_dir ../../data/open_images/detectron_fix_100/fc6 \
  --input_box_dir ../../data/open_images/detectron_fix_100/fc6 \
  --input_ocr_dir ../../data/myang_features/m4c_textvqa_google_ocr_en_frcn_features \
  --input_aug_ocr_dir ../../data/m4c_textvqa_ocr_en_frcn_features \
  --batch_size 10 \
  --grad_clip_style Norm \
  --grad_clip 0.1 \
  --use_bn 0 \
  --use_obj_ln 2 \
  --ocr_score_mode rel_same_PTN \
  --learning_rate_decay_every 3 \
  --learning_rate 2e-4 \
  --learning_rate_decay_start 0 \
  --scheduled_sampling_start 0 \
  --checkpoint_path $ckpt_path \
  $start_from \
  --language_eval 1 \
  --val_images_use -1 \
  --max_epochs 100 \
  --result_dir $result_dir \
  --beam_size 2 \
  --rnn_size 1000 \
  --input_encoding_size 1000 \
  --ocr_size 80 \
  --use_ocr_with_angle 1 \
  --iteration_eval_every 1000000000000000000 \
  --epoch_eval_every 1 \
  --epoch_test_more 100