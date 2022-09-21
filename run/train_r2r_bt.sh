name=SEVol_bt
flag="--attn soft --train auglistener --selfTrain
      --aug tasks/R2R/data/aug_paths.json
      --speaker snap/clip_speaker/best_val_unseen_bleu
      --load methods/SEvol/snaps/w_o_data_aug_best/best_val_unseen
      --visual_feat --angle_feat
      --top_N_obj 8
      --gcn_topk 5 --glove_dim 300 --distance_decay_function same
      --accumulateGrad
      --featdropout 0.4
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 200000 --maxAction 20"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$1 python3 methods/SEvol/train.py $flag --name $name 

# Try this with file logging:
# CUDA_VISIBLE_DEVICES=$1 unbuffer python methods/SEvol/train.py $flag --name $name | tee snap/$name/log
