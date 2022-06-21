name=test
flag="--attn soft --train validlistener
      --featdropout 0.3
      --load methods/nvem/snaps/release_best/best_val_unseen
      --visual_feat --angle_feat
      --features clip --gcn_topk 5 --distance_decay_function same
      --glove_dim 300  --submit
      --subout max --dropout 0.5 --maxAction 25"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$1 coverage run methods/SEvol/train.py $flag --name $name 

