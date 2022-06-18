name=SEVol-debug
flag="--attn soft --train listener
      --featdropout 0.3
      --visual_feat --angle_feat
      --feedback sample
      --mlWeight 0.2 
      --gcn_topk 5 --glove_dim 300 --top_N_obj 8 --distance_decay_function same  
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 80000 --maxAction 20"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$1 python3 methods/nvem/train.py $flag --name $name 

# Try this with file logging:
# CUDA_VISIBLE_DEVICES=$1 unbuffer python r2r_src/train.py $flag --name $name | tee snap/$name/log
