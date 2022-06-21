name=new_train_speaker_test
flag="--attn soft --train speaker
      --visual_feat --angle_feat
      --speaker methods/SEvol/snaps/speaker/best_val_unseen_bleu
      --subout max --dropout 0.6 --optim adam --lr 1e-4 --iters 80000 --maxAction 35"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$1 python3 methods/SEvol/train.py $flag --name $name 
