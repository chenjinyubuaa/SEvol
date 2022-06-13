
import torch

import os
import time
import json
import numpy as np
from collections import defaultdict
from speaker import Speaker

from utils import read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,timeSince, read_img_features
import utils
from env import R2RBatch
from agent import Seq2SeqAgent
from eval import Evaluation
from param import args,DEBUG_FILE

import time
import warnings
warnings.filterwarnings("ignore")
import pickle as pkl

from tensorboardX import SummaryWriter

#os.environ["CUDA_VISIBLE_DEVICES"] = "4"

log_dir = 'snap/%s' % args.name
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

TRAIN_VOCAB = None
TRAINVAL_VOCAB = None

IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
PLACE365_FEATURES = 'img_features/ResNet-152-places365.tsv'
if args.features == 'imagenet':
    features = IMAGENET_FEATURES
elif args.features == 'places365':
    features = PLACE365_FEATURES
else:
    raise NotImplementedError

if args.fast_train:
    name, ext = os.path.splitext(features)
    features = name + "-fast" + ext

feedback_method = args.feedback # teacher or sample

print(args)


def train_speaker(train_env, tok, n_iters, log_every=500, val_envs={}):
   return None


def train(train_env, tok, n_iters, log_every=100, val_envs={}, aug_env=None):
    return None

def valid(train_env, tok, val_envs={}):
    agent = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    print("Loaded the listener model at iter %d from %s" % (agent.load(args.load), args.load))

    for env_name, (env, evaluator) in val_envs.items():
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        agent.test(use_dropout=False, feedback='argmax', iters=iters)
        result = agent.get_results()

        if env_name != '':
            score_summary, _ = evaluator.score(result)
            loss_str = "Env name: %s" % env_name
            for metric,val in score_summary.items():
                loss_str += ', %s: %.4f' % (metric, val)
            print(loss_str)

        if args.submit:
            json.dump(
                result,
                open(os.path.join(log_dir, "submit_%s.json" % env_name), 'w'),
                sort_keys=True, indent=4, separators=(',', ': ')
            )

def setup():
    np.random.seed(233)
    torch.manual_seed(233)
    torch.cuda.manual_seed(233)
    # Check for vocabs
    if not os.path.exists(TRAIN_VOCAB):
        write_vocab(build_vocab(splits=['train']), TRAIN_VOCAB)
    if not os.path.exists(TRAINVAL_VOCAB):
        write_vocab(build_vocab(splits=['train','val_seen','val_unseen']), TRAINVAL_VOCAB)


def train_val():
    ''' Train on the training set, and validate on seen and unseen splits. '''
    # args.fast_train = True
    setup()
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)
    if args.debug:
        from collections import defaultdict
        feat_dict = defaultdict(lambda:np.zeros((36,2048),dtype=np.float32))
        featurized_scans = set(json.load(open('./methods/neural_symbolic/debug_featurized_scans.json','r')))
        args.views = 36
    else:
        feat_dict = read_img_features(features)
        featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])
    with open('./img_features/objects/pano_object_class.pkl', 'rb') as f:
        obj_dict = pkl.load(f)
    
    if args.debug:
        train_env = R2RBatch(feat_dict,obj_dict,batch_size=args.batchSize,
                         splits=['val_unseen'], tokenizer=tok)
    elif args.train_sub:
        train_env = R2RBatch(feat_dict,obj_dict,batch_size=args.batchSize,
                         splits=['train-sub'], tokenizer=tok)
    else:
        train_env = R2RBatch(feat_dict,obj_dict,batch_size=args.batchSize,
                            splits=['train'], tokenizer=tok)
    from collections import OrderedDict

    val_env_names = ['train', 'val_seen', 'val_unseen']
    if args.train_sub:
        val_env_names = ['train-sub', 'val_seen', 'val_unseen']
    if args.submit:
        val_env_names = ['test','val_seen','val_unseen']
    else:
        pass

    if args.debug:
        val_env_names = ['val_unseen']

    val_envs = OrderedDict(
        ((split,
          (R2RBatch(feat_dict, obj_dict, batch_size=args.batchSize, splits=[split], tokenizer=tok),
           Evaluation([split], featurized_scans, tok))
          )
         for split in val_env_names
         )
    )

    if args.train == 'listener':
        train(train_env, tok, args.iters, val_envs=val_envs)
    elif args.train == 'validlistener':
        valid(train_env, tok, val_envs=val_envs)
    elif args.train == 'speaker':
        train_speaker(train_env, tok, args.iters, val_envs=val_envs)
    elif args.train == 'validspeaker':
        valid_speaker(tok, val_envs)
    else:
        assert False


def valid_speaker(tok, val_envs):
    import tqdm
    listner = Seq2SeqAgent(None, "", tok, args.maxAction)
    speaker = Speaker(None, listner, tok)
    speaker.load(args.load)

    for env_name, (env, evaluator) in val_envs.items():
        if env_name == 'train':
            continue
        print("............ Evaluating %s ............." % env_name)
        speaker.env = env
        path2inst, loss, word_accu, sent_accu = speaker.valid(wrapper=tqdm.tqdm)
        path_id = next(iter(path2inst.keys()))
        print("Inference: ", tok.decode_sentence(path2inst[path_id]))
        print("GT: ", evaluator.gt[path_id]['instructions'])
        pathXinst = list(path2inst.items())
        name2score = evaluator.lang_eval(pathXinst, no_metrics={'METEOR'})
        score_string = " "
        for score_name, score in name2score.items():
            score_string += "%s_%s: %0.4f " % (env_name, score_name, score)
        print("For env %s" % env_name)
        print(score_string)
        print("Average Length %0.4f" % utils.average_length(path2inst))



if __name__ == "__main__":
    if args.train in ['speaker', 'rlspeaker', 'validspeaker',
                      'listener', 'validlistener']:
        train_val()
    elif args.train == 'auglistener':
        train_val_augment()
    else:
        assert False

