
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules import activation
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from param import args
from egcn_h import GRCU_Cell
if args.egcn_activation == 'relu':
     act = torch.nn.RReLU()
egcn = GRCU_Cell(args,act)
egcn.cuda()
egcn.train()
cand_angle_feat,cand_obj_feat,near_angle_feat ,near_obj_mask ,near_obj_feat = torch.load('test')
cand_angle_feat=cand_angle_feat.cuda()
cand_obj_feat=cand_obj_feat.cuda()
near_angle_feat =near_angle_feat.cuda()
near_obj_mask =near_obj_mask.cuda()
near_obj_feat=near_obj_feat.cuda()
object_graph_feat = torch.cat((cand_obj_feat.unsqueeze(2),near_obj_feat),2)

angle_graph_feat = near_angle_feat.unsqueeze(3).expand(-1,-1,-1,object_graph_feat.shape[3],-1)

object_graph_feat = object_graph_feat.reshape(near_obj_feat.shape[0],-1,near_obj_feat.shape[-1])

angle_graph_feat = angle_graph_feat.reshape(near_angle_feat.shape[0],-1,angle_graph_feat.shape[-1])

object_graph_feat = torch.cat((object_graph_feat,angle_graph_feat),2)

adj_list = torch.ones(object_graph_feat.shape[0],object_graph_feat.shape[1],object_graph_feat.shape[1]).cuda()

adj_list = adj_list/object_graph_feat.shape[1]

mask = torch.ones(object_graph_feat.shape[0],object_graph_feat.shape[1]).cuda()

node_feats = egcn(adj_list,object_graph_feat,mask)
