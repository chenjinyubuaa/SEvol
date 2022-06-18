import utils as u
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
from param import args

class SEM(torch.nn.Module):
    def __init__(self,args,activation):
        super().__init__()
        self.args = args
        cell_args = u.Namespace({})
        cell_args.rows = args.gcn_dim
        cell_args.cols = args.gcn_dim

        self.evolve_weights = mat_GRU_cell(cell_args)

        self.activation = activation
        self.GCN_init_mapping = Parameter(torch.Tensor(1,args.gcn_dim))
        self.init_mapping = nn.Sequential(nn.Linear(args.rnn_dim,args.gcn_dim),nn.Tanh())
        self.static_weights = nn.Parameter(torch.Tensor(args.gcn_dim,args.gcn_dim))
        self.reset_param(self.static_weights)
        self.reset_param(self.GCN_init_mapping)
        self.GCN_pre_weights = None
        self.GCN_init_weights = None
    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def init_weights(self,ht):
        ht = self.init_mapping(ht)
        self.GCN_pre_weights = torch.matmul(ht.unsqueeze(-1),self.GCN_init_mapping)
        self.GCN_init_weights = self.GCN_pre_weights

    def forward(self,Ahat,node_embs,mask,ht=None):
            #first evolve the weights from the initial and use the new weights with the node_embs
        policy_score = None
        if self.GCN_pre_weights is not None:
            GCN_weights,policy_score,scorer,entropy_object,selected_indexes = self.evolve_weights(self.GCN_pre_weights,node_embs,mask,ht)
            node_embs = node_embs.gather(1,selected_indexes.unsqueeze(-1).expand(-1,-1,node_embs.shape[-1]))
            Ahat = Ahat.gather(1,selected_indexes.unsqueeze(-1).expand(-1,-1,Ahat.shape[-1]))
            Ahat = Ahat.gather(2,selected_indexes.unsqueeze(1).expand(-1,Ahat.shape[1],-1))
            di = Ahat.sum(dim=1)
            di = di.pow(-1/2)
            Ahat =  di.unsqueeze(1)*Ahat*di.unsqueeze(-1).detach()
            node_embs = self.activation(Ahat.matmul(node_embs.matmul(GCN_weights)))
            node_embs1 = self.activation(Ahat.matmul(node_embs.matmul(self.static_weights)))
            node_embs = (node_embs+node_embs1)/2
            self.GCN_pre_weights = GCN_weights
        return node_embs,policy_score,scorer,entropy_object

class mat_GRU_cell(torch.nn.Module):
    def __init__(self,args1):
        super().__init__()
        self.args = args1
        self.update = mat_GRU_gate(args1.rows,
                                   args1.cols,
                                   torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(args1.rows,
                                   args1.cols,
                                   torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(args1.rows,
                                   args1.cols,
                                   torch.nn.Tanh())
        
        # self.choose_topk = TopK(feats = args.rows,
        #                         k = args.cols)
        self.choose_topk = TopK_with_h()

    def forward(self,prev_Q,prev_Z,mask,ht):


        z_topk,policy_score,scorer,entropy_object,topk_indices_out = self.choose_topk(prev_Z,mask,ht)
        update = self.update(z_topk,prev_Q)
        reset = self.reset(z_topk,prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q,policy_score,scorer,entropy_object,topk_indices_out

        

class mat_GRU_gate(torch.nn.Module):
    def __init__(self,rows,cols,activation):
        super().__init__()
        self.activation = activation
        #the k here should be in_feats which is actually the rows
        self.W = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.W)

        self.U = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.U)

        self.bias = Parameter(torch.zeros(rows,cols))

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self,x,hidden):
        out = self.activation(self.W.matmul(x) + \
                              self.U.matmul(hidden) + \
                              self.bias)

        return out

class TopK(torch.nn.Module):
    def __init__(self,feats,k):
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats,1))
        self.reset_param(self.scorer)
        
        self.k = k

    def reset_param(self,t):
        #Initialize based on the number of rows
        stdv = 1. / math.sqrt(t.size(0))
        t.data.uniform_(-stdv,stdv)

    def forward(self,node_embs,mask):
        batch_size,graph_size,feat_size = node_embs.shape
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        scores = scores.squeeze() + mask
        vals, topk_indices = scores.view(batch_size,-1).topk(self.k,dim=1)
        topk_indices = topk_indices[vals > -float("Inf")].view(batch_size,-1)
        if topk_indices.size(1) < self.k:
            topk_indices = u.pad_with_last_val(topk_indices,self.k)
        tanh = torch.nn.Tanh()

        if isinstance(node_embs, torch.sparse.FloatTensor) or \
           isinstance(node_embs, torch.cuda.sparse.FloatTensor):
            node_embs = node_embs.to_dense()
        out = node_embs.gather(1,topk_indices.unsqueeze(-1).expand(-1,-1,feat_size)) * tanh(scores.gather(1,topk_indices)).unsqueeze(-1)
        #we need to transpose the output
        return out.transpose(1,2)
    
    
class TopK_with_h(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.mapper =  nn.Sequential(nn.Linear(args.rnn_dim,args.gcn_dim),nn.Tanh())
        self.softmax = nn.Softmax()
        self.k = args.gcn_topk
    def reset_param(self,t):
        return None
    def forward(self,node_embs,mask,h_t = None):
        batch_size,graph_size,feat_size = node_embs.shape
        scorer = self.mapper(h_t.detach())
        scores = node_embs.bmm(scorer.unsqueeze(-1)).squeeze()/scorer.norm(dim=1).unsqueeze(-1)
        scores = scores.squeeze()
        vals, topk_indices = scores.view(batch_size,-1).topk(self.k,dim=1)
        topk_indices_out = topk_indices[vals > -float("Inf")].view(batch_size,-1)
        # if topk_indices.size(1) < self.k:
        #     topk_indices = u.pad_with_last_val(topk_indices,self.k)
        repeat_num = args.gcn_dim / self.k + 1
        topk_indices = topk_indices.repeat(1,int(repeat_num))[:,:args.gcn_dim]
        tanh = torch.nn.Tanh()
        tanh = torch.nn.Tanh()

        if isinstance(node_embs, torch.sparse.FloatTensor) or \
           isinstance(node_embs, torch.cuda.sparse.FloatTensor):
            node_embs = node_embs.to_dense()
        out = node_embs.gather(1,topk_indices.unsqueeze(-1).expand(-1,-1,feat_size)) * tanh(scores.gather(1,topk_indices)).unsqueeze(-1)
        scores = self.softmax(scores.view(batch_size,-1))
        entropy_object = torch.distributions.Categorical(scores).entropy()
        c = scores.log()
        score =  c.gather(-1,topk_indices)
        policy_score = score.mean(dim=1)
        #we need to transpose the output
        return out.transpose(1,2),policy_score,scorer,entropy_object,topk_indices_out