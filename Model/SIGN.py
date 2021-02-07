import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import dgl
import dgl.function as fn
import dgl.backend as bknd

from .dgl_layers import DSGATConv, AngleOrientedConv
from .torch_layers import ResLayer, DenseLayer, DistRBF, AngleRBF, ShrinkDistRBF, GlorotOrthogonal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

prot_atom_ids = [6, 7, 8, 16]
drug_atom_ids = [6, 7, 8, 9, 15, 16, 17, 35, 53]

def sum_hetero_nodes(bg, node_type, feats):
    batch_size = bg.batch_size
    batch_num_nodes = bg.batch_num_nodes(node_type)

    seg_id = torch.from_numpy(np.arange(batch_size, dtype='int64').repeat(batch_num_nodes))
    seg_id = seg_id.to(feats.device)

    return bknd.unsorted_1d_segment_sum(feats, seg_id, batch_size, 0)

def sum_hetero_edges(bg, edge_type, feats):
    batch_size = bg.batch_size
    batch_num_edges = bg.batch_num_edges(edge_type)

    seg_id = torch.from_numpy(np.arange(batch_size, dtype='int64').repeat(batch_num_edges))
    seg_id = seg_id.to(feats.device)

    return bknd.unsorted_1d_segment_sum(feats, seg_id, batch_size, 0)

class DistGraphInputModule(nn.Module):
    def __init__(self, node_type_universe, edge_continuous_dim, hidden_dim, cut_r, activation, initial='glorot'):
        super(DistGraphInputModule, self).__init__()
        
        # function to convert discrete node feature (atomic number) to continuous node features
        self.node_embedding_layer = nn.Embedding(node_type_universe, hidden_dim)
        self.node_input_layer = DenseLayer(hidden_dim, hidden_dim, activation, bias=True)
        
        self.dist_embedding_layer = nn.Embedding(int(cut_r)-1, hidden_dim)
        self.dist_input_layer = DenseLayer(hidden_dim, hidden_dim, activation, bias=True)
        
        # radial basis function to convert edge distance to continuous feature
        self.edge_rbf = ShrinkDistRBF(edge_continuous_dim, cut_r)
        self.edge_input_layer = DenseLayer(edge_continuous_dim, hidden_dim, activation, bias=True)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # initialize parameters for nodes
        self.node_embedding_layer.weight.data.uniform_(-math.sqrt(3), math.sqrt(3))
        self.dist_embedding_layer.weight.data.uniform_(-math.sqrt(3), math.sqrt(3))
        
    def forward(self, node_feat_continuous, node_feat_discrete, edge_feat_continuous):
        h = self.node_embedding_layer(node_feat_discrete)
        h = self.node_input_layer(h)
        h_in = node_feat_continuous
        
        dist = torch.clamp(edge_feat_continuous.squeeze(), 1.0, 4.99999).type(torch.int64) - 1
        eh_emb = self.dist_embedding_layer(dist)
        eh_emb = self.dist_input_layer(eh_emb)
        
        eh_rbf = self.edge_rbf(edge_feat_continuous)
        eh_rbf = self.edge_input_layer(eh_rbf)
        return h, h_in, eh_rbf, eh_emb

    
class LineGraphInputModule(nn.Module):
    def __init__(self, node_type_universe, node_continuous_dim, edge_continuous_dim, hidden_dim, cut_r, activation):
        super(LineGraphInputModule, self).__init__()
        
        # function to convert discrete body feature to continuous body features
        self.node_embedding_layer = nn.Embedding(node_type_universe, hidden_dim)
        
        # radial basis function to convert body distance to continuous feature
        self.node_rbf = DistRBF(node_continuous_dim, cut_r)
        self.node_input_layer = DenseLayer(hidden_dim + node_continuous_dim, hidden_dim, activation, bias=True)
        
        # radial basis function to convert edge distance to continuous feature
        self.edge_rbf = AngleRBF(edge_continuous_dim)
        self.edge_input_layer = DenseLayer(edge_continuous_dim, hidden_dim, activation, bias=True)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # initialize parameters for nodes
        self.node_embedding_layer.weight.data.uniform_(-math.sqrt(3), math.sqrt(3))
        
    def forward(self, node_feat_continuous, node_feat_discrete, edge_feat_continuous):
        h_discrete = self.node_embedding_layer(node_feat_discrete)
        h_continuous = self.node_rbf(node_feat_continuous)
        
        h = torch.cat([h_discrete, h_continuous], dim=1)
        h = self.node_input_layer(h)
        
        eh = [self.edge_rbf(angle_feat) for angle_feat in edge_feat_continuous]
        eh = [self.edge_input_layer(eh_) for eh_ in eh]
        
        return h, eh

    
class OutputModule(nn.Module):
    def __init__(self, node_type, node_type_universe, hidden_dim, activation, mean, std):
        super(OutputModule, self).__init__()
        self.domestic_node_type = node_type
        
        # output layer which output the final prediction value (for regression)
        self.node_out_residual = nn.ModuleList()
        self.node_out_layer = nn.Linear(hidden_dim, 1, bias=True)
        
        # scale layer (Important for scaling the output)
        self.node_out_scale = nn.Embedding(node_type_universe, 1)
        self.node_out_bias = nn.Embedding(node_type_universe, 1)
        
        self.mean = nn.Parameter(torch.FloatTensor([mean]))
        self.std = nn.Parameter(torch.FloatTensor([std]))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.node_out_bias.weight.data.zero_()
        self.node_out_scale.weight.data.zero_()
        
        GlorotOrthogonal(self.node_out_layer.weight.data)
        self.node_out_layer.bias.data.zero_()
    
    def forward(self, batch_g, h, node_feat_discrete):
        node_out = self.node_out_layer(h)
        node_scale = self.node_out_scale(node_feat_discrete)
        node_bias = self.node_out_bias(node_feat_discrete)
        node_out = node_scale * node_out + node_bias
        
        node_out = node_out * self.std + self.mean
        
        node_score = sum_hetero_nodes(batch_g, self.domestic_node_type, node_out)
        graph_feat = sum_hetero_nodes(batch_g, self.domestic_node_type, h)
        
        return graph_feat, node_score

    
class AggBondModule(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, activation, initial='glorot'):
        super(AggBondModule, self).__init__()
        in_dim = node_feat_dim * 2 + edge_feat_dim
        self.agg_fc = DenseLayer(in_dim, edge_feat_dim, activation=activation, bias=True)
    
    def forward(self, batch_g, node_feat, edge_feat):
        atom_graph = batch_g.edge_type_subgraph(['a2a']).local_var()
        atom_graph.nodes['atom'].data.update({'h': node_feat})
        atom_graph.apply_edges(fn.copy_u('h', 'eu'))
        atom_graph.apply_edges(fn.u_add_v('h', 'h', 'euv'))
        
        src_feat = atom_graph.edges['a2a'].data['eu']
        dst_feat = atom_graph.edges['a2a'].data['euv'] - src_feat
        # print(src_feat.shape, dst_feat.shape, edge_feat.shape)
        bond_feat = torch.cat([src_feat, dst_feat, edge_feat], dim=1)
        bond_feat = self.agg_fc(bond_feat)
        
        return bond_feat
    

class BondOutputModule(nn.Module):
    def __init__(self, in_dim, hidden_dim, graph_fc=False, space_num=6, bo_merge='cat'):
        super(BondOutputModule, self).__init__()
        self.in_dim = in_dim
        self.bo_merge = bo_merge
        self.graph_fc = graph_fc
        self.space_num = space_num
        self.etype_list = ['b2a_'+str(i)+str(j) for i in prot_atom_ids for j in drug_atom_ids]

        if bo_merge == 'fc':
            fc_in_dim = space_num * in_dim
            self.fc_1 = DenseLayer(fc_in_dim, hidden_dim, activation=F.relu, bias=True)
        if graph_fc:
            fc_in_dim = space_num * in_dim if bo_merge == 'cat' else in_dim
            fc_in_dim = hidden_dim if bo_merge == 'fc' else fc_in_dim
            self.fc_2 = DenseLayer(fc_in_dim, hidden_dim, activation=F.relu, bias=True)
            # self.fc_2 = nn.ModuleList()
            # for _ in range(len(self.etype_list)):
            #     self.fc_2.append(DenseLayer(fc_in_dim, hidden_dim, activation=F.relu, bias=True))

        fc_in_dim = space_num * in_dim if bo_merge == 'cat' and not graph_fc else hidden_dim
        self.softmax = nn.Softmax(dim=1)
        # output layer which output the final prediction value (for regression)
        self.output_layer = nn.Linear(fc_in_dim, 1, bias=False)


    def forward(self, batch_g, h, mask_mat):
        if self.bo_merge != 'cat':
            h = h.view(-1, self.space_num, self.in_dim)
        if self.bo_merge == 'mean':
            h = torch.mean(h, dim=1)
        if self.bo_merge == 'sum':
            h = torch.sum(h, dim=1)
        if self.bo_merge == 'max':
            h = torch.max(h, dim=1)[0]
        if self.bo_merge == 'fc':
            h = self.fc_1(h.view(-1, self.space_num * self.in_dim))

        graph_vl = []
        for i, etype in enumerate(self.etype_list):
            graph = batch_g.edge_type_subgraph([etype]).local_var()
            # print(graph, h.shape)
            graph.nodes['bond'].data.update({'h': h})
            graph.apply_edges(fn.copy_u('h', 'eh'))
            eh = graph.edata.pop('eh')
            graph_feat = sum_hetero_edges(batch_g, etype, eh) # (bs, hidden_dim)
            if self.graph_fc:
                graph_feat = self.fc_2(graph_feat) # (bs, hidden_dim), optional
            graph_v = self.output_layer(graph_feat) # (bs, 1)
            graph_vl.append(graph_v)
            # graph.nodes['bond'].data.clear()
            # graph.edges[etype].data.clear()

        graph_vl = torch.stack(graph_vl, dim=1).squeeze() # (bs, num_etype, 1) -> (bs, num_etype)
        graph_vl = graph_vl.masked_fill(mask=mask_mat, value=torch.tensor(-1e9))
        graph_vl = self.softmax(graph_vl) # (bs, num_etype, 1)
        return graph_vl
        
    
class DenseOutputModule(nn.Module):
    def __init__(self, in_dim, hidden_dims, feat_drop=0.2, initial='glorot'):
        super(DenseOutputModule, self).__init__()

        self.mlp = nn.ModuleList()
        for out_dim in hidden_dims:
            self.mlp.append(DenseLayer(in_dim, out_dim, activation=F.relu, bias=True))
            in_dim = out_dim
        self.feat_drop = nn.Dropout(feat_drop)
        # output layer which output the final prediction value (for regression)
        self.output_layer = nn.Linear(in_dim, 1, bias=True)

        self.softmax = nn.Softmax(dim=1)
        self.out_layer = nn.Linear(in_dim, 36, bias=True)
        
    def reset_parameters(self):
        GlorotOrthogonal(self.node_out_layer.weight.data)
        self.node_out_layer.bias.data.zero_()

    def forward(self, batch_g, h):
        graph_feat = sum_hetero_nodes(batch_g, 'atom', h)
        for layer in self.mlp:
            graph_feat = layer(graph_feat)
            # graph_feat = self.feat_drop(graph_feat)
            
        output = self.output_layer(graph_feat)
        out_scores = self.out_layer(graph_feat)
        # out_scores = self.softmax(out_scores)
        return output, out_scores
    
    
class SIGN(nn.Module):
    def __init__(self,
                 num_convs,
                 dense_dims,
                 conv_dim,
                 infeat_dim,
                 hidden_dim,
                 dg_node_type_universe,
                 lg_node_type_universe,
                 n_space,
                 cut_r,
                 model_b2b='GAT',
                 merge_b2b='cat',
                 merge_b2a='mean',
                 activation=F.relu,
                 rbf_dim=64,
                 num_heads=4,
                 feat_drop=0.):
        
        super(SIGN, self).__init__()
        self.num_convs = num_convs
        self.activation = activation
        
        self.dg_input_module = DistGraphInputModule(dg_node_type_universe, rbf_dim, hidden_dim, cut_r, activation)
        # self.lg_input_module = LineGraphInputModule(lg_node_type_universe, rbf_dim, rbf_dim, hidden_dim, cut_r, activation)
        
        self.atom2bond_layer = nn.ModuleList()
        self.bond2bond_layer = nn.ModuleList()
        self.bond2atom_layer = nn.ModuleList()

        if conv_dim == 0:
            conv_dim = infeat_dim
        atom_in_dim = infeat_dim
        layer_in_dim = infeat_dim # conv_dim
        layer_out_dim = hidden_dim
        for _ in range(num_convs):
            
            self.atom2bond_layer.append(
                    AggBondModule(
                        node_feat_dim=layer_in_dim,
                        edge_feat_dim=layer_out_dim,
                        activation=activation)
                )
            
            layer_in_dim = hidden_dim
            layer_out_dim = conv_dim
            
            self.bond2bond_layer.append(
                    AngleOrientedConv(
                        in_feats=layer_in_dim,
                        out_feats=layer_out_dim,
                        space_num=n_space,
                        feat_drop=feat_drop,
                        attn_drop=feat_drop,
                        model=model_b2b,
                        merge=merge_b2b,
                        fc_num=0,
                        activation=activation)
                    )
            
            layer_out_dim = conv_dim
            if merge_b2b == 'cat' or merge_b2b == 'cat_max':
                layer_in_dim = (conv_dim * n_space, atom_in_dim)
            else:
                layer_in_dim = (conv_dim, atom_in_dim)
            
            self.bond2atom_layer.append(
                    DSGATConv(
                        in_node_feats=layer_in_dim,
                        in_edge_feats=hidden_dim,
                        out_feats=layer_out_dim,
                        num_heads=num_heads,
                        edge_type='b2a',
                        feat_drop=feat_drop,
                        attn_drop=feat_drop,
                        merge=merge_b2a,
                        activation=activation)
                    )
            
            if merge_b2a == 'cat':
                layer_in_dim = conv_dim * num_heads
                atom_in_dim = conv_dim * num_heads
            else:
                layer_in_dim = conv_dim
                atom_in_dim = conv_dim
            layer_out_dim = hidden_dim
            
        self.dg_output_module = DenseOutputModule(layer_in_dim, dense_dims)
        self.bd_output_module = BondOutputModule(conv_dim, conv_dim, space_num=6, bo_merge='fc', graph_fc=False)
   
    def forward(self,
                batch_hg,
                dg_node_feat_continuous,
                dg_node_feat_discrete, 
                lg_node_feat_continuous,
                lg_node_feat_discrete,
                dg_edge_feat,
                lg_edge_feat,
                mask_mat):
        
        dg_h, dg_h_in, dg_eh_rbf, dg_eh_emb = self.dg_input_module(dg_node_feat_continuous,
                                                                   dg_node_feat_discrete,
                                                                   dg_edge_feat)

        atom_h = dg_h_in
        for i in range(self.num_convs):
            bond_h = self.atom2bond_layer[i](batch_hg, atom_h, dg_eh_emb)
            bond_h = self.bond2bond_layer[i](batch_hg, bond_h)
            atom_h = self.bond2atom_layer[i](batch_hg, (bond_h, atom_h), dg_eh_emb)

        pred_socre, atom_inter_scores = self.dg_output_module(batch_hg, atom_h)
        bond_inter_scores = self.bd_output_module(batch_hg, bond_h, mask_mat)

        return pred_socre, atom_inter_scores, bond_inter_scores
