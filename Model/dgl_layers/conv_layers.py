import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

from ..torch_layers import ResLayer, DenseLayer, GlorotOrthogonal

from dgl.nn import edge_softmax, GATConv, GraphConv, GINConv
from dgl import DGLError
# from dgl.utils import Identity
from dgl.utils import expand_as_pair


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """Return input"""
        return x
    
    
class DSGATConv(nn.Module):
    def __init__(self,
                 in_node_feats,
                 in_edge_feats,
                 out_feats,
                 num_heads,
                 edge_type,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 merge=None,
                 init_para='xavier_normal',
                 residual=False,
                 activation=F.relu):
        super(DSGATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_node_feats)
        self._in_edge_feats = in_edge_feats
        self._out_feats = out_feats
        self._edge_type = edge_type
        self._merge = merge
        
        if isinstance(in_node_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        # edge features
        if self._in_edge_feats:
            self.attn_e = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
            self.fc_e = nn.Linear(self._in_edge_feats, out_feats * num_heads, bias=False)
        
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else: # bipartite graph neural networks
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        
        if self._in_edge_feats:
            nn.init.xavier_normal_(self.fc_e.weight, gain=gain)
            nn.init.xavier_normal_(self.attn_e, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, node_feat, edge_feat=None):
        graph = graph.local_var()
        graph = graph.edge_type_subgraph([self._edge_type])
        
        if isinstance(node_feat, tuple):
            h_src = self.feat_drop(node_feat[0])
            h_dst = self.feat_drop(node_feat[1])
            feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
            feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
        else:
            h_src = h_dst = self.feat_drop(node_feat)
            feat_src = feat_dst = self.fc(h_src).view(
                -1, self._num_heads, self._out_feats)
        
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)

        graph.srcdata.update({'ft': feat_src, 'el': el})
        graph.dstdata.update({'er': er})
        
        if self._in_edge_feats:
            h_edge = self.feat_drop(edge_feat)
            feat_edge = self.fc_e(h_edge).view(
                -1, self._num_heads, self._out_feats)
            ee = (feat_edge * self.attn_e).sum(dim=-1).unsqueeze(-1)
            graph.edges[self._edge_type].data.update({'ee': ee})
            graph.apply_edges(fn.e_add_u('ee', 'el', 'e'))
            graph.apply_edges(fn.e_add_v('e', 'er', 'e'))
        else:
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        
        e = self.leaky_relu(graph.edata.pop('e'))
        # compute softmax
        graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
        # message passing
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        rst = graph.dstdata['ft']
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
            rst = rst + resval
        # multi-head combine
        if self._merge == 'mean':
            rst = torch.mean(rst, dim=1)
        if self._merge == 'cat':
            rst = rst.view(-1, self._num_heads * self._out_feats)
        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst
    
    
class AngleOrientedConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 space_num,
                 feat_drop=0.,
                 attn_drop=0.,
                 model='GAT',
                 merge='cat',
                 fc_num=0,
                 init_para='xavier_normal',
                 residual=False,
                 activation=F.relu):
        super(AngleOrientedConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._space_num = space_num
        self._fc_num = fc_num
        self._merge = merge
        
        # GraphConv Layers
        self.conv_layer = nn.ModuleList()
        for i in range(space_num):
            if model == 'GAT':
                conv = GATConv(in_feats,
                               out_feats,
                               num_heads=1,
                               feat_drop=feat_drop,
                               attn_drop=attn_drop,
                               residual=residual,
                               activation=activation)
            if model == 'GCN':
                conv = GraphConv(in_feats,
                                 out_feats,
                                 norm='both',
                                 bias=True,
                                 activation=activation)
            if model == 'GATL':
                conv = GATLayer(in_feats,
                                out_feats,
                                feat_drop=feat_drop,
                                attn_drop=feat_drop,
                                transform=False,
                                residual=False,
                                activation=None)

            self.conv_layer.append(conv)
        
        # fc residual layers
        self.fc_layer = nn.ModuleList()
        for i in range(fc_num):
            self.fc_layer.append(nn.Linear(in_feats, out_feats, bias=True))
        for i in range(fc_num):
            nn.init.xavier_uniform_(self.fc_layer[i].weight)
        self.activation = activation

        
    def forward(self, graph, feat):
        graph = graph.local_var()
        
        h_list = []
        for k in range(self._space_num):
            edge_type = 'b2b_space_' + str(k)
            sub_g = graph.edge_type_subgraph([edge_type])
            h_list.append(self.conv_layer[k](sub_g, feat))
            # sub_g.nodes.data.clear()
            
        if self._merge == 'cat':
            feat_h = torch.cat(h_list, dim=-1)
        if self._merge == 'mean':
            feat_h = torch.mean(torch.stack(h_list, dim=-1), dim=-1)
        if self._merge == 'sum':
            feat_h = torch.sum(torch.stack(h_list, dim=-1), dim=-1)
            # print(feat_h.shape)
        if self._merge == 'max':
            feat_h = torch.max(torch.stack(h_list, dim=-1), dim=-1)
        if self._merge == 'cat_max':
            feat_h = torch.stack(h_list, dim=1)
            feat_max = torch.max(feat_h, dim=1)[0]
            feat_max = feat_max.view(-1, 1, self._out_feats)
            feat_h = (feat_h * feat_max).view(-1, self._space_num * self._out_feats)
        
        for fc in self.fc_layer:
            feat_h = fc(feat_h)
            if self.activation:
                feat_h = self.activation(feat_h)
                
        return feat_h

    
class GATLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 feat_drop=0.,
                 attn_drop=0.,
                 residual=False,
                 transform=False,
                 activation=None):
        super(GATLayer, self).__init__()
        self.transform = transform
        if self.transform:
            self.fc = nn.Linear(in_feats, out_feats, bias=False)
            attn_in_feats = 2 * out_feats
        else:
            attn_in_feats = 2 * in_feats
        self.attn_fc = nn.Linear(attn_in_feats, out_feats, bias=True)
        self.attn_out = nn.Linear(out_feats, 1, bias=False)

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.tanh = nn.Tanh()
        if residual:
            if in_feats != out_feats:
                self.res_fc = nn.Linear(
                    in_feats, out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('tanh')
        
        if self.transform:
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_out.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=nn.init.calculate_gain('relu'))
            
    def edge_attention(self, edges):
        h_c = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        h_c = self.attn_fc(h_c)
        h_c = self.tanh(h_c)
        h_s = self.attn_out(h_c)
        return {'e': h_s}
    
    def message_func(self, edges):
        return {'h': edges.src['h'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        alpha = self.attn_drop(alpha)
        z = torch.sum(alpha * nodes.mailbox['h'], dim=1)
        return {'z': z}
    
    def forward(self, graph, feat):
        graph = graph.local_var()
        feat = self.feat_drop(feat)
        if self.transform:
            feat = self.fc(feat)
        
        graph.ndata['h'] = feat
        graph.apply_edges(self.edge_attention)
        graph.update_all(self.message_func, self.reduce_func)
        rst = graph.ndata.pop('h')

        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
            rst = rst + resval
        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst