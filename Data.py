import dgl
import numpy as np
import math
import torch
from scipy.spatial import distance
from scipy.sparse import coo_matrix

atom_type = ['B', 'C', 'N', 'O', 'P', 'S', 'Se', 'halogen', 'metal']

lg_node_type = []
lg_node_index = {}

for i in range(len(atom_type)):
    for j in range(i, len(atom_type)):
        lg_node_index[(i, j)] = len(lg_node_type)
        lg_node_type.append((i, j))


prot_atom_ids = [6, 7, 8, 16]
drug_atom_ids = [6, 7, 8, 9, 15, 16, 17, 35, 53]
pair_ids = [(i, j) for i in prot_atom_ids for j in drug_atom_ids]
# pair_ids = {}

for i in range(len(atom_type)):
    for j in range(i, len(atom_type)):
        lg_node_index[(i, j)] = len(lg_node_type)
        lg_node_type.append((i, j))

def my_float(num):
    ''' my function to convert complicated string to float
    '''
    try:
        return float(num)
    except:
        pos = num.find('*^')
        base = num[:pos]
        exp = num[pos+2:]
        return float(base + 'e' + exp)

def setxor(a, b):
    n = len(a)
    
    res = []
    link = []
    i, j = 0, 0
    while i < n and j < n:
        if a[i] == b[j]:
            link.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            res.append(a[i])
            i += 1
        else:
            res.append(b[j])
            j += 1
    
    if i < j:
        res.append(a[-1])
    elif i > j:
        res.append(b[-1])
    else:
        link.append(a[-1])
    
    return res, link

class Molecule(object):
    def __init__(self, num_nodes, coords, atoms, feats, drug_nodes, full_atoms, co_feats, y, cut_r, n_space):
        self.cut_r = cut_r
        self.n_space = n_space
        self.na = num_nodes # number of atoms
        self.nd = drug_nodes
        self.prpty = y
        self.error = False
        
        self.atoms = np.array(atoms)
        self.feats = feats
        self.coordinates = coords
        
        self.dist = distance.cdist(self.coordinates, self.coordinates, 'euclidean')
        np.fill_diagonal(self.dist, np.inf)
        
        self.full_atoms = full_atoms 
        self.co_feats = np.array([co_feats]) # [co_feats[(i, j)] for i in prot_atom_ids for j in drug_atom_ids]
        # self.co_feats = (self.co_feats - self.co_feats.mean()) / self.co_feats.std()
        self.co_feats = self.co_feats / self.co_feats.sum() # normalized feature

        self._build_hetero_graph()
    
    def _build_hetero_graph(self):
        ################################
        # build the atom to atom graph #
        ################################
        self.dg_num_nodes = self.na
        self.dg_node_feat_discrete = torch.LongTensor(self.atoms)
        self.dg_node_feat_continuous = torch.FloatTensor(self.feats)
        self.graph_feat = torch.FloatTensor(self.co_feats)
        
        dist_graph_base = self.dist.copy()
        self.dg_edge_feat = torch.FloatTensor(dist_graph_base[dist_graph_base < self.cut_r]).unsqueeze(1)
        dist_graph_base[dist_graph_base >= self.cut_r] = 0.
        
        atom_graph = coo_matrix(dist_graph_base)
        
        ################################
        # build the bond to bond graph #
        ################################
        num_atoms = self.dist.shape[0]
        bond_feat_discrete = []
        bond_feat_continuous = []
        indices = []
        inter_edges_dict = {k: [] for k in pair_ids}
        for i in range(num_atoms):
            for j in range(num_atoms):
                a = self.dist[i, j]
                
                if a < self.cut_r:
                    at_i, at_j = self.full_atoms[i], self.full_atoms[j]
                    if i < self.nd and j >= self.nd and (at_j, at_i) in inter_edges_dict:
                        inter_edges_dict[(at_j, at_i)] += [(len(indices), j)]
                    if i >= self.nd and j < self.nd and (at_i, at_j) in inter_edges_dict:
                        inter_edges_dict[(at_i, at_j)] += [(len(indices), j)]

                    bond_feat_continuous.append([a])
                    indices.append([i, j])
                    tp = tuple(sorted(self.atoms[[i, j]]))
                    bond_feat_discrete.append(lg_node_index[tp]) # two directional bonds share the same type (i.e., C-N)
        
        num_bonds = len(indices)
        self.lg_num_nodes = num_bonds
        self.lg_node_feat_discrete = torch.LongTensor(bond_feat_discrete)
        self.lg_node_feat_continuous = torch.FloatTensor(bond_feat_continuous)
        
        if self.cut_r == 3:
            for x1, x2 in zip(bond_feat_continuous, self.dg_edge_feat):
                assert x1[0]-x2.numpy()[0] < 1e-6
        
        #######################################################
        # build the bond to atom graph #
        #######################################################
        assignment_b2a = np.zeros((num_bonds, num_atoms), dtype=np.int64) # Maybe need too much memory
        assignment_a2b = np.zeros((num_atoms, num_bonds), dtype=np.int64) # Maybe need too much memory
        for i, idx in enumerate(indices):
            assignment_b2a[i, idx[1]] = 1
            assignment_a2b[idx[0], i] = 1

        bond2atom_graph = coo_matrix(assignment_b2a)
        
        ################################
        # build the bond to bond graph #
        ################################
        bond_graph_base = assignment_b2a @ assignment_a2b
        np.fill_diagonal(bond_graph_base, 0) # eliminate self connections
        # bond_graph = dgl.graph(csr_matrix(bond_graph_base), 'bond', 'b2b')
        
        ##############################################
        # build edge feature for the bond2bond graph #
        ##############################################
        x, y = np.where(bond_graph_base > 0)
        num_edges = len(x)
        edge_feat_continuous = np.zeros_like(x, dtype=np.float32)

        for i in range(num_edges):
            body1 = indices[x[i]]
            body2 = indices[y[i]]
            
            bodyxor, link = setxor(body1, body2)
            
            a = self.dist[body1[0], body1[1]]
            b = self.dist[body2[0], body2[1]]
            c = self.dist[bodyxor[0], bodyxor[1]]
            
            if a == 0 or b == 0:
                self.error = True
                return
            else:
                edge_feat_continuous[i] = self._cos_formula(a, b, c) # calculate the cos value of the angle (-1, 1)
                
        # self.lg_edge_feat = torch.FloatTensor(edge_feat_continuous).unsqueeze(1)

        ##############################################
        # build angle-oriented bond2bond graph #
        ##############################################
        # devide edges into different spaces
        unit = 180.0 / self.n_space
        # angle_feat = np.rad2deg(edge_feat_continuous)
        # angle_feat = (180.0 - angle_feat + 0.001) / unit
        angle_feat = np.rad2deg(edge_feat_continuous) / unit
        angle_feat = angle_feat.astype('int64')
        angle_feat = np.clip(angle_feat, 0, self.n_space-1)

        src_ids_list = [[] for _ in range(self.n_space)]
        dst_ids_list = [[] for _ in range(self.n_space)]
        lg_radian_list = [[] for _ in range(self.n_space)]
        for i, (space_idx, radian) in enumerate(zip(angle_feat, edge_feat_continuous)):
            src_ids_list[space_idx].append(x[i])
            dst_ids_list[space_idx].append(y[i])
            lg_radian_list[space_idx].append(radian)
        
        graph_dict = {
        ('atom', 'a2a', 'atom'): (torch.tensor(atom_graph.row, dtype=torch.int64), torch.tensor(atom_graph.col, dtype=torch.int64)),
        ('bond', 'b2a', 'atom'): (torch.tensor(bond2atom_graph.row, dtype=torch.int64), torch.tensor(bond2atom_graph.col, dtype=torch.int64))}
        
        mask_vec = []
        for k in pair_ids:
            v = inter_edges_dict[k]
            mask = 0
            if len(v) == 0:
                v = [(0, 0)]
                mask = 1
            mask_vec += [mask]
            graph_k = ('bond', 'b2a_'+str(k[0])+str(k[1]), 'atom')
            src_ids = torch.tensor([x[0] for x in v], dtype=torch.int64)
            dst_ids = torch.tensor([x[1] for x in v], dtype=torch.int64)
            graph_dict.update({graph_k: (src_ids, dst_ids)})
        self.mask_vec = torch.BoolTensor([mask_vec])
        # assert sum(mask_vec) < len(pair_ids)

        self.lg_edge_feat = []
        for space_idx in range(self.n_space):
            src_ids = torch.tensor(src_ids_list[space_idx], dtype=torch.int64)
            dst_ids = torch.tensor(dst_ids_list[space_idx], dtype=torch.int64)
            graph_k = ('bond', 'b2b_space_' + str(space_idx), 'bond')
            graph_dict.update({graph_k: (src_ids, dst_ids)})
            self.lg_edge_feat.append(torch.FloatTensor(lg_radian_list[space_idx]).unsqueeze(1))

        self.hetero_graph = dgl.heterograph(graph_dict)
    
    def _cos_formula(self, a, b, c):
        ''' formula to calculate the angle between two edges
            a and b are the edge lengths, c is the angle length.
        '''
        res = (a**2 + b**2 - c**2) / (2 * a * b)

        # sanity check
        res = -1. if res < -1. else res
        res = 1. if res > 1. else res
        return np.arccos(res)
    
    def get_hetero_graph(self):
        return self.hetero_graph
    
    def get_dg_node_feat_continuous(self):
        return self.dg_node_feat_continuous
    
    def get_dg_node_feat_discrete(self):
        return self.dg_node_feat_discrete
    
    def get_lg_node_feat_continuous(self):
        return self.lg_node_feat_continuous
        # return self.dg_edge_feat
    
    def get_lg_node_feat_discrete(self):
        return self.lg_node_feat_discrete
    
    def get_dg_edge_feat(self):
        return self.dg_edge_feat
    
    def get_lg_edge_feat(self):
        return self.lg_edge_feat
    
    def get_dg_num_nodes(self):
        return self.dg_num_nodes
    
    def get_lg_num_nodes(self):
        return self.lg_num_nodes
