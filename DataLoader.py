import dgl
import torch
import pickle
import numpy as np
from ase.units import Hartree, eV

from Data import Molecule, lg_node_type, atom_type

dgl.load_backend('pytorch')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataLoader(object):
    def __init__(self, filename):
        ''' Load data
        '''
        self.data = pickle.load(open(filename, 'rb'))
        self.dg_node_type_universe = len(atom_type)
        self.lg_node_type_universe = len(lg_node_type)
        
        # build a reference for atoms, 2 bodies and 3 bodies
        self.dg_node_ref = np.zeros(len(self.data) + 1, dtype=np.int64)
        for i in range(len(self.data)):
            self.dg_node_ref[i+1] = self.dg_node_ref[i] + self.data[i].dg_num_nodes
            
        self.lg_node_ref = np.zeros(len(self.data) + 1, dtype=np.int64)
        for i in range(len(self.data)):
            self.lg_node_ref[i+1] = self.lg_node_ref[i] + self.data[i].lg_num_nodes
        
        self.trn_iter = 0
        self.val_iter = 0
        self.indices = np.arange(len(self.data), dtype=np.int64)
        
        self.y = torch.FloatTensor([mol.prpty for mol in self.data])
        
    def __len__(self):
        return len(self.data)
    
    def get_statistics(self):
        dg_mean, dg_count, lg_mean, lg_count = 0., 0., 0., 0.
        
        dg_per_diff = torch.zeros(self.dg_node_ref[-1], dtype=torch.float32)
        lg_per_diff = torch.zeros(self.lg_node_ref[-1], dtype=torch.float32)
        
        for i, m in enumerate(self.data):
            target_value = self.y[i]
            dg_per_diff[self.dg_node_ref[i]:self.dg_node_ref[i+1]] = target_value / m.dg_num_nodes
            lg_per_diff[self.lg_node_ref[i]:self.lg_node_ref[i+1]] = target_value / m.lg_num_nodes
            
            dg_mean += target_value
            lg_mean += target_value
            dg_count += m.dg_num_nodes
            lg_count += m.lg_num_nodes
        
        dg_mean = (dg_mean / dg_count).item()
        lg_mean = (lg_mean / lg_count).item()
        dg_std = torch.sqrt(torch.mean((dg_per_diff - dg_mean)**2)).item()
        lg_std = torch.sqrt(torch.mean((lg_per_diff - lg_mean)**2)).item()
        
        return dg_mean, lg_mean, dg_std, lg_std
        
    def next_batch(self, batch_size):
        ''' generate a sequential batch
        '''
        # when the remaining moleculars is larger than the batch_size
        if self.val_iter + batch_size < len(self.data):
            indices = np.arange(self.val_iter, self.val_iter+batch_size, dtype=np.int64)
            self.val_iter += batch_size
        else:
            indices = np.arange(self.val_iter, len(self.data), dtype=np.int64)
            self.val_iter = 0
        
        return self._generate_batch(indices)
        
    def next_random_batch(self, batch_size):
        ''' generate a random batch
        '''
        if self.trn_iter + batch_size < len(self.data):
            indices = self.indices[self.trn_iter:(self.trn_iter + batch_size)]
            self.trn_iter += batch_size
        else:
            np.random.shuffle(self.indices)
            indices = self.indices[:batch_size]
            self.trn_iter = batch_size
        
        return self._generate_batch(indices)
    
    def _generate_batch(self, indices):
        batch_g = dgl.batch_hetero([self.data[i].get_hetero_graph() for i in indices])
        
        dg_node_feat_continuous = torch.cat([self.data[i].get_dg_node_feat_continuous() for i in indices], dim=0)
        dg_node_feat_discrete = torch.cat([self.data[i].get_dg_node_feat_discrete() for i in indices], dim=0)
        lg_node_feat_continuous = torch.cat([self.data[i].get_lg_node_feat_continuous() for i in indices], dim=0)
        lg_node_feat_discrete = torch.cat([self.data[i].get_lg_node_feat_discrete() for i in indices], dim=0)
        dg_edge_feat = torch.cat([self.data[i].get_dg_edge_feat() for i in indices], dim=0)
        lg_edge_feat = [torch.cat([self.data[i].lg_edge_feat[k] for i in indices], dim=0) for k in range(self.data[0].n_space)]
        co_feat = torch.cat([self.data[i].graph_feat for i in indices], dim=0)
        mask_mat = torch.cat([self.data[i].mask_vec for i in indices], dim=0)

        y = self.y[indices]
        
        return batch_g, dg_node_feat_continuous, dg_node_feat_discrete, lg_node_feat_continuous, lg_node_feat_discrete, dg_edge_feat, lg_edge_feat, co_feat, mask_mat, y
