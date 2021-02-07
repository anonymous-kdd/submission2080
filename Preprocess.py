import os
import pickle
import argparse
import numpy as np

from tqdm import tqdm
from Data import Molecule


def preprocess(data_path, dataset_name, cut_r, n_space, flag='train'):
    '''load'''
    filename = os.path.join(data_path, "{0}_{1}.pickle".format(dataset_name, flag))
    with open(filename, 'rb') as reader:
        data_g, data_f, data_y = pickle.load(reader)
    print(filename, 'loaded.')

    data = []
    for g, f, y in tqdm(zip(data_g, data_f, data_y)):
        num_nodes, num_nodes_d, features, edges, coords, atoms, atoms_full, dist = g
        assert num_nodes_d <= num_nodes
        assert len(coords) == len(atoms) and len(coords) == len(atoms_full)
        mol = Molecule(num_nodes, coords, atoms, features, num_nodes_d, atoms_full, f, y, cut_r, n_space)
        if not mol.error:
            data.append(mol)
        else:
            print('error')
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATADIR', type=str, default='./data/PDBbind/',
                        help='directory to the data.')
    parser.add_argument('--dataset', type=str, default='refined',
                        help='directory to the data.')
    parser.add_argument('--target_dir', type=str, default='./data/processed/',
                        help='directory to save the data.')
    parser.add_argument('--cut_r', type=float, default=5.,
                        help='cut off distance.')
    parser.add_argument('--n_space', type=int, default=6,
                        help='size of angle space.')
    args = parser.parse_args()
    
    if not os.path.isdir(args.target_dir):
        os.mkdir(args.target_dir)
        
    dataset_name = args.dataset + '_cut' + str(int(args.cut_r))
    data = preprocess(args.DATADIR, dataset_name, args.cut_r, args.n_space, flag='test')
    pickle.dump(data, open(os.path.join(args.target_dir, args.dataset, 'test_ca_r%d_n%d.data' % (args.cut_r, args.n_space)), 'wb'))
    del data
    
    data = preprocess(args.DATADIR, dataset_name, args.cut_r, args.n_space, flag='train')
    pickle.dump(data, open(os.path.join(args.target_dir, args.dataset, 'train_ca_r%d_n%d.data' % (args.cut_r, args.n_space)), 'wb'))
    del data
    
    data = preprocess(args.DATADIR, dataset_name, args.cut_r, args.n_space, flag='valid')
    pickle.dump(data, open(os.path.join(args.target_dir, args.dataset, 'val_ca_r%d_n%d.data' % (args.cut_r, args.n_space)), 'wb'))
    del data
    