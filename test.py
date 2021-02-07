import os
import time
import math
import argparse
import random
import numpy as np

# import dgl
import torch
import torch.nn.functional as F
from math import sqrt
from sklearn.linear_model import LinearRegression

from Model import SIGN
from Model.torch_layers import shifted_softplus
from Data import Molecule, atom_type, lg_node_type
from DataLoader import DataLoader
import warnings
warnings.filterwarnings("ignore")

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse

def mae(y,f):
    mae = (np.abs(y-f)).mean()
    return mae

def sd(y,f):
    f,y = f.reshape(-1,1),y.reshape(-1,1)
    lr = LinearRegression()
    lr.fit(f,y)
    y_ = lr.predict(f)
    sd = (((y - y_) ** 2).sum() / (len(y) - 1)) ** 0.5
    return sd

def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.backends.cudnn.benchmark = False
    # dgl.random.seed(seed)
    # torch.cuda.manual_seed(seed)

def evaluate(model, data, device, batch_size=32, return_attn=False):
    model.eval()
    with torch.no_grad():
        n_batch = len(data) // batch_size
        n_batch = (n_batch + 1) if (len(data) % batch_size) != 0 else n_batch
        
        y_hat_list = []
        y_list = []
        
        for i in range(n_batch):
            batch_hg, dg_node_feat_continuous, dg_node_feat_discrete, lg_node_feat_continuous,\
            lg_node_feat_discrete, dg_edge_feat, lg_edge_feat, co_feat, mask_mat, y = data.next_batch(batch_size)
            
            # batch_hg = batch_hg.to(device)
            dg_node_feat_discrete = dg_node_feat_discrete.to(device)
            lg_node_feat_discrete = lg_node_feat_discrete.to(device)
            dg_node_feat_continuous = dg_node_feat_continuous.to(device)
            lg_node_feat_continuous = lg_node_feat_continuous.to(device)
            dg_edge_feat = dg_edge_feat.to(device)
            lg_edge_feat = [feat.to(device) for feat in lg_edge_feat]
            co_feat = co_feat.to(device)
            mask_mat = mask_mat.to(device)
            y = y.to(device)
            
            y_hat, _, _ = model(batch_hg,
                          dg_node_feat_continuous,
                          dg_node_feat_discrete,
                          lg_node_feat_continuous,
                          lg_node_feat_discrete,
                          dg_edge_feat,
                          lg_edge_feat,
                          mask_mat)
            
            y_hat_list += y_hat.tolist()
            y_list += y.tolist()
        
        y_hat = np.array(y_hat_list).reshape(-1,)
        y = np.array(y_list).reshape(-1,)
        return rmse(y, y_hat), mae(y, y_hat), sd(y, y_hat), pearson(y, y_hat)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/processed/CSAR-HiQ/')
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--restore', type=int, default=0)
    parser.add_argument('--model_dir', type=str, default='./output/pretrained_model')
    parser.add_argument('--act', type=str, default='relu')
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--loss_fn', type=str, default='l1')
    parser.add_argument('--seed', type=int, default=123)
    
    parser.add_argument("--feat_drop", type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_dec_rate", type=float, default=0.5)
    parser.add_argument("--eva_steps", type=int, default=100)
    parser.add_argument("--dec_steps", type=int, default=20000)
    parser.add_argument('--tol_steps', type=int, default=20000)
    parser.add_argument('--all_steps', type=int, default=300000)

    parser.add_argument("--num_convs", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--infeat_dim", type=int, default=36)
    parser.add_argument("--conv_dim", type=int, default=128)
    parser.add_argument("--dense_dims", type=str, default='128*4,128*2,128')
    
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--cut_r', type=float, default=5.)
    parser.add_argument('--n_space', type=float, default=6)
    parser.add_argument('--model_b2b', type=str, default='GATL')
    parser.add_argument('--merge_b2b', type=str, default='cat')
    parser.add_argument('--merge_b2a', type=str, default='mean')
    
    parser.add_argument('--return_attn', type=int, default=0)
    parser.add_argument("--residual", action="store_true", default=True)
    parser.add_argument("--num_interaction_residual", type=int, default=1,
                        help="number of residual layers for message output")
    parser.add_argument("--num_atom_residual", type=int, default=1,
                        help="number of residual layers for node output")
    
    args = parser.parse_args()
    if args.seed:
        setup_seed(args.seed)
    
    device = torch.device("cuda:" + args.cuda if torch.cuda.is_available() else "cpu")
    dense_dims = [eval(dim) for dim in args.dense_dims.split(',')]
    activation = shifted_softplus if args.act == 'softplus' else F.relu
    model_dir = args.model_dir
    
    model = SIGN(num_convs=args.num_convs,
                dense_dims=dense_dims,
                conv_dim=args.conv_dim,
                infeat_dim=args.infeat_dim,
                hidden_dim=args.hidden_dim,
                dg_node_type_universe=len(atom_type),
                lg_node_type_universe=len(lg_node_type),
                n_space=args.n_space,
                cut_r=args.cut_r,
                model_b2b=args.model_b2b,
                merge_b2b=args.merge_b2b,
                merge_b2a=args.merge_b2a,
                activation=activation,
                num_heads=args.num_heads,
                feat_drop=args.feat_drop).to(device)

    tst_data = DataLoader(os.path.join(args.data_dir, 'test_ca_r%d_n%d.data' % (args.cut_r, args.n_space)))
    state_dict = torch.load(os.path.join(model_dir, 'Best_model.pt'))
    model.load_state_dict(state_dict['model_state_dict'])
    rmse_tst, mae_tst, sd_tst, r_tst = evaluate(model, tst_data, device, args.batch_size)
    print('%s - RMSE: %.6f, MAE: %.6f, SD: %.6f, R: %.6f.' % (args.data_dir, rmse_tst, mae_tst, sd_tst, r_tst))