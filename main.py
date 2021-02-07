import os
import time
import math
import argparse
import random
import numpy as np

# import dgl
import torch
import torch.nn.functional as F

from Model import SIGN
from Model.torch_layers import shifted_softplus
from Data import Molecule
from DataLoader import DataLoader
from Util import load_model_state, save_model_state, evaluate, lr_scheduler, post_op_process

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

def train(batch_hg, loss_fn,
          dg_node_feat_continuous,
          dg_node_feat_discrete,
          lg_node_feat_continuous,
          lg_node_feat_discrete,
          dg_edge_feat,
          lg_edge_feat, co_feat, mask_mat, y, model, optimizer):
    
    model.train()
    batch_size = batch_hg.batch_size
    
    optimizer.zero_grad()
    y_hat, a_scores, b_scores = model(batch_hg,
                                    dg_node_feat_continuous,
                                    dg_node_feat_discrete,
                                    lg_node_feat_continuous,
                                    lg_node_feat_discrete,
                                    dg_edge_feat,
                                    lg_edge_feat,
                                    mask_mat)

    # L1 LOSS or L2 LOSS
    if loss_fn == 'l1':
        loss = F.l1_loss(y_hat.squeeze(), y, reduction='sum')
    else:
        loss = F.mse_loss(y_hat.squeeze(), y, reduction='mean')

    # loss += 1.0 * F.cross_entropy(a_scores.squeeze(), co_feat, reduction='sum')
    # print(loss)
    # print(b_scores.sum(), co_feat.sum())
    # loss += 1 * F.kl_div(a_scores, co_feat, reduction='sum')
    loss += 3.0 / 2 * F.l1_loss(b_scores, co_feat, reduction='sum')
    loss_b = 1.0 / 1 * F.l1_loss(b_scores, co_feat, reduction='sum')
    # loss_b = loss
    
    if math.isnan(loss.item()) or math.isinf(loss.item()):
        raise RuntimeError('Something is wrong with the Loss.')
    loss.backward()
    optimizer.step()
    # post_op_process(model)
    
    return loss.item(), loss_b.item()


def trainIter(model, 
              optimizer, 
              trn_param,
              device,
              loss_fn,
              model_dir,
              trn_data, 
              val_data, 
              tst_data,
              batch_size=32,
              eva_steps=100,
              dec_steps=2000,
              tol_steps=2000,
              all_steps=30000,
              lr_dec_rate=0.1):
    
    best_rmse = trn_param['best_rmse']
    best_iter = trn_param['best_iter']
    iteration = trn_param['iteration']
    log = trn_param['log']
    start = time.time()
    
    sum_loss = 0.
    sum_loss_b = 0.
    for it in range(iteration + 1, all_steps + 1):
        batch_hg, dg_node_feat_continuous, dg_node_feat_discrete, lg_node_feat_continuous,\
        lg_node_feat_discrete, dg_edge_feat, lg_edge_feat, co_feat, mask_mat, y = trn_data.next_random_batch(batch_size)
        
        cuda_hg = batch_hg.to(device)
        dg_node_feat_discrete = dg_node_feat_discrete.to(device)
        lg_node_feat_discrete = lg_node_feat_discrete.to(device)
        dg_node_feat_continuous = dg_node_feat_continuous.to(device)
        lg_node_feat_continuous = lg_node_feat_continuous.to(device)
        dg_edge_feat = dg_edge_feat.to(device)
        lg_edge_feat = [feat.to(device) for feat in lg_edge_feat]
        co_feat = co_feat.to(device)
        mask_mat = mask_mat.to(device)
        y = y.to(device)
        
        loss, loss_b = train(cuda_hg, loss_fn,
                     dg_node_feat_continuous,
                     dg_node_feat_discrete,
                     lg_node_feat_continuous,
                     lg_node_feat_discrete,
                     dg_edge_feat,
                     lg_edge_feat, co_feat, mask_mat, y, model, optimizer)
        sum_loss += loss
        sum_loss_b += loss_b
        end = time.time()
        
        if it % eva_steps == 0:
            rmse_val, mae_val, sd_val, r_val = evaluate(model, val_data, device, 128, False)
            rmse_tst, mae_tst, sd_tst, r_tst = evaluate(model, tst_data, device, 128, False)
            end_val = time.time()
            
            print('-----------------------------------------------------------------------')
            print('Steps: %d / %d, loss: %.4f, loss_b: %.4f, time: %.4f, val_time: %.4f.' % 
                  (it, all_steps, sum_loss/(eva_steps*batch_size), sum_loss_b/(eva_steps*batch_size), end-start, end_val-end))
            print('Val - RMSE: %.6f, MAE: %.6f, SD: %.6f, R: %.6f.' % (rmse_val, mae_val, sd_val, r_val))
            print('Test - RMSE: %.6f, MAE: %.6f, SD: %.6f, R: %.6f.' % (rmse_tst, mae_tst, sd_tst, r_tst))
           
            log += '-----------------------------------------------------------------------\n'
            log += 'Steps: %d / %d, loss: %.4f, time: %.4f, val_time: %.4f. \n' % \
                  (it, all_steps, sum_loss/(eva_steps*batch_size), end-start, end_val-end)
            log += 'Val - RMSE: %.6f, MAE: %.6f, SD: %.6f, R: %.6f. \n' % (rmse_val, mae_val, sd_val, r_val)
            log += 'Test - RMSE: %.6f, MAE: %.6f, SD: %.6f, R: %.6f. \n' % (rmse_tst, mae_tst, sd_tst, r_tst)

            if rmse_val < best_rmse:
                best_rmse = rmse_val
                best_iter = it
                torch.save(model, os.path.join(model_dir, 'Best_model.pt'))
                log += 'Best - RMSE: %.6f, MAE: %.6f, SD: %.6f, R: %.6f. \n' % (rmse_tst, mae_tst, sd_tst, r_tst)
            
            start = time.time()
            sum_loss = 0.
            sum_loss_b = 0.
            
        if it % dec_steps == 0:
            optimizer = lr_scheduler(optimizer, lr_dec_rate)
        
        # stop training if the mae does not decrease in tol_steps on validation set
        if it - best_iter > tol_steps:
            break
        
        if it % eva_steps == 0:
            trn_param['iteration'] = it
            trn_param['best_mae'] = best_rmse
            trn_param['best_rmse'] = best_iter
            trn_param['log'] = log
            save_model_state(model, optimizer, trn_param, os.path.join(model_dir, 'checkpoint.tar'))
            
            # write the log
            f = open(os.path.join(model_dir, 'log.txt'), 'w')
            f.write(log)
            f.close()
    
    # write the log
    log += 'The best iter is %d, best val RMSE is %.6f. \n' % (best_iter, best_rmse)
    f = open(os.path.join(model_dir, 'log.txt'), 'w')
    f.write(log)
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/processed/PDBbind/')
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--restore', type=int, default=0)
    parser.add_argument('--model_dir', type=str, default='pretrained_model')
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
    parser.add_argument("--dec_steps", type=int, default=8000)
    parser.add_argument('--tol_steps', type=int, default=20000)
    parser.add_argument('--all_steps', type=int, default=20000)

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
    model_dir = './output/' + args.model_dir
    
    # training
    if args.train:
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        
        # load training set and validation set
        trn_data = DataLoader(os.path.join(args.data_dir, 'train_ca_r%d_n%d.data' % (args.cut_r, args.n_space)))
        tst_data = DataLoader(os.path.join(args.data_dir, 'test_ca_r%d_n%d.data' % (args.cut_r, args.n_space)))
        val_data = DataLoader(os.path.join(args.data_dir, 'val_ca_r%d_n%d.data' % (args.cut_r, args.n_space)))
        
        # calculate mean and std from the training set
        dg_mean, lg_mean, dg_std, lg_std = trn_data.get_statistics()
        print('Per atom mean: %.7f, std: %.7f.' % (dg_mean, dg_std))
        print('Per edge mean: %.7f, std: %.7f.' % (lg_mean, lg_std))
        
        model = SIGN(num_convs=args.num_convs,
                     dense_dims=dense_dims,
                     conv_dim=args.conv_dim,
                     infeat_dim=args.infeat_dim,
                     hidden_dim=args.hidden_dim,
                     dg_node_type_universe=trn_data.dg_node_type_universe,
                     lg_node_type_universe=trn_data.lg_node_type_universe,
                     n_space=args.n_space,
                     cut_r=args.cut_r,
                     model_b2b=args.model_b2b,
                     merge_b2b=args.merge_b2b,
                     merge_b2a=args.merge_b2a,
                     activation=activation,
                     num_heads=args.num_heads,
                     feat_drop=args.feat_drop).to(device)
        
        # initialize the optimizer
        if args.weight_decay > 0:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
        
        # try to load previous training
        if args.restore:
            try:
                trn_param = load_model_state(model, optimizer, os.path.join(model_dir, 'checkpoint.tar'))
            except:
                trn_param = {'iteration':0, 'best_rmse': np.inf, 'best_iter': 1, 'log':str(args)+'\n'}
        else:
            trn_param = {'iteration':0, 'best_rmse': np.inf, 'best_iter': 1, 'log':str(args)+'\n'}
        
        trainIter(model,
                  optimizer,
                  trn_param,
                  device=device,
                  loss_fn=args.loss_fn,
                  model_dir=model_dir,
                  trn_data=trn_data,
                  val_data=val_data,
                  tst_data=tst_data,
                  batch_size=args.batch_size,
                  eva_steps=args.eva_steps,
                  dec_steps=args.dec_steps,
                  tol_steps=args.tol_steps,
                  all_steps=args.all_steps,
                  lr_dec_rate=args.lr_dec_rate)
        
