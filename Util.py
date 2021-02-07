import torch
import torch.nn.functional as F
import numpy as np
from math import sqrt
from sklearn.linear_model import LinearRegression

def save_model_state(model, optimizer, trn_param, filename):
    torch.save({
            'trn_param': trn_param,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, filename)
    
def load_model_state(model, optimizer, filename):
    checkpoint = torch.load(filename)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['trn_param']

def lr_scheduler(optimizer, decrease_rate=0.9):
    """Decay learning rate by a factor."""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decrease_rate

    return optimizer

def post_op_process(model):
    model.dg_input_module.edge_rbf.mu.data.clamp_(0)
    model.dg_input_module.edge_rbf.beta.data.clamp_(0)
    model.lg_input_module.node_rbf.mu.data.clamp_(0)
    model.lg_input_module.node_rbf.beta.data.clamp_(0)
    model.lg_input_module.edge_rbf.mu.data.clamp_(0)
    model.lg_input_module.edge_rbf.beta.data.clamp_(0)
    model.lg_output_module.std.data.clamp_(0)
    model.dg_output_module.std.data.clamp_(0)

    
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
            
            # get the attention score for each body
            if return_attn:
                pass

        if return_attn:
            pass
        
        y_hat = np.array(y_hat_list).reshape(-1,)
        y = np.array(y_list).reshape(-1,)
        # print(y_hat.shape, y.shape)
        return rmse(y, y_hat), mae(y, y_hat), sd(y, y_hat), pearson(y, y_hat)