import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from .utils import get_link_labels


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



@torch.no_grad()
def member_infer_attack(target_model, attack_model, data, logits=None):
    '''Membership inference attack'''

    edge = data.train_pos_edge_index[:, data.df_mask]  # Deleted edges in the training set
    z = target_model(data.x, data.train_pos_edge_index[:, data.dr_mask])
    feature1 = target_model.decode(z, edge).sigmoid()
    feature0 = 1 - feature1
    feature = torch.stack([feature0, feature1], dim=1) # Posterior MI
    # feature = torch.cat([z[edge[0]], z[edge][1]], dim=-1)  # Embedding/Repr. MI
    logits = attack_model(feature)
    _, pred = torch.max(logits, 1)
    suc_rate = 1 - pred.float().mean()


    return torch.softmax(logits, dim=-1).squeeze().tolist(), suc_rate.cpu().item()


@torch.no_grad()
def member_infer_attack_GE(target_model, attack_model, data, feat_train, logits=None):
    '''Membership inference attack'''

    feat_train = feat_train[:feat_train.shape[0]//2]
    feature1 = target_model(feat_train[data.df_mask]).squeeze().sigmoid()
    feature0 = 1 - feature1
    feature = torch.stack([feature0, feature1], dim=1) # Posterior MI
    # feature = torch.cat([z[edge[0]], z[edge][1]], dim=-1) # Embedding/Repr. MI
    logits = attack_model(feature)
    _, pred = torch.max(logits, 1)
    suc_rate = 1 - pred.float().mean()
    return torch.softmax(logits, dim=-1).squeeze().tolist(), suc_rate.cpu().item()



@torch.no_grad()
def get_node_embedding_data(model, data):
    model.eval()
    
    if hasattr(data, 'dtrain_mask') and data.dtrain_mask is not None:
        node_embedding = model(data.x.to(device), data.train_pos_edge_index[:, data.dtrain_mask].to(device))
    else:
        node_embedding = model(data.x.to(device), data.train_pos_edge_index.to(device))

    return node_embedding



