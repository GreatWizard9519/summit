#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 13/2/2023 12:58 pm
# @Author  : Wizard Chenhan Zhang
# @FileName: graph_editor.py
# @Software: PyCharm


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse
from scipy.sparse.csgraph import shortest_path
from torch_geometric.utils import negative_sampling, add_self_loops
from torch.utils.data import DataLoader
from torch_scatter import scatter
from torch_sparse import SparseTensor
from torch_geometric.utils import coalesce, add_self_loops, to_undirected, from_scipy_sparse_matrix
import torch_geometric.transforms as T
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
from torch_geometric.utils import negative_sampling, train_test_split_edges, subgraph, is_undirected
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, mean_squared_error

from .base import Trainer
from .base_backup import KGTrainer, NodeClassificationTrainer
from ..evaluation import *
from ..utils import *


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# graph adj utils
def row_norm(adj):
    if isinstance(adj, torch_sparse.SparseTensor):
        # Add self loop
        adj_t = torch_sparse.fill_diag(adj, 1)
        deg = torch_sparse.sum(adj_t, dim=1)
        deg_inv = 1. / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0.)
        adj_t = torch_sparse.mul(adj_t, deg_inv.view(-1, 1))
        return adj_t

def sym_norm(adj):
    from torch_geometric.nn.conv.gcn_conv import gcn_norm
    if isinstance(adj, torch_sparse.SparseTensor):
        adj_t = gcn_norm(adj, add_self_loops=True)
        return adj_t

def remove_self_loop(adj):
    # this also support adjs of any shapes
    row   = adj.storage.row()
    col   = adj.storage.col()
    value = adj.storage.value()
    size  = adj.sparse_sizes()

    keep_inds = [row != col]

    return torch_sparse.SparseTensor(row   = row[keep_inds],
                                     col   = col[keep_inds],
                                     value = adj.storage.value()[keep_inds],
                                     sparse_sizes=size)

# close form solution utils
def remove_data(X_prime, Y_prime, XtX_inv, W):
    # (X_prime, Y_prime): data to remove;
    # W_orginal: params before remove;

    num_data = X_prime.shape[0]

    A = XtX_inv@X_prime.T
    B = torch.linalg.inv(torch.eye(num_data).to(X_prime) - X_prime@XtX_inv@X_prime.T)
    C = Y_prime - X_prime@W
    D = X_prime@XtX_inv

    return XtX_inv + A@B@D, W - A@B@C

def add_data(X_prime, Y_prime, XtX_inv, W):
    # (X_prime, Y_prime): data to add;
    # W_orginal: params before add data;

    num_data = X_prime.shape[0]

    A = XtX_inv@X_prime.T
    B = torch.linalg.inv(torch.eye(num_data).to(X_prime) + X_prime@XtX_inv@X_prime.T)
    C = Y_prime - X_prime@W
    D = X_prime@XtX_inv

    return XtX_inv - A@B@D, W + A@B@C

def find_w(X, Y, lam=0):
    try:
        Xtx_inv = torch.linalg.inv(X.T@X + lam*torch.eye(X.size(1)))
        Xty = X.T@Y
        W = Xtx_inv@Xty
    except:
        try:
            print('Feat matrix is not inversible, add random noise')
            X = X + torch.randn_like(X)*1e-5
            Xtx_inv = torch.linalg.inv(X.T@X + lam*torch.eye(X.size(1)))
            Xty = X.T@Y
            W = Xtx_inv@Xty
        except:
            print('Feat matrix is not inversible, use psudo inverse')
            Xtx_inv = torch.linalg.pinv(X.T@X + lam*torch.eye(X.size(1)))
            Xty = X.T@Y
            W = Xtx_inv@Xty
    return Xtx_inv, W

def predict(X, W, activation='None'):
    return X@W

# cross and smoothing utils
def pred_test(out, data, split_idx, evaluator):
    pred = out.argmax(dim=-1, keepdim=True)
    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': pred[split_idx['train']]
    })['acc']
    val_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': pred[split_idx['valid']]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': pred[split_idx['test']]
    })['acc']
    return train_acc, val_acc, test_acc


def get_pos_neg_edges(split, split_edge, edge_index, num_nodes, percent=100):

    pos_edge = split_edge[split]['edge'].t()
    if split == 'train':
        new_edge_index, _ = add_self_loops(edge_index)
        neg_edge = negative_sampling(
            new_edge_index, num_nodes=num_nodes,
            num_neg_samples=pos_edge.size(1))
    else:
        neg_edge = split_edge[split]['edge_neg'].t()

    # subsample for pos_edge
    np.random.seed(123)
    num_pos = pos_edge.size(1)
    perm = np.random.permutation(num_pos)
    perm = perm[:int(percent / 100 * num_pos)]
    pos_edge = pos_edge[:, perm]
    # subsample for neg_edge
    np.random.seed(123)
    num_neg = neg_edge.size(1)
    perm = np.random.permutation(num_neg)
    perm = perm[:int(percent / 100 * num_neg)]
    neg_edge = neg_edge[:, perm]

    return pos_edge, neg_edge

def neighbors(fringe, A):
    return set(A[list(fringe)].indices)

def k_hop_subgraph(src, dst, num_hops, A, sample_ratio=1.0, max_nodes_per_hop=None):
    # Extract the k-hop enclosing subgraph around link (src, dst) from A.
    nodes = [src, dst]
    dists = [0, 0]
    visited = set([src, dst])
    fringe = set([src, dst])
    for dist in range(1, num_hops+1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if sample_ratio < 1.0:
            fringe = random.sample(fringe, int(sample_ratio*len(fringe)))
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes + list(fringe)
        dists = dists + [dist] * len(fringe)
    subgraph = A[nodes, :][:, nodes]

    # Remove target link between the subgraph.
    subgraph[0, 1] = 0
    subgraph[1, 0] = 0

    return nodes, subgraph, dists

def drnl_node_labeling(adj, src=0, dst=1):
    # Double Radius Node Labeling (DRNL).
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst-1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    z[torch.isnan(z)] = 0.

    return z.to(torch.long)


def get_all_edges(data):
    # train
    neg_edge = negative_sampling(data.train_pos_edge_index, data.num_nodes)

    train_links = torch.cat([data.train_pos_edge_index, neg_edge], 1).t().tolist()
    train_labels = [1] * data.train_pos_edge_index.size(1) + [0] * neg_edge.size(1)
    train_length = len(train_labels)

    # valid
    valid_links = torch.cat([data.val_pos_edge_index, data.val_neg_edge_index], 1).t().tolist()
    valid_labels = [1] * data.val_pos_edge_index.size(1) + [0] * data.val_neg_edge_index.size(1)
    valid_length = len(valid_labels)

    # test
    test_links = torch.cat([data.test_pos_edge_index, data.test_neg_edge_index], 1).t().tolist()
    test_labels = [1] * data.test_pos_edge_index.size(1) + [0] * data.test_neg_edge_index.size(1)
    test_length = len(test_labels)

    all_links  = train_links + valid_links + test_links
    all_labels = train_labels + valid_labels + test_labels

    return all_links, all_labels, train_length, valid_length, test_length

def get_1_hop_neighbors(num_nodes, adj_t_scipy, sample_size=-1):
    neighbor_nodes = []

    # pbar = tqdm(total=num_nodes)
    # pbar.set_description('Compute neighbors')

    for node_i in range(num_nodes):
        one_hop_neighbors = np.setdiff1d(adj_t_scipy[node_i].indices, node_i)
        if sample_size > 0 and len(one_hop_neighbors) > sample_size:
            one_hop_neighbors_weight = adj_t_scipy[node_i, one_hop_neighbors].data
            one_hop_neighbors = np.random.choice(one_hop_neighbors, p=one_hop_neighbors_weight/np.sum(one_hop_neighbors_weight), size=sample_size, replace=False)

        neighbors = np.concatenate([
            np.array([node_i]), one_hop_neighbors
        ])
        neighbor_nodes.append(neighbors)

        # pbar.update(1)

    subgraph_relation = [[] for _ in range(num_nodes)]
    for neighbors_ in neighbor_nodes:
        node_i, neighbors_ = neighbors_[0], neighbors_[1:]
        for node_j in neighbors_:
            subgraph_relation[node_j].append(node_i)

    return neighbor_nodes, subgraph_relation

def compute_edge_feats(data, adj, all_links, all_subgraph_nodes):
    # pbar = tqdm(total=len(all_links))
    # pbar.set_description('Compute edge feats')

    edge_feats_all = np.zeros((len(all_links), data.x.size(1)))

    for cnt, (n_i, n_j) in enumerate(all_links):

        intersect = all_subgraph_nodes[cnt]

        adj_intersect = adj[intersect, :][:, intersect]
        x_intersect = data.x[intersect].numpy()
        node_feat = adj_intersect@x_intersect
        edge_feats_all[cnt, :] = node_feat[0] * node_feat[1]

    #     pbar.update(1)

    # pbar.close()
    return torch.from_numpy(edge_feats_all)

def compute_edge_feats_and_subgraphs(data, neighbor_nodes, adj, all_links):

    # pbar = tqdm(total=len(all_links))
    # pbar.set_description('Compute edge feats')

    all_subgraph_nodes = []
    edge_feats_all = np.zeros((len(all_links), data.x.size(1)))

    for cnt, (n_i, n_j) in enumerate(all_links):

        intersect = np.intersect1d(neighbor_nodes[n_i], neighbor_nodes[n_j])
        intersect = np.setdiff1d(intersect, np.array([n_i, n_j]))
        intersect = np.concatenate([np.array([n_i, n_j]), intersect])
        all_subgraph_nodes.append(intersect)

        adj_intersect = adj[intersect, :][:, intersect]
        x_intersect = data.x[intersect].numpy()
        node_feat = adj_intersect@x_intersect
        edge_feats_all[cnt, :] = node_feat[0] * node_feat[1]

    #     pbar.update(1)

    # pbar.close()
    return edge_feats_all, all_subgraph_nodes

def compute_CN_score(A, edge_index, batch_size=100000):
    if edge_index.size(1) == 2:
        edge_index = edge_index.t()

    # The Common Neighbor heuristic score.
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)

    scores = []
    for ind in link_loader:
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten()
        scores.append(cur_scores)

    return torch.from_numpy(np.concatenate(scores, 0))


def get_node_subgraph_relation(data, all_subgraph_nodes):
    subgraph_relation = [[] for _ in range(data.num_nodes)]

    for sg_ind, all_subgraph_nodes_ in enumerate(all_subgraph_nodes):
        for node_i in all_subgraph_nodes_:
            subgraph_relation[node_i].append(sg_ind)

    return subgraph_relation


def get_affected_subgraphs(src, dst, all_subgraph_nodes, subgraph_relation, adj):

    src_subgraphs = subgraph_relation[src]
    dst_subgraphs = subgraph_relation[dst]

    overlap_subgraphs = np.intersect1d(np.array(src_subgraphs), np.array(dst_subgraphs))

    return_subgraph_ind = []
    return_subgraph_ind_affected = []
    for subgraph_ind in overlap_subgraphs:

        # check if edge exist
        nodes = np.array(all_subgraph_nodes[subgraph_ind])
        subgraph = adj[nodes, :][:, nodes]

        src_local = np.where(nodes==src)[0]
        dst_local = np.where(nodes==dst)[0]

        if dst_local in subgraph[src_local].indices:
            return_subgraph_ind_affected.append(subgraph_ind)
            if src_local <= 1 and dst_local<= 1:
                return_subgraph_ind.append(subgraph_ind)

    return return_subgraph_ind, return_subgraph_ind_affected


def graph_editor(
        delete_link_ind, X_tx_inv_optim, W_optim, train_inds, data, adj, all_links, all_subgraph_nodes, subgraph_relation, all_edge_feat_with_aug, all_edge_labels):

    ###############################
    # Step 0: find all affect nodes + compute new feats

    delete_subgraphs = []
    affect_subgraphs_all = []
    for link_ind in delete_link_ind:

        node_i, node_j = all_links[link_ind][0], all_links[link_ind][1]
        _delete_sg, _affect_sg = get_affected_subgraphs(node_i, node_j, all_subgraph_nodes, subgraph_relation, adj)
        delete_subgraphs.extend(_delete_sg)
        affect_subgraphs_all.extend(_affect_sg)

    # print('>>> 1', delete_subgraphs, affect_subgraphs_all)
    delete_subgraphs = list(np.intersect1d(np.array(delete_subgraphs), train_inds))

    affect_subgraphs_eval = list(np.setdiff1d(np.array(affect_subgraphs_all), train_inds))
    affect_subgraphs = list(np.intersect1d(np.array(affect_subgraphs_all), train_inds))

    affect_links = [all_links[link_ind] for link_ind in affect_subgraphs]
    affect_edge_feats = torch.cat([
        compute_edge_feats(data, adj, affect_links, all_subgraph_nodes),
        compute_CN_score(adj, torch.tensor(affect_links)).unsqueeze(-1)], dim=1).to(all_edge_feat_with_aug)

    affect_links_eval = [all_links[link_ind] for link_ind in affect_subgraphs_eval]
    if len(affect_links_eval) > 0:
        affect_edge_feats_eval = torch.cat([
            compute_edge_feats(data, adj, affect_links_eval, all_subgraph_nodes),
            compute_CN_score(adj, torch.tensor(affect_links_eval)).unsqueeze(-1)], dim=1).to(all_edge_feat_with_aug)

    ###############################
    # Step 1: remove edge feats

    X_tx_inv_prime, W_prime = remove_data(X_prime=all_edge_feat_with_aug[delete_subgraphs+affect_subgraphs],
                                          Y_prime=all_edge_labels[delete_subgraphs+affect_subgraphs],
                                          XtX_inv=X_tx_inv_optim,
                                          W      =W_optim)

    ###############################
    # Step 2: Update edge feats

    all_edge_feat_with_aug[affect_subgraphs] = affect_edge_feats
    if len(affect_links_eval) > 0:
        all_edge_feat_with_aug[affect_subgraphs_eval] = affect_edge_feats_eval

    all_edge_feat_with_aug[delete_subgraphs] = -1
    all_edge_labels[delete_subgraphs] = -1

    X_tx_inv_prime, W_prime = add_data(X_prime=affect_edge_feats,
                                       Y_prime=all_edge_labels[affect_subgraphs],
                                       XtX_inv=X_tx_inv_prime,
                                       W      =W_prime)

    return X_tx_inv_prime, W_prime


def fine_tune(W, epochs, X, Y, args):
    model = GNN(W.clone()).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    X = X.to(args.device)
    Y = Y.to(args.device).float()

    for epoch in trange(1, 1 + epochs, desc='Finetune'):
        loss = F.binary_cross_entropy_with_logits(model(X), Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # pos_valid_pred = model(valid_edge_feats[:num_valid_pos_edge].to(args.device)).cpu()
        # neg_valid_pred = model(valid_edge_feats[num_valid_pos_edge:].to(args.device)).cpu()

    return model.W.data.cpu().numpy()

class GNN(nn.Module):
    def __init__(self, W):
        super(GNN, self).__init__()
        self.W = torch.nn.Parameter(W)

    def forward(self, X, edge=None):
        return X@self.W

    def decode(self, z, pos_edge_index, neg_edge_index=None):
        if neg_edge_index is not None:
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

        else:
            edge_index = pos_edge_index
            logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

        return logits


class GraphEditorTrainer(Trainer):
    def train(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):

        start_time = time.time()
        #------
        test_time_start = time.time()
        #------

        adj = sp.csr_matrix((torch.ones(data.edge_index.shape[1]), (data.edge_index[0], data.edge_index[1])), shape=(data.num_nodes, data.num_nodes))

        all_links, all_labels, train_length, valid_length, test_length = get_all_edges(data)
        all_links_tensor = torch.tensor(all_links)
        all_edge_labels  = torch.tensor(all_labels).unsqueeze(-1)

        # compute 1-hop neighbors
        neighbor_nodes, subgraph_relation = get_1_hop_neighbors(data.num_nodes, adj, args.hop_neighbors)
        del subgraph_relation


        # compute node feats
        edge_feats_all, all_subgraph_nodes = compute_edge_feats_and_subgraphs(data, neighbor_nodes, adj, all_links)
        edge_feats_all = torch.from_numpy(edge_feats_all)

        # compute node feats
        all_cn_score = torch.tensor(compute_CN_score(adj, all_links_tensor))

        # Prepare features
        all_edge_feat_with_aug = torch.cat([edge_feats_all, all_cn_score.unsqueeze(-1)], dim=1).float()


        # pre-train
        train_edge_feats, valid_edge_feats, test_edge_feats = torch.split(
            all_edge_feat_with_aug, [train_length, valid_length, test_length], dim=0)
        train_edge_labels, valid_edge_labels, test_edge_labels = torch.split(
            all_edge_labels, [train_length, valid_length, test_length], dim=0)

        pretrain_start = time.time()
        if os.path.exists(os.path.join(self.args.checkpoint_dir, 'closed_form_solution.pt')):
            ckpt = torch.load(os.path.join(self.args.checkpoint_dir, 'closed_form_solution.pt'))
            X_tx_inv_optim = ckpt['x']
            W_optim = ckpt['w']
        else:
            X_tx_inv_optim, W_optim = find_w(train_edge_feats, 2*(train_edge_labels-0.5), args.lam)
            torch.save(
                {'x': X_tx_inv_optim, 'w': W_optim},
                os.path.join(self.args.checkpoint_dir, 'closed_form_solution.pt'))
        self.trainer_log['pretrain_time'] = time.time() - pretrain_start

        model = GNN(W_optim.clone()).to(self.args.device)

        subgraph_relation = get_node_subgraph_relation(data, all_subgraph_nodes)

        train_inds = np.where(np.array(all_labels[:train_length])==1)[0]
        delete_links_all = np.random.choice(train_inds, args.num_remove_links)
        batch = self.args.parallel_unlearning
        delete_link_batch = [[] for _ in range(batch)]
        for i, link_ind in enumerate(delete_links_all):
            delete_link_batch[i % batch].append(link_ind)


        train_inds = np.arange(train_length)
        start_time = time.time()
        for delete_links in tqdm(delete_link_batch, desc='Remove'):

            # print('>>> GraphEditor: delete edges {} from graph'.format(delete_links))

            # grapheditor
            delete_links = list(delete_links)
            X_tx_inv_optim, W_optim = graph_editor(
                delete_links, X_tx_inv_optim, W_optim, train_inds, data, adj, all_links,
                all_subgraph_nodes, subgraph_relation, all_edge_feat_with_aug, all_edge_labels)

            # remove edges from adj
            for link_ind in delete_links:
                adj[all_links[link_ind][0]].data[all_links[link_ind][1] == adj[all_links[link_ind][0]].indices] = 0
            adj.eliminate_zeros()

        finetune_start = time.time()
        W_finetune = fine_tune(W_optim, 500, train_edge_feats, train_edge_labels, args)
        end_time = time.time()
        self.trainer_log['finetune_time'] = end_time - finetune_start
        self.trainer_log['training_time'] = end_time - start_time
        #------
        test_time_end = time.time()
        test_time = test_time_end - test_time_start
        print(test_time)
        #------
        torch.save({'model_state': W_finetune}, os.path.join(self.args.checkpoint_dir, 'model_best.pt'))
        torch.save({'train': train_edge_feats, 'valid': valid_edge_feats, 'test': test_edge_feats}, os.path.join(self.args.checkpoint_dir, 'features.pt'))

    @torch.no_grad()
    def eval(self, model, data, stage='val', pred_all=False):
        model.eval()
        pos_edge_index = data[f'{stage}_pos_edge_index']
        neg_edge_index = data[f'{stage}_neg_edge_index']

        if self.args.eval_on_cpu:
            model = model.to('cpu')

        feats = torch.load(os.path.join(self.args.checkpoint_dir, 'features.pt'))
        logits = model(feats['test']).squeeze()
        label = self.get_link_labels(pos_edge_index, neg_edge_index)

        # DT AUC AUP
        loss = F.binary_cross_entropy_with_logits(logits, label).cpu().item()
        dt_auc = roc_auc_score(label.cpu(), logits.cpu())
        dt_aup = average_precision_score(label.cpu(), logits.cpu())

        # DF AUC AUP
        feat_train = feats['train']
        feat_train = feat_train[:feat_train.shape[0]//2]
        if self.args.unlearning_model in ['original']:
            df_logit = []
        else:
            # df_logit = model.decode(z, data.train_pos_edge_index[:, data.df_mask]).sigmoid().tolist()
            df_logit = model(feat_train[data.df_mask]).squeeze().sigmoid().tolist()

        all_pos_logit = model(feat_train[data.dr_mask]).squeeze().sigmoid()
        shape = all_pos_logit.shape[0]

        if len(df_logit) > 0:
            df_auc = []
            df_aup = []

            # Sample pos samples
            if len(self.df_pos_edge) == 0:
                for i in range(500):
                    mask = torch.zeros(shape, dtype=torch.bool)
                    idx = torch.randperm(shape)[:len(df_logit)]
                    mask[idx] = True
                    self.df_pos_edge.append(mask)

            # Use cached pos samples
            for mask in self.df_pos_edge:
                pos_logit = all_pos_logit[mask].tolist()  # all_pos_logit tensor([0.5096, 0.7023, 0.7761,  ..., 0.5576, 0.5012, 0.5263])

                logit = df_logit + pos_logit
                label = [0] * len(df_logit) +  [1] * len(df_logit)
                df_auc.append(roc_auc_score(label, logit))
                df_aup.append(average_precision_score(label, logit))

            df_auc = np.mean(df_auc)
            df_aup = np.mean(df_aup)

        else:
            df_auc = np.nan
            df_aup = np.nan

        # DF Consistency ((deleted dataset))
        if len(df_logit) > 0:
            df_con_auc = []
            df_con_mse = []

            # Sample pos samples 采样500次求平均
            if len(self.df_neg_edge) == 0:
                for i in range(500):
                    mask = torch.zeros(neg_edge_index.shape[1], dtype=torch.bool) # tensor([False, False, False,  ..., False, False, False]) |E|
                    idx = torch.randperm(neg_edge_index.shape[1])[:len(df_logit)] # 在remain的里面抽样 df个样本
                    mask[idx] = True
                    self.df_neg_edge.append(mask)

            # Use cached pos samples
            neg_mask = torch.zeros(data.df_mask.shape[0], dtype=torch.bool)
            neg_mask[neg_edge_index] = True
            all_neg_logit = model(feat_train[neg_mask]).sigmoid().squeeze()

            neg_logit = all_neg_logit.tolist()
            df_con_mse.append(mean_squared_error(df_logit[:len(neg_logit)], neg_logit))
            logit = df_logit[:len(neg_logit)] + neg_logit  # 285 + 285
            label = [0] * len(neg_logit) +  [1] * len(neg_logit)
            df_con_auc = np.mean(roc_auc_score(label, logit))

            df_con_auc = np.mean(df_con_auc)
            df_con_mse = np.mean(df_con_mse)
        else:
            df_con_auc = np.nan
            df_con_mse = np.nan

        # Logits for all node pairs
        logit_all_pair = None

        log = {
            f'{stage}_loss': loss,
            f'{stage}_dt_auc': dt_auc,
            f'{stage}_dt_aup': dt_aup,
            f'{stage}_df_auc': df_auc,
            f'{stage}_df_aup': df_aup,
            f'{stage}_df_con_auc': df_con_auc,
            f'{stage}_df_con_mse': df_con_mse,
            f'{stage}_df_logit_mean': np.mean(df_logit) if len(df_logit) > 0 else np.nan,
            f'{stage}_df_logit_std': np.std(df_logit) if len(df_logit) > 0 else np.nan
        }

        if self.args.eval_on_cpu:
            model = model.to(self.args.device)

        return loss, dt_auc, dt_aup, df_auc, df_aup, df_con_auc, df_con_mse, df_logit, logit_all_pair, log

    @torch.no_grad()
    def test(self, model, data, model_retrain=None, attack_model_all=None, attack_model_sub=None, ckpt='best'):

        W = torch.load(os.path.join(self.args.checkpoint_dir, 'model_best.pt'))['model_state']
        model = GNN(torch.tensor(W)).to('cpu')

        if 'ogbl' in self.args.dataset:
            pred_all = False
        else:
            pred_all = True
        loss, dt_auc, dt_aup, df_auc, df_aup, df_con_auc, df_con_mse, df_logit, logit_all_pair, test_log = self.eval(model, data, 'test', pred_all)

        self.trainer_log['dt_loss'] = loss
        self.trainer_log['dt_auc'] = dt_auc
        self.trainer_log['dt_aup'] = dt_aup
        self.trainer_log['df_logit'] = df_logit
        self.logit_all_pair = logit_all_pair
        self.trainer_log['df_auc'] = df_auc
        self.trainer_log['df_aup'] = df_aup
        self.trainer_log['df_con_auc'] = df_con_auc
        self.trainer_log['df_con_auc'] = df_con_mse

        if model_retrain is not None:    # Deletion
            self.trainer_log['ve'] = verification_error(model, model_retrain).cpu().item()
            # self.trainer_log['dr_kld'] = output_kldiv(model, model_retrain, data=data).cpu().item()
        feats = torch.load(os.path.join(self.args.checkpoint_dir, 'features.pt'))
        feat_train = feats['train'].to(self.args.device)
        model.to(self.args.device)

        # MI Attack after unlearning
        if attack_model_all is not None:
            mi_logit_all_after, mi_sucrate_all_after = member_infer_attack_GE(model, attack_model_all, data, feat_train)
            self.trainer_log['mi_logit_all_before'] = 0
            self.trainer_log['mi_sucrate_all_before'] = 0
            self.trainer_log['mi_logit_all_after'] = mi_logit_all_after
            self.trainer_log['mi_sucrate_all_after'] = mi_sucrate_all_after
        if attack_model_sub is not None:
            mi_logit_sub_after, mi_sucrate_sub_after = member_infer_attack_GE(model, attack_model_sub, data, feat_train)
            self.trainer_log['mi_logit_sub_after'] = mi_logit_sub_after
            self.trainer_log['mi_sucrate_sub_after'] = mi_sucrate_sub_after

            self.trainer_log['mi_ratio_all'] = np.mean([i[1] / j[1] for i, j in zip(self.trainer_log['mi_logit_all_after'], self.trainer_log['mi_logit_all_before'])])
            self.trainer_log['mi_ratio_sub'] = np.mean([i[1] / j[1] for i, j in zip(self.trainer_log['mi_logit_sub_after'], self.trainer_log['mi_logit_sub_before'])])
            print(self.trainer_log['mi_ratio_all'], self.trainer_log['mi_ratio_sub'], self.trainer_log['mi_sucrate_all_after'], self.trainer_log['mi_sucrate_sub_after'])
            print(self.trainer_log['df_auc'], self.trainer_log['df_aup'])

        return loss, dt_auc, dt_aup, df_auc, df_aup, df_con_auc, df_con_mse, df_logit, logit_all_pair, test_log

