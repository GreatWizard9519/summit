
import os
import copy
import json
import wandb
import pickle
import argparse
import torch
import torch.nn as nn
from torch_geometric.utils import to_undirected, to_networkx, k_hop_subgraph, is_undirected
from torch_geometric.data import Data
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.seed import seed_everything
import traceback
import time

from sklearn.model_selection import train_test_split

from framework import get_model, get_trainer, get_attacker, get_shadow_model
from framework.training_args import parse_args
from framework.utils import *
from train_mi import MLPAttacker

from email_me import send_email


args = parse_args()
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def load_args(path):
    with open(path, 'r') as f:
        d = json.load(f)
    parser = argparse.ArgumentParser()
    for k, v in d.items():
        parser.add_argument('--' + k, default=v)
    try:
        parser.add_argument('--df_size', default=0.5)
    except:
        pass
    args = parser.parse_args()

    for k, v in d.items():
        setattr(args, k, v)

    return args





    return output

torch.autograd.set_detect_anomaly(True)
def main():
    args = parse_args()
    args.df = 'in'
    #
    # args.epochs = 2
    # args.valid_freq = 1
    original_path = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, 'original', str(args.random_seed))
    attack_path_all = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, 'member_infer_all', str(args.random_seed))
    args.attack_dir = attack_path_all
    if not os.path.exists(attack_path_all):
        os.makedirs(attack_path_all)
    shadow_path_all = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, 'shadow_all', str(args.random_seed))
    args.shadow_dir = shadow_path_all
    if not os.path.exists(shadow_path_all):
        os.makedirs(shadow_path_all)
    attack_path_sub = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, 'member_infer_sub', str(args.random_seed))
    seed_everything(args.random_seed)

    if 'gnndelete' in args.unlearning_model:
        args.checkpoint_dir = os.path.join(
            args.checkpoint_dir, args.dataset, args.gnn, args.unlearning_model, 
            '-'.join([str(i) for i in [args.loss_fct, args.loss_type, args.alpha, args.neg_sample_random]]),
            '-'.join([str(i) for i in [args.df, args.df_size, args.random_seed]]))
    else:
        args.checkpoint_dir = os.path.join(
            args.checkpoint_dir, args.dataset, args.gnn, args.unlearning_model, 
            '-'.join([str(i) for i in [args.df, args.df_size, args.random_seed]]))
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Dataset
    with open(os.path.join(args.data_dir, args.dataset, f'd_{args.random_seed}.pkl'), 'rb') as f:
        dataset, data = pickle.load(f)
    print('Directed dataset:', dataset, data)
    if args.gnn not in ['rgcn', 'rgat']:
        args.in_dim = dataset.num_features

    print('Training args', args)
    # wandb.init(config=args)

    # Df and Dr
    assert args.df != 'none'

    if args.df_size >= 100:     # df_size is number of nodes/edges to be deleted
        df_size = int(args.df_size)
    else:                       # df_size is the ratio
        df_size = int(args.df_size / 100 * data.train_pos_edge_index.shape[1])
    print(f'Original size: {data.train_pos_edge_index.shape[1]:,}')
    print(f'Df size: {df_size:,}')

    df_mask_all = torch.load(os.path.join(args.data_dir, args.dataset, f'df_{args.random_seed}.pt'))[args.df]
    df_nonzero = df_mask_all.nonzero().squeeze()

    idx = torch.randperm(df_nonzero.shape[0])[:df_size]
    df_global_idx = df_nonzero[idx]

    print('Deleting the following edges:', df_global_idx)

    # df_idx = [int(i) for i in args.df_idx.split(',')]
    # df_idx_global = df_mask.nonzero()[df_idx]
    
    dr_mask = torch.ones(data.train_pos_edge_index.shape[1], dtype=torch.bool)
    dr_mask[df_global_idx] = False

    df_mask = torch.zeros(data.train_pos_edge_index.shape[1], dtype=torch.bool)
    df_mask[df_global_idx] = True

    # For testing
    data.directed_df_edge_index = data.train_pos_edge_index[:, df_mask]
    if args.gnn in ['rgcn', 'rgat']:
        data.directed_df_edge_type = data.train_edge_type[df_mask]
        



    # Edges in S_Df
    _, two_hop_edge, _, two_hop_mask = k_hop_subgraph(
        data.train_pos_edge_index[:, df_mask].flatten().unique(), 
        2, 
        data.train_pos_edge_index,
        num_nodes=data.num_nodes)
    data.sdf_mask = two_hop_mask

    # Nodes in S_Df
    _, one_hop_edge, _, one_hop_mask = k_hop_subgraph(
        data.train_pos_edge_index[:, df_mask].flatten().unique(), 
        1, 
        data.train_pos_edge_index,
        num_nodes=data.num_nodes)
    sdf_node_1hop = torch.zeros(data.num_nodes, dtype=torch.bool)
    sdf_node_2hop = torch.zeros(data.num_nodes, dtype=torch.bool)

    sdf_node_1hop[one_hop_edge.flatten().unique()] = True
    sdf_node_2hop[two_hop_edge.flatten().unique()] = True

    assert sdf_node_1hop.sum() == len(one_hop_edge.flatten().unique())
    assert sdf_node_2hop.sum() == len(two_hop_edge.flatten().unique())

    data.sdf_node_1hop_mask = sdf_node_1hop
    data.sdf_node_2hop_mask = sdf_node_2hop



    
    train_pos_edge_index, [df_mask, two_hop_mask] = to_undirected(data.train_pos_edge_index, [df_mask.int(), two_hop_mask.int()])
    two_hop_mask = two_hop_mask.bool()
    df_mask = df_mask.bool()
    dr_mask = ~df_mask
    
    data.train_pos_edge_index = train_pos_edge_index
    data.edge_index = train_pos_edge_index
    assert is_undirected(data.train_pos_edge_index)


    print('Undirected dataset:', data)

    data.sdf_mask = two_hop_mask
    data.df_mask = df_mask  # 
    data.dr_mask = dr_mask  # 

    # Model
    model = get_model(args, sdf_node_1hop, sdf_node_2hop, num_nodes=data.num_nodes, num_edge_type=args.num_edge_type)
    if not args.unlearning_model == 'gnndelete':
        shadow_model = get_model(args, sdf_node_1hop, sdf_node_2hop, num_nodes=data.num_nodes, num_edge_type=args.num_edge_type)
    else:
        shadow_model = get_shadow_model(args, sdf_node_1hop, sdf_node_2hop, num_nodes=data.num_nodes, num_edge_type=args.num_edge_type)



    if args.unlearning_model != 'retrain':  # Start from trained GNN model
        if os.path.exists(os.path.join(original_path, 'pred_proba.pt')):
            logits_ori = torch.load(os.path.join(original_path, 'pred_proba.pt'))
            if logits_ori is not None:
                logits_ori = logits_ori.to(device)
        else:
            logits_ori = None

        model_ckpt = torch.load(os.path.join(original_path, 'model_best.pt'), map_location=device)
        model.load_state_dict(model_ckpt['model_state'], strict=False)
   
    else:       # Initialize a new GNN model
        retrain = None
        logits_ori = None

    model = model.to(device)



    if 'gnndelete' in args.unlearning_model:
        parameters_to_optimize = [
            {'params': [p for n, p in model.named_parameters() if 'del' in n], 'weight_decay': 0.0}
        ]
        print('parameters_to_optimize', [n for n, p in model.named_parameters() if 'del' in n])
    
    else:
        parameters_to_optimize = [
            {'params': [p for n, p in model.named_parameters()], 'weight_decay': 0.0}
        ]
        print('parameters_to_optimize', [n for n, p in model.named_parameters()])
    
    optimizer = torch.optim.Adam(parameters_to_optimize, lr=args.lr)#, weight_decay=args.weight_decay)

    if args.attack:
        # MI attack model
        attack_trainer = get_attacker(args)
        # Leak mode
        leak = 'posterior' # repr or posterior
        if not args.unlearning_model == 'gnndelete':  #

            if os.path.exists(os.path.join(attack_path_all, f'{leak}_attack_model_best.pt')):
                attack_model_all = MLPAttacker(args, leak=leak)
                attack_ckpt = torch.load(os.path.join(attack_path_all, f'{leak}_attack_model_best.pt'))
            else:
                # Train Shadow model
                if not os.path.exists(os.path.join(shadow_path_all, 'shadow_model_best.pt')):
                    print('train shadow model')

                    shadow_optimizer = torch.optim.Adam(shadow_model.parameters(), lr=args.lr)
                    attack_trainer.train_shadow(shadow_model, data, shadow_optimizer, args)  # Train Shadow model
                    shadow_ckpt = torch.load(os.path.join(shadow_path_all, 'shadow_model_best.pt'))
                    shadow_model.load_state_dict(shadow_ckpt['model_state'])
                else:
                    print('load shadow model')
                    shadow_ckpt = torch.load(os.path.join(shadow_path_all, 'shadow_model_best.pt'))
                    shadow_model.load_state_dict(shadow_ckpt['model_state'])
                shadow_model = shadow_model.to(args.device)
                data = data.to(args.device)


                # Prerpare Attack Data
                attack_X, attack_Y = attack_trainer.prepare_attack_training_data(shadow_model, data, leak=leak)
                attack_train_data_X, attack_test_data_X, attack_train_data_y, attack_test_data_y = train_test_split(attack_X.numpy(), attack_Y.numpy(),
                                                                                                                    test_size=0.2,
                                                                                                                    stratify=attack_Y)  

                attack_train_data = torch.utils.data.TensorDataset(torch.from_numpy(attack_train_data_X).float(), torch.from_numpy(
                    attack_train_data_y))
                attack_train_loader = torch.utils.data.DataLoader(attack_train_data, batch_size=1024, shuffle=True)

                attack_test_data = torch.utils.data.TensorDataset(torch.from_numpy(attack_test_data_X).float(), torch.from_numpy(
                    attack_test_data_y))  # convert to float to fix  uint8_t overflow error
                attack_test_loader = torch.utils.data.DataLoader(attack_test_data, batch_size=1024, shuffle=True)

                # Train Attack model
                attack_model_all = MLPAttacker(args, leak=leak)
                attack_optimizer = torch.optim.Adam(attack_model_all.parameters(), lr=0.001)
                attack_trainer.train_attack(attack_model_all, attack_train_loader, attack_test_loader, attack_optimizer, leak, args) # Train
                attack_ckpt = torch.load(os.path.join(attack_path_all, f'{leak}_attack_model_best.pt'))

            attack_model_all.load_state_dict(attack_ckpt['model_state'])
            attack_model_all = attack_model_all.to(device)
        else:  # For GNNDelete
            # Leak mode
            if os.path.exists(os.path.join(attack_path_all, f'{leak}_gd_attack_model_best.pt')):
                attack_model_all = MLPAttacker(args, leak=leak)
                attack_ckpt = torch.load(os.path.join(attack_path_all, f'{leak}_gd_attack_model_best.pt'))
            else:
                # Train Shadow model
                if not os.path.exists(os.path.join(shadow_path_all, 'shadow_gd_model_best.pt')):
                    print('train shadow model')
                    parameters_to_optimize_gd = [
                        {'params': [p for n, p in shadow_model.named_parameters() if not 'del' in n], 'weight_decay': 0.0}
                    ]
                    shadow_optimizer = torch.optim.Adam(parameters_to_optimize_gd, lr=args.lr)
                    attack_trainer.train_shadow(shadow_model, data, shadow_optimizer, args)  # Train Shadow model
                    shadow_ckpt = torch.load(os.path.join(shadow_path_all, 'shadow_gd_model_best.pt'))
                    shadow_model.load_state_dict(shadow_ckpt['model_state'])
                else:
                    print('load shadow model')
                    shadow_ckpt = torch.load(os.path.join(shadow_path_all, 'shadow_gd_model_best.pt'))
                    shadow_model.load_state_dict(shadow_ckpt['model_state'])
                shadow_model = shadow_model.to(args.device)
                data = data.to(args.device)


                # Prerpare Attack Data
                attack_X, attack_Y = attack_trainer.prepare_attack_training_data(shadow_model, data, leak=leak)
                attack_train_data_X, attack_test_data_X, attack_train_data_y, attack_test_data_y = train_test_split(attack_X.numpy(), attack_Y.numpy(),
                                                                                                                    test_size=0.2,
                                                                                                                    stratify=attack_Y)  

                attack_train_data = torch.utils.data.TensorDataset(torch.from_numpy(attack_train_data_X).float(), torch.from_numpy(
                    attack_train_data_y))
                attack_train_loader = torch.utils.data.DataLoader(attack_train_data, batch_size=1024, shuffle=True)

                attack_test_data = torch.utils.data.TensorDataset(torch.from_numpy(attack_test_data_X).float(), torch.from_numpy(
                    attack_test_data_y))  # convert to float to fix  uint8_t overflow error
                attack_test_loader = torch.utils.data.DataLoader(attack_test_data, batch_size=1024, shuffle=True)

                # Train Attack model
                attack_model_all = MLPAttacker(args, leak=leak)
                attack_optimizer = torch.optim.Adam(attack_model_all.parameters(), lr=0.001)
                attack_trainer.train_attack(attack_model_all, attack_train_loader, attack_test_loader, attack_optimizer, leak, args) # Train
                attack_ckpt = torch.load(os.path.join(attack_path_all, f'{leak}_gd_attack_model_best.pt'))

            attack_model_all.load_state_dict(attack_ckpt['model_state'])
            attack_model_all = attack_model_all.to(device)
    else:
        attack_model_all = None
    attack_model_sub = None
        # attack_model_sub = MLPAttacker(args)
        # attack_ckpt = torch.load(os.path.join(attack_path_sub, 'attack_model_best.pt'))
        # attack_model_sub.load_state_dict(attack_ckpt['model_state'])
        # attack_model_sub = attack_model_sub.to(device)

    # Train (Unlearning)
    trainer = get_trainer(args)
    trainer.train(model, data, optimizer, args, logits_ori, attack_model_all, attack_model_sub)  # model is learned GNN model


    # Test
    if args.unlearning_model != 'retrain':
        retrain_path = os.path.join(
            'checkpoint', args.dataset, args.gnn, 'retrain', 
            '-'.join([str(i) for i in [args.df, args.df_size, args.random_seed]]), 
            'model_best.pt')
        if os.path.exists(retrain_path):
            retrain_ckpt = torch.load(retrain_path, map_location=device)
            retrain_args = copy.deepcopy(args)
            retrain_args.unlearning_model = 'retrain'
            retrain = get_model(retrain_args, num_nodes=data.num_nodes, num_edge_type=args.num_edge_type)
            retrain.load_state_dict(retrain_ckpt['model_state'])
            retrain = retrain.to(device)
            retrain.eval()
        else:
            retrain = None
    else:
        retrain = None

    if args.unlearning_model == 'gnndelete':
        shadow_model.to(args.device)
        data.to(args.device)
        model.to(args.device)
        test_results = trainer.test(model, data, model_retrain=retrain, attack_model_all=attack_model_all, attack_model_sub=attack_model_sub)
    else:
        model.to(args.device)
        data.to(args.device)
        test_results = trainer.test(model, data, model_retrain=retrain, attack_model_all=attack_model_all, attack_model_sub=attack_model_sub)
    print(test_results[-1])


    trainer.save_log()
    # send_email(args.random_seed)
    if args.gnn == 'gin' and args.dataset == 'CS':
        send_email(args.random_seed)

if __name__ == "__main__":
    main()



