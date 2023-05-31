import os
import time
import json
import wandb
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange, tqdm
from ogb.graphproppred import Evaluator
from torch_geometric.data import DataLoader
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.utils import negative_sampling, subgraph
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, mean_squared_error

from ..evaluation import *
from ..training_args import parse_args
from ..utils import *


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = 'cpu'

class Trainer:
    def __init__(self, args):
        self.args = args
        self.trainer_log = {
            'unlearning_model': args.unlearning_model, 
            'dataset': args.dataset, 
            'log': []}
        self.logit_all_pair = None
        self.df_pos_edge = []
        self.df_neg_edge = []

        with open(os.path.join(self.args.checkpoint_dir, 'training_args.json'), 'w') as f:
            json.dump(vars(args), f)

    def freeze_unused_weights(self, model, mask):
        grad_mask = torch.zeros_like(mask)
        grad_mask[mask] = 1

        model.deletion1.deletion_weight.register_hook(lambda grad: grad.mul_(grad_mask))
        model.deletion2.deletion_weight.register_hook(lambda grad: grad.mul_(grad_mask))
    
    @torch.no_grad()
    def get_link_labels(self, pos_edge_index, neg_edge_index):
        E = pos_edge_index.size(1) + neg_edge_index.size(1)
        link_labels = torch.zeros(E, dtype=torch.float, device=pos_edge_index.device)
        link_labels[:pos_edge_index.size(1)] = 1.
        return link_labels

    @torch.no_grad()
    def get_embedding(self, model, data, on_cpu=False):
        original_device = next(model.parameters()).device

        if on_cpu:
            model = model.cpu()
            data = data.cpu()
        
        z = model(data.x, data.train_pos_edge_index[:, data.dtrain_mask])

        model = model.to(original_device)

        return z

    def train(self, model, data, optimizer, args):
        if self.args.dataset in ['Cora', 'PubMed', 'DBLP', 'CS']:
            return self.train_fullbatch(model, data, optimizer, args)

        if self.args.dataset in ['Physics']:
            return self.train_minibatch(model, data, optimizer, args)

        if 'ogbl' in self.args.dataset:
            return self.train_minibatch(model, data, optimizer, args)

    def train_fullbatch(self, model, data, optimizer, args):
        start_time = time.time()
        best_valid_loss = 1000000

        data = data.to(self.args.device)
        for epoch in trange(args.epochs, desc='Epoch'):
            model.train()

            # Positive and negative sample
            neg_edge_index = negative_sampling(
                edge_index=data.train_pos_edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=data.dtrain_mask.sum())
            
            z = model(data.x, data.train_pos_edge_index)
            # edge = torch.cat([train_pos_edge_index, neg_edge_index], dim=-1)
            # logits = model.decode(z, edge[0], edge[1])
            logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)
            label = get_link_labels(data.train_pos_edge_index, neg_edge_index)
            loss = F.binary_cross_entropy_with_logits(logits, label)
            print(loss)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()

            if (epoch+1) % args.valid_freq == 0:
                valid_loss, dt_auc, dt_aup, df_auc, df_aup, dt_con_auc, dt_con_mse, df_logit, logit_all_pair, valid_log = self.eval(model, data, 'val')

                train_log = {
                    'epoch': epoch,
                    'train_loss': loss.item()
                }
                
                for log in [train_log, valid_log]:
                    # wandb.log(log)
                    msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                    tqdm.write(' | '.join(msg))

                self.trainer_log['log'].append(train_log)
                self.trainer_log['log'].append(valid_log)

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_epoch = epoch

                    print(f'Save best checkpoint at epoch {epoch:04d}. Valid loss = {valid_loss:.4f}')
                    ckpt = {
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }
                    torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))
                    torch.save(z, os.path.join(args.checkpoint_dir, 'node_embeddings.pt'))

        self.trainer_log['training_time'] = time.time() - start_time

        # Save models and node embeddings
        print('Saving final checkpoint')
        ckpt = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_final.pt'))

        print(f'Training finished. Best checkpoint at epoch = {best_epoch:04d}, best valid loss = {best_valid_loss:.4f}')

        self.trainer_log['best_epoch'] = best_epoch
        self.trainer_log['best_valid_loss'] = best_valid_loss
    # DF Consistency ((deleted dataset))
    def train_minibatch(self, model, data, optimizer, args):
        start_time = time.time()
        best_valid_loss = 1000000

        data.edge_index = data.train_pos_edge_index
        loader = GraphSAINTRandomWalkSampler(
            data, batch_size=args.batch_size, walk_length=2, num_steps=args.num_steps,
        )
        for epoch in trange(args.epochs, desc='Epoch'):
            model.train()

            epoch_loss = 0
            for step, batch in enumerate(tqdm(loader, desc='Step', leave=False)):
                # Positive and negative sample
                train_pos_edge_index = batch.edge_index.to(self.args.device)
                z = model(batch.x.to(self.args.device), train_pos_edge_index)

                neg_edge_index = negative_sampling(
                    edge_index=train_pos_edge_index,
                    num_nodes=z.size(0))
                
                logits = model.decode(z, train_pos_edge_index, neg_edge_index)
                label = get_link_labels(train_pos_edge_index, neg_edge_index)
                loss = F.binary_cross_entropy_with_logits(logits, label)

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()

                log = {
                    'epoch': epoch,
                    'step': step,
                    'train_loss': loss.item(),
                }
                # wandb.log(log)
                msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                tqdm.write(' | '.join(msg))

                epoch_loss += loss.item()

            if (epoch+1) % args.valid_freq == 0:
                valid_loss, dt_auc, dt_aup, df_auc, df_aup, dt_con_auc, dt_con_mse, df_logit, logit_all_pair, valid_log = self.eval(model, data, 'val')

                train_log = {
                    'epoch': epoch,
                    'train_loss': epoch_loss / step
                }
                
                for log in [train_log, valid_log]:
                    # wandb.log(log)
                    msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                    tqdm.write(' | '.join(msg))

                self.trainer_log['log'].append(train_log)
                self.trainer_log['log'].append(valid_log)

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_epoch = epoch

                    print(f'Save best checkpoint at epoch {epoch:04d}. Valid loss = {valid_loss:.4f}')
                    ckpt = {
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }
                    torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))
                    torch.save(z, os.path.join(args.checkpoint_dir, 'node_embeddings.pt'))

        self.trainer_log['training_time'] = time.time() - start_time

        # Save models and node embeddings
        print('Saving final checkpoint')
        ckpt = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_final.pt'))

        print(f'Training finished. Best checkpoint at epoch = {best_epoch:04d}, best valid loss = {best_valid_loss:.4f}')

        self.trainer_log['best_epoch'] = best_epoch
        self.trainer_log['best_valid_loss'] = best_valid_loss
        self.trainer_log['training_time'] = np.mean([i['epoch_time'] for i in self.trainer_log['log'] if 'epoch_time' in i])

    @torch.no_grad()
    def eval(self, model, data, stage='val', pred_all=False):
        model.eval()
        pos_edge_index = data[f'{stage}_pos_edge_index']
        neg_edge_index = data[f'{stage}_neg_edge_index']

        if self.args.eval_on_cpu:
            model = model.to('cpu')
        
        if hasattr(data, 'dtrain_mask'):
            mask = data.dtrain_mask
        else:
            mask = data.dr_mask
        z = model(data.x, data.train_pos_edge_index[:, mask])
        logits = model.decode(z, pos_edge_index, neg_edge_index).sigmoid()
        label = self.get_link_labels(pos_edge_index, neg_edge_index)

        # DT AUC AUP (remaining dataset)
        loss = F.binary_cross_entropy_with_logits(logits, label).cpu().item()
        dt_auc = roc_auc_score(label.cpu(), logits.cpu())
        dt_aup = average_precision_score(label.cpu(), logits.cpu())

        # DF AUC AUP ((deleted dataset))
        if self.args.unlearning_model in ['original']:
            df_logit = []
        else:
            # df_logit = model.decode(z, data.train_pos_edge_index[:, data.df_mask]).sigmoid().tolist()
            df_logit = model.decode(z, data.directed_df_edge_index).sigmoid().tolist()

        if len(df_logit) > 0:
            df_auc = []
            df_aup = []
        
            # Sample pos samples 采样500次求平均
            if len(self.df_pos_edge) == 0:
                for i in range(500):
                    mask = torch.zeros(data.train_pos_edge_index[:, data.dr_mask].shape[1], dtype=torch.bool) # tensor([False, False, False,  ..., False, False, False]) |E|
                    idx = torch.randperm(data.train_pos_edge_index[:, data.dr_mask].shape[1])[:len(df_logit)] # 在remain的里面抽样 df个样本
                    mask[idx] = True
                    self.df_pos_edge.append(mask)  # 作为正样本
            
            # Use cached pos samples
            for mask in self.df_pos_edge:
                pos_logit = model.decode(z, data.train_pos_edge_index[:, data.dr_mask][:, mask]).sigmoid().tolist()
                
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
            for mask in self.df_neg_edge:
                neg_logit = model.decode(z, neg_edge_index[:, mask]).sigmoid().tolist()

                # df_con_mse.append(mean_squared_error(df_logit, neg_logit))
                df_con_mse = 0

                logit = df_logit + neg_logit  # 285 + 285
                label = [0] * len(df_logit) +  [1] * len(df_logit)
                # df_con_auc.append(roc_auc_score(label, logit))
                df_con_auc = 0

            df_con_auc = np.mean(df_con_auc)
            df_con_mse = np.mean(df_con_mse)
        else:
            df_con_auc = np.nan
            df_con_mse = np.nan



        # Logits for all node pairs
        if pred_all:
            logit_all_pair = (z @ z.t()).cpu()
        else:
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
    def eval_GraphEraser(self, shard_logits, model, data, stage='val', aggregator='mean', pred_all=False):
        model.eval()
        pos_edge_index = data[f'{stage}_pos_edge_index']
        neg_edge_index = data[f'{stage}_neg_edge_index']

        # DT evaluation:
        dt_logtis = shard_logits['0']['dt']
        for shard in range(self.num_shards):
            dt_logtis += shard_logits[f'{shard}']['dt']

        dt_logtis = dt_logtis/self.num_shards
        label = self.get_link_labels(pos_edge_index, neg_edge_index)
        loss = F.binary_cross_entropy_with_logits(dt_logtis, label).cpu().item()
        dt_auc = roc_auc_score(label.cpu(), dt_logtis.cpu())
        dt_aup = average_precision_score(label.cpu(), dt_logtis.cpu())


        # DF AUC AUP
        df_auc = []
        df_aup = []
        for mask_id in range(len(shard_logits['1']['df'])):
            df_logtis = np.array(shard_logits['0']['df'][mask_id])

            for shard in range(1, self.num_shards):
                df_logtis += np.array(shard_logits[f'{shard}']['df'][mask_id])
            df_logits = df_logtis/self.num_shards
            label = [0] * int(df_logits.shape[0]/2) +  [1] * int(df_logits.shape[0]/2)
            df_auc.append(roc_auc_score(label, df_logits))
            df_aup.append(average_precision_score(label, df_logits))

        df_auc = np.mean(df_auc)
        df_aup = np.mean(df_aup)

        # DF Consistency ((deleted dataset))
        df_con_auc = []
        df_con_mse = []

        for mask_id in range(len(shard_logits['1']['df2'])):
            df_logtis_2 = np.array(shard_logits['0']['df2'][mask_id])

            for shard in range(1, self.num_shards):
                df_logtis_2 += np.array(shard_logits[f'{shard}']['df2'][mask_id])
            df_logtis_2 = df_logtis_2/self.num_shards
            label = [0] * int(df_logtis_2.shape[0]/2) +  [1] * int(df_logtis_2.shape[0]/2)
            df_con_auc.append(roc_auc_score(label, df_logtis_2))
            df_con_mse.append(mean_squared_error(df_logtis_2[:int(df_logtis_2.shape[0]/2)], df_logtis_2[int(df_logtis_2.shape[0]/2):]))

        df_con_auc = np.mean(df_con_auc)
        df_con_mse = np.mean(df_con_mse)





        # Logits for all node pairs
        if pred_all:
            logit_all_pair = None
        else:
            logit_all_pair = None

        log = {
            f'{stage}_loss': loss,
            f'{stage}_dt_auc': dt_auc,
            f'{stage}_dt_aup': dt_aup,
            f'{stage}_df_auc': df_auc,
            f'{stage}_df_aup': df_aup,
            f'{stage}_df_con_auc': df_con_auc,
            f'{stage}_df_con_mse': df_con_mse,
            f'{stage}_df_logit_mean': np.nan,
            f'{stage}_df_logit_std': np.nan
        }

        if self.args.eval_on_cpu:
            model = model.to(self.args.device)

        return loss, dt_auc, dt_aup, df_auc, df_aup, df_con_auc, df_con_mse, df_logits, logit_all_pair, log

    @torch.no_grad()
    def logits_gen_GraphEraser(self, model, data, stage='val', pred_all=False):
        model.eval()
        pos_edge_index = data[f'{stage}_pos_edge_index']
        neg_edge_index = data[f'{stage}_neg_edge_index']

        if self.args.eval_on_cpu:
            model = model.to('cpu')

        if hasattr(data, 'dtrain_mask'):
            mask = data.dtrain_mask
        else:
            mask = data.dr_mask
        z = model(data.x, data.train_pos_edge_index[:, mask])
        dt_logits = model.decode(z, pos_edge_index, neg_edge_index).sigmoid()


        # DF AUC AUP
        if self.args.unlearning_model in ['original']:
            df_logits = []
        else:
            # df_logit = model.decode(z, data.train_pos_edge_index[:, data.df_mask]).sigmoid().tolist()
            df_logits = model.decode(z, data.directed_df_edge_index).sigmoid().tolist()

        df_logits_list = []
        df_logits_list_2 = []
        if len(df_logits) > 0:
            # Sample pos samples
            if len(self.df_pos_edge) == 0:
                for i in range(500):
                    mask = torch.zeros(data.train_pos_edge_index[:, data.dr_mask].shape[1], dtype=torch.bool)
                    idx = torch.randperm(data.train_pos_edge_index[:, data.dr_mask].shape[1])[:len(df_logits)]
                    mask[idx] = True
                    self.df_pos_edge.append(mask)

            # Use cached pos samples

            for mask_id, mask in enumerate(self.df_pos_edge):
                pos_logit = model.decode(z, data.train_pos_edge_index[:, data.dr_mask][:, mask]).sigmoid().tolist()

                logit = df_logits + pos_logit
                df_logits_list.append(logit)

            if len(self.df_neg_edge) == 0:
                for i in range(500):
                    mask = torch.zeros(neg_edge_index.shape[1], dtype=torch.bool) # tensor([False, False, False,  ..., False, False, False]) |E|
                    idx = torch.randperm(neg_edge_index.shape[1])[:len(df_logits)] # 在remain的里面抽样 df个样本
                    mask[idx] = True
                    self.df_neg_edge.append(mask)
            for mask in self.df_neg_edge:
                neg_logit = model.decode(z, neg_edge_index[:, mask]).sigmoid().tolist()
                logit_neg = df_logits + neg_logit  # 285 + 285
                df_logits_list_2.append(logit_neg)
        else:
            pass

        return dt_logits, df_logits_list, df_logits_list_2

    @torch.no_grad()
    def test(self, model, data, model_retrain=None, attack_model_all=None, attack_model_sub=None, ckpt='best'):
        if not self.__class__.__name__ == 'GraphEraserTrainer': # Not GraphEraser
            if ckpt == 'best':    # Load best ckpt
                if self.args.unlearning_model == 'moo' and self.args.lmd != 'moo':
                    ckpt = torch.load(os.path.join(self.args.checkpoint_dir, f'model_best_{self.args.lmd}.pt'))
                else:
                    ckpt = torch.load(os.path.join(self.args.checkpoint_dir, 'model_best.pt'))
                # if self.args.unlearning_model == 'gnndelete':
                #     del ckpt['model_state']['deletion1.deletion_weight']
                #     del ckpt['model_state']['deletion2.deletion_weight']
                model.load_state_dict(ckpt['model_state'])
                model.to(self.args.device)
            elif ckpt == 'final':
                if self.args.unlearning_model == 'moo' and self.args.lmd != 'moo':
                    ckpt = torch.load(os.path.join(self.args.checkpoint_dir, f'model_final_{self.args.lmd}.pt'))
                else:
                    ckpt = torch.load(os.path.join(self.args.checkpoint_dir, 'model_final.pt'))
                if self.args.unlearning_model == 'gnndelete':
                    del ckpt['model_state']['deletion1.deletion_weight']
                    del ckpt['model_state']['deletion2.deletion_weight']
                model.load_state_dict(ckpt['model_state'])
                model.to(self.args.device)


            if 'ogbl' in self.args.dataset:
                pred_all = False
            else:
                pred_all = True
            loss, dt_auc, dt_aup, df_auc, df_aup, df_con_auc, df_con_mse, df_logit, logit_all_pair, test_log = self.eval(model, data, 'test', pred_all)
        else:  # GraphEraser
            shard_logits = {}
            if 'ogbl' in self.args.dataset:
                pred_all = False
            else:
                pred_all = True
            mi_sucrate_all_after_list = []
            for shard_id in range(self.num_shards):
                ckpt_shard = torch.load(os.path.join(self.args.checkpoint_dir, f'model_{shard_id}.pt'))
                model_shard_id = copy.deepcopy(model).to(self.args.device)
                model_shard_id.load_state_dict(ckpt_shard)
                dt_logits, df_logits_list, df_logits_list_2 = self.logits_gen_GraphEraser(model_shard_id, data, 'test', pred_all)
                shard_logits[f'{shard_id}'] = {}
                shard_logits[f'{shard_id}']['dt'] = dt_logits
                shard_logits[f'{shard_id}']['df'] = df_logits_list
                shard_logits[f'{shard_id}']['df2'] = df_logits_list_2


                # MI Evaluation
                subset_train = self.subsets['shard_id'].to(self.args.device)


                subset_train_T = subset_train.T.cpu().numpy()
                df_T = data.train_pos_edge_index[:, data.df_mask].T.cpu().numpy()

                m = [(df_T==irow).all(axis=1).any() for irow in subset_train_T]

                subset_df = torch.from_numpy(subset_train_T[m].T).to(self.args.device)
                # print(subset_train_T[m])
                # m = (subset_train_T[:, None] == df_T).all(-1).any(1)

                # if (m == False).all():
                #     print('0000')
                # else:
                #     print('1111')


                edge = subset_df
                z = model_shard_id(data.x, subset_train)
                attack_feature_1 = model_shard_id.decode(z, edge).sigmoid()
                attack_feature_0 = 1 - attack_feature_1
                attack_feature = torch.stack([attack_feature_0, attack_feature_1], dim=1)
                logits = attack_model_all(attack_feature)
                _, pred = torch.max(logits, 1)
                suc_rate = 1 - pred.float().mean()
                mi_logit_all_after = torch.softmax(logits, dim=-1).squeeze().tolist()
                mi_sucrate_all_after = suc_rate.cpu().item()
                mi_sucrate_all_after_list.append(mi_sucrate_all_after)


            self.trainer_log['mi_logit_all_after'] = 0
            self.trainer_log['mi_sucrate_all_after'] = min(mi_sucrate_all_after_list)

            loss, dt_auc, dt_aup, df_auc, df_aup, df_con_auc, df_con_mse, df_logit,  logit_all_pair, test_log = self.eval_GraphEraser(shard_logits, model, data, 'test', 'mean' , pred_all)


        self.trainer_log['dt_loss'] = loss
        self.trainer_log['dt_auc'] = dt_auc
        self.trainer_log['dt_aup'] = dt_aup
        try:
            self.trainer_log['df_logit'] = df_logit.tolist()
        except AttributeError:
            self.trainer_log['df_logit'] = df_logit
        self.logit_all_pair = logit_all_pair
        self.trainer_log['df_auc'] = df_auc
        self.trainer_log['df_aup'] = df_aup
        self.trainer_log['df_con_auc'] = df_con_auc
        self.trainer_log['df_con_auc'] = df_con_mse
        self.trainer_log['auc_sum'] = dt_auc + df_auc
        self.trainer_log['aup_sum'] = dt_aup + df_aup
        self.trainer_log['auc_gap'] = abs(dt_auc - df_auc)
        self.trainer_log['aup_gap'] = abs(dt_aup - df_aup)
        # # AUC AUP on Df
        # if len(df_logit) > 0:
        #     auc = []
        #     aup = []

        #     if self.args.eval_on_cpu:
        #         model = model.to('cpu')
            
        #     z = model(data.x, data.train_pos_edge_index[:, data.dtrain_mask])
        #     for i in range(500):
        #         mask = torch.zeros(data.train_pos_edge_index[:, data.dr_mask].shape[1], dtype=torch.bool)
        #         idx = torch.randperm(data.train_pos_edge_index[:, data.dr_mask].shape[1])[:len(df_logit)]
        #         mask[idx] = True
        #         pos_logit = model.decode(z, data.train_pos_edge_index[:, data.dr_mask][:, mask]).sigmoid().tolist()

        #         logit = df_logit + pos_logit
        #         label = [0] * len(df_logit) +  [1] * len(df_logit)
        #         auc.append(roc_auc_score(label, logit))
        #         aup.append(average_precision_score(label, logit))

        #     self.trainer_log['df_auc'] = np.mean(auc)
        #     self.trainer_log['df_aup'] = np.mean(aup)


        if model_retrain is not None:    # Deletion
            self.trainer_log['ve'] = verification_error(model, model_retrain).cpu().item()
            # self.trainer_log['dr_kld'] = output_kldiv(model, model_retrain, data=data).cpu().item()

        if not self.__class__.__name__ == 'GraphEraserTrainer': # Not GraphEraser
            # MI Attack after unlearning
            model.to(self.args.device)
            if attack_model_all is not None:
                mi_logit_all_after, mi_sucrate_all_after = member_infer_attack(model, attack_model_all, data)
                self.trainer_log['mi_logit_all_after'] = mi_logit_all_after
                self.trainer_log['mi_sucrate_all_after'] = mi_sucrate_all_after
            if attack_model_sub is not None:
                mi_logit_sub_after, mi_sucrate_sub_after = member_infer_attack(model, attack_model_sub, data)
                self.trainer_log['mi_logit_sub_after'] = mi_logit_sub_after
                self.trainer_log['mi_sucrate_sub_after'] = mi_sucrate_sub_after

                self.trainer_log['mi_ratio_all'] = np.mean([i[1] / j[1] for i, j in zip(self.trainer_log['mi_logit_all_after'], self.trainer_log['mi_logit_all_before'])])
                self.trainer_log['mi_ratio_sub'] = np.mean([i[1] / j[1] for i, j in zip(self.trainer_log['mi_logit_sub_after'], self.trainer_log['mi_logit_sub_before'])])
                print(self.trainer_log['mi_ratio_all'], self.trainer_log['mi_ratio_sub'], self.trainer_log['mi_sucrate_all_after'], self.trainer_log['mi_sucrate_sub_after'])
                print(self.trainer_log['df_auc'], self.trainer_log['df_aup'])

        return loss, dt_auc, dt_aup, df_auc, df_aup, df_con_auc, df_con_mse, df_logit, logit_all_pair, test_log

    @torch.no_grad()
    def get_output(self, model, node_embedding, data):
        model.eval()
        node_embedding = node_embedding.to(self.args.device)
        edge = data.edge_index.to(self.args.device)
        output = model.decode(node_embedding, edge, edge_type)

        return output

    def save_log(self):
        # print(self.trainer_log)
        if self.args.lmd == 'moo':
            with open(os.path.join(self.args.checkpoint_dir, 'trainer_log.json'), 'w') as f:
                json.dump(self.trainer_log, f)
        else:
            with open(os.path.join(self.args.checkpoint_dir, f'trainer_log_{self.args.lmd}.json'), 'w') as f:
                json.dump(self.trainer_log, f)
        # with open(os.path.join(self.args.checkpoint_dir, 'trainer_log.json'), 'r') as f:
        #     result = json.load(f)
        
        # torch.save(self.logit_all_pair, os.path.join(self.args.checkpoint_dir, 'pred_proba.pt'))



