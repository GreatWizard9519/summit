import os
import time
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import GraphSAINTRandomWalkSampler  # minibatch

from .base import Trainer
from ..evaluation import *
from ..utils import *

from ..models import GCN, GAT, GIN

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class MooTrainer(Trainer):
    def train(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        if 'ogbl' in self.args.dataset:
            return self.train_minibatch(model, data, optimizer, args, logits_ori, attack_model_all, attack_model_sub)

        else:
            return self.train_fullbatch(model, data, optimizer, args, logits_ori, attack_model_all, attack_model_sub)

    def gather_rows(self, input, index):
        """Gather the rows specificed by index from the input tensor"""
        return torch.gather(input, 0, index.unsqueeze(-1).expand((-1, input.shape[1])))

    def node_pair_kld(self, args, mu, sigma, u_index, v_index):
        mu_u = self.gather_rows(mu, u_index)
        mu_v = self.gather_rows(mu, v_index)

        sigma_u = self.gather_rows(sigma, u_index)
        sigma_v = self.gather_rows(sigma, v_index)

        diff_uv = mu_u - mu_v
        ob = torch.abs(diff_uv).mean()
        ratio_vu = sigma_v / sigma_u
        kld = 1 - torch.log(0.1 + 0.5 * (
                ratio_vu.sum(axis=-1)
                + (diff_uv ** 2 / sigma_u).sum(axis=-1)
                - args.hidden_dim / 2
                - torch.log(ratio_vu).sum(axis=-1)
        ))

        return kld

    def z_kld(self, args, mu, sigma, mu_moo, sigma_moo):
        diff = mu_moo - mu
        ratio = sigma / sigma_moo
        kld = 0.5 * torch.sum(ratio.sum(axis=-1) + (diff ** 2 / sigma_moo).sum(axis=-1) - (args.hidden_dim/2) - torch.log(ratio).sum(axis=-1))
        return kld

    def freeze_param(self, model):
        for name, child in model.named_children():
            for param in child.parameters():
                param.requires_grad = False
            self.freeze_param(child)



    def reparameterize(self, mu, logvar, training_status):  
        if training_status:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu



    def get_model(self, args, data):
        mask_1hop = data.sdf_node_1hop_mask
        mask_2hop = data.sdf_node_2hop_mask
        num_nodes = data.num_nodes
        num_edge_type = args.num_edge_type
        model_mapping = {'gcn': GCN, 'gat': GAT, 'gin': GIN}

        return model_mapping[args.gnn](args, mask_1hop=mask_1hop, mask_2hop=mask_2hop, num_nodes=num_nodes, num_edge_type=num_edge_type)

    def train_fullbatch(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        model.eval()
        if self.args.dataset == 'Cora' and self.args.gnn == 'gin':
            model.cpu()
            with torch.no_grad():
                _, mu, sigma = model.encode(data.x, data.train_pos_edge_index[:, data.dr_mask])
                z = model(data.x, data.train_pos_edge_index[:, data.dr_mask])
            z = z.to(self.args.device)
            mu = mu.to(self.args.device)
            sigma = sigma.to(self.args.device)
            model = model.to(self.args.device)  # learned
            data = data.to(self.args.device)
        else:
            model = model.to(self.args.device)  # learned
            data = data.to(self.args.device)
            with torch.no_grad():
                _, mu, sigma = model.encode(data.x, data.train_pos_edge_index[:, data.dr_mask])
                z = model(data.x, data.train_pos_edge_index[:, data.dr_mask])



        moo = self.get_model(args, data).to(self.args.device)
        moo.load_state_dict(model.state_dict())
        optimizer_moo = torch.optim.Adam(moo.parameters(), lr=args.lr)

        kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

        model.eval()
        self.freeze_param(model)

        best_metric = 0
        if attack_model_all is not None:
            mi_logit_all_before, mi_sucrate_all_before = member_infer_attack(model, attack_model_all, data)
            self.trainer_log['mi_logit_all_before'] = mi_logit_all_before
            self.trainer_log['mi_sucrate_all_before'] = mi_sucrate_all_before
        if attack_model_sub is not None:
            mi_logit_sub_before, mi_sucrate_sub_before = member_infer_attack(model, attack_model_sub, data)
            self.trainer_log['mi_logit_sub_before'] = mi_logit_sub_before
            self.trainer_log['mi_sucrate_sub_before'] = mi_sucrate_sub_before

        self.trainer_log['train_loss_all'] = []
        for epoch in trange(args.epochs, desc='Unlearning'):
            moo.train()
            start_time = time.time()
            total_step = 0
            total_loss = 0

            neg_edge_index = negative_sampling(
                edge_index=data.train_pos_edge_index[:, data.dr_mask],
                num_nodes=data.num_nodes,
                num_neg_samples=data.dr_mask.sum())


            with torch.no_grad():
                logits_y_r = model.decode(z, data.train_pos_edge_index[:, data.dr_mask], neg_edge_index)


            # -------------forgetting loss-------------  # Cora and CS' feature dimensions are large! GPU may overflow!
            ## loss_fgt_t1
            u_index = data.train_pos_edge_index[:, data.df_mask][0] # deleted node-level index u
            v_index = data.train_pos_edge_index[:, data.df_mask][1] # deleted node-level index v
            _, mu_moo, sigma_moo = moo.encode(data.x, data.train_pos_edge_index[:, data.dr_mask])
            loss_fgt_t1 = torch.mean(self.node_pair_kld(args, mu_moo, sigma_moo, u_index, v_index))  #


            z_moo = moo(data.x, data.train_pos_edge_index[:, data.dr_mask])
            training_status = moo.training
            logits_y_f_moo = moo.decode(z_moo, data.train_pos_edge_index[:, data.df_mask], neg_edge_index)
            label_f = self.get_link_labels(data.train_pos_edge_index[:, data.df_mask], neg_edge_index)
            loss_fgt_t2 = F.binary_cross_entropy_with_logits(logits_y_f_moo, label_f)




            ## loss_fgt
            loss_fgt = loss_fgt_t1 + loss_fgt_t2  # loss_fgt_t1: node kld,  loss_fgt_t1: moo_repre on D_r
            # ------------------------------------------


            # -------------remembering loss-------------
            ## loss_rmb_t1
            loss_rmb_t1 = self.z_kld(args, mu, sigma, mu_moo, sigma_moo)  

            ## loss_rmb_t2
            logits_y_r_moo = moo.decode(z_moo, data.train_pos_edge_index[:, data.dr_mask], neg_edge_index)
            loss_rmb_t2 = kl_loss(F.log_softmax(logits_y_r_moo, dim=-1), F.log_softmax(logits_y_r, dim=-1)) 
            loss_rmb =  loss_rmb_t1 + loss_rmb_t2
            # ------------------------------------------

            # -------------This is a practical implementation of MGDA for MOO-------------
            if args.lmd == 'moo':
                if loss_fgt <= 0:
                    lmd = 0
                else:
                    lmd = loss_fgt / (loss_fgt + loss_rmb)
            else:
                lmd = float(args.lmd)
            # ----------------------------------------------------


            # --------Final unlearning loss---------
            print('fgt loss:', loss_fgt)
            print('rmb loss:', loss_rmb)
            print('lmd', lmd)
            loss = lmd * loss_fgt + (1 - lmd) * loss_rmb
            self.trainer_log['train_loss_all'].append(loss.item())
            print()


            print('loss', loss)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(moo.parameters(), 1)
            optimizer_moo.step()
            optimizer_moo.zero_grad()

            total_step += 1
            total_loss += loss.item()

            end_time = time.time()
            epoch_time = end_time - start_time

            step_log = {
                'Epoch': epoch,
                'train_loss': loss.item(),
                'train_time': epoch_time
            }

            msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in step_log.items()]
            tqdm.write(' | '.join(msg))

            if (epoch + 1) % self.args.valid_freq == 0:
                valid_loss, dt_auc, dt_aup, df_auc, df_aup, df_con_auc, df_con_mse, df_logit, logit_all_pair, valid_log = self.eval(moo, data, 'val')
                valid_log['epoch'] = epoch

                train_log = {
                    'epoch': epoch,
                    'train_loss': loss.item(),
                    'train_time': epoch_time,
                }

                for log in [train_log, valid_log]:
                    msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                    tqdm.write(' | '.join(msg))
                    self.trainer_log['log'].append(log)


                metric = dt_auc + df_auc
                if metric > best_metric:
                    best_metric = metric
                    best_epoch = epoch

                    print(f'Save best checkpoint at epoch {epoch:04d}. Valid loss = {valid_loss:.4f}')
                    ckpt = {
                        'model_state': moo.state_dict(),
                        'optimizer_state': optimizer_moo.state_dict(),
                    }
                    if args.lmd == 'moo':
                        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))
                    else:
                        torch.save(ckpt, os.path.join(args.checkpoint_dir, f'model_best_{args.lmd}.pt'))

        self.trainer_log['training_time'] = time.time() - start_time


        # Save
        ckpt = {
            'model_state': moo.state_dict(),

            'optimizer_state': optimizer_moo.state_dict(),
        }
        if args.lmd == 'moo':
            torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_final.pt'))
        else:
            torch.save(ckpt, os.path.join(args.checkpoint_dir, f'model_final_{args.lmd}.pt'))

        print(f'Training finished. Best checkpoint at epoch = {best_epoch:04d}, best metric = {best_metric:.4f}')

        self.trainer_log['best_epoch'] = best_epoch
        self.trainer_log['best_metric'] = best_metric

    def train_minibatch(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        pass


    