from .models import GCN, GAT, GIN, GCNDelete, GATDelete, GINDelete
from .trainer.base import Trainer
from .trainer.retrain import RetrainTrainer, KGRetrainTrainer
from .trainer.gnndelete import GNNDeleteTrainer
from .trainer.gradient_ascent import GradientAscentTrainer, KGGradientAscentTrainer
from .trainer.descent_to_delete import DtdTrainer
from .trainer.graph_eraser import GraphEraserTrainer
from .trainer.graph_editor import GraphEditorTrainer
from .trainer.member_infer import MIAttackTrainer

# Ours
from .trainer.summit import MooTrainer

trainer_mapping = {
    'original': Trainer,
    'original_node': NodeClassificationTrainer,
    'retrain': RetrainTrainer,
    'gnndelete': GNNDeleteTrainer,
    'gradient_ascent': GradientAscentTrainer,
    'descent_to_delete': DtdTrainer,
    'graph_eraser': GraphEraserTrainer,
    'graph_editor': GraphEditorTrainer,
    'member_infer_all': MIAttackTrainer,
    'member_infer_sub': MIAttackTrainer,
    'summit': MooTrainer,
}



def get_shadow_model(args, mask_1hop=None, mask_2hop=None, num_nodes=None, num_edge_type=None):
    model_mapping = {'gcn': GCN, 'gat': GAT, 'gin': GIN, 'rgcn': RGCN, 'rgat': RGAT}

    return model_mapping[args.gnn](args, mask_1hop=mask_1hop, mask_2hop=mask_2hop, num_nodes=num_nodes, num_edge_type=num_edge_type)

def get_model(args, mask_1hop=None, mask_2hop=None, num_nodes=None, num_edge_type=None):
    if 'gnndelete' in args.unlearning_model:
        model_mapping = {'gcn': GCNDelete, 'gat': GATDelete, 'gin': GINDelete, 'rgcn': RGCNDelete, 'rgat': RGATDelete}
    else:
        model_mapping = {'gcn': GCN, 'gat': GAT, 'gin': GIN, 'rgcn': RGCN, 'rgat': RGAT}

    return model_mapping[args.gnn](args, mask_1hop=mask_1hop, mask_2hop=mask_2hop, num_nodes=num_nodes, num_edge_type=num_edge_type)


def get_trainer(args):
    if args.gnn in ['rgcn', 'rgat']:
        return kg_trainer_mapping[args.unlearning_model](args)

    else:
        return trainer_mapping[args.unlearning_model](args)

def get_attacker(args):
    return trainer_mapping['member_infer_all'](args)