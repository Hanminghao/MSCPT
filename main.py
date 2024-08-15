import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from model import MInterface
from data import DInterface
from utils import load_model_path_by_args
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# os.environ["WANDB_MODE"]="offline"
import wandb
wandb.init(mode="offline")
label_dicts = {
    'NSCLC_subtyping': {'LUAD': 0, 'LUSC': 1},
    'BRCA_subtyping': {'IDC': 0, 'ILC': 1},
    'RCC_subtyping': {'CHRCC': 0, 'CCRCC': 1, 'PRCC': 2},
    'COADREAD_subtyping': {'COAD': 0, 'READ': 1},
    'CAMELYON16': {'Normal': 0, 'Tumor': 1}
}
class MeterlessProgressBar(TQDMProgressBar):

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.dynamic_ncols = False
        bar.ncols = 0
        return bar
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.dynamic_ncols = False
        bar.ncols = 0
        return bar


def main(args):

    pl.seed_everything(args.seed)
    load_path = load_model_path_by_args(args)
    data_module = DInterface(**vars(args))

    if args.model_name == 'linear':
        name = f'{args.model_name}_epochs_{args.max_epochs}_seed_{args.seed}_numshots_{args.num_shots}_npro_{args.n_vpro}_aggregation_{args.linear_aggregation}_scale_{args.coop_scale}'
    elif args.model_name in ['coop', 'cocoop', 'maple', 'metaprompt']:
        name = f'{args.model_name}_epochs_{args.max_epochs}_seed_{args.seed}_numshots_{args.num_shots}_npro_{args.n_vpro}_scale_{args.coop_scale}'
    elif 'ablation' in args.model_name:
        name = f'{args.model_name}_epochs_{args.max_epochs}_seed_{args.seed}_numshots_{args.num_shots}_ISGPT_{str(args.ISGPT)}_CGP_{str(args.CGP)}'
        print('GPT:', args.ISGPT, 'CGP:', args.CGP)
    elif 'aggregation' in args.model_name:
        name = f'{args.model_name}_epochs_{args.max_epochs}_seed_{args.seed}_numshots_{args.num_shots}_aggregation_{args.aggregation}'
    elif 'graph' in args.model_name:
        name = f'{args.model_name}_epochs_{args.max_epochs}_seed_{args.seed}_numshots_{args.num_shots}_graph_base_{args.graph_base}_graph_type_{args.graph_type}'
    elif args.model_name =='mscpt':
        name = f'{args.model_name}_seed_{args.seed}_numshots_{args.num_shots}_LLM_{args.LLM}'
    else:
        name = f'{args.model_name}_seed_{args.seed}_numshots_{args.num_shots}'
    
    logger = WandbLogger(project=args.logger_name, name=name)
    if args.model_name == 'linear':
        args.txt_result_path = os.path.join('./result', args.base_model, args.dataset_name, args.linear_aggregation)
    else:
        args.txt_result_path = os.path.join('./result', args.base_model, args.dataset_name)
    args.heatmap_path = os.path.join('./heatmap', args.base_model, args.dataset_name, args.model_name)
    args.wandb_name = name
    if not os.path.exists(args.txt_result_path):
        os.makedirs(args.txt_result_path)
    if not os.path.exists(args.heatmap_path):
        os.makedirs(args.heatmap_path)
    args.callbacks=[]
    bar = MeterlessProgressBar()
    args.callbacks.append(bar)
    early_stop_callback = EarlyStopping(monitor="val_best_score", min_delta=0.00, patience=args.patience, verbose=True, mode="max")
    args.callbacks.append(early_stop_callback)
    
    args.logger = logger
    if load_path is None:
        model = MInterface(**vars(args))
    else:
        model = MInterface(**vars(args))
        args.ckpt_path = load_path
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, data_module)


if __name__ == '__main__':
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--seed', default=11, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--use_gpu', default=True, type=bool, help='use cpu or gpu')
    parser.add_argument('--device_ids' ,default=[0])
    parser.add_argument('--num_shots', default=16, type=int, help='num of few-shot')
    parser.add_argument('--total_epochs', default=30, type=int, help='num of epochs')

    # LR Scheduler
    parser.add_argument('--lr_scheduler', choices=['step', 'cosine'], type=str)
    parser.add_argument('--lr_decay_steps', default=20, type=int)
    parser.add_argument('--lr_decay_rate', default=0.5, type=float)
    parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)

    # Restart Control
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=int)

    # Training Info
    parser.add_argument('--dataset', default='my_data', type=str)   #my_data
    parser.add_argument('--dataset_name', default='RCC', type=str, choices=['RCC', 'Lung', 'BRCA'])
    parser.add_argument('--data_dir', default='SELECTED_PATCHES_DIRECTORY', type=str)
    parser.add_argument('--feat_data_dir', default='FEATURES_DIRECTORY', type=str)
    parser.add_argument('--description_dir', default='./train_data/')
    parser.add_argument('--csv_dir', default='./tcga_rcc.csv')
    parser.add_argument('--model_name', default='path_gnn', type=str)
    parser.add_argument('--loss', default='ce', type=str)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--log_dir', default='lightning_logs', type=str)
    parser.add_argument('--task', default='RCC_subtyping', type=str)
    parser.add_argument('--target_size', default=224, type=int)
    parser.add_argument('--gc', default=1, type=int, help='Gradient Accumulation')
    parser.add_argument('--val', action='store_true')
    
    # Model Hyperparameters
    parser.add_argument('--hid', default=64, type=int)
    parser.add_argument('--block_num', default=8, type=int)
    parser.add_argument('--in_channel', default=3, type=int)
    parser.add_argument('--layer_num', default=5, type=int)

    # Other
    parser.add_argument('--aug_prob', default=0.5, type=float)

    # Model parameters
    parser.add_argument('--base_model', default='plip', type=str, help='Base Model Name')
    parser.add_argument('--trainer_perc', default=32, type=int, help='双精，单精度， 混合精度')
    parser.add_argument('--n_set', default=5, type=int, help='length of des')
    parser.add_argument('--n_tpro', default=5, type=int, help='length of text prompt')
    parser.add_argument('--n_vpro', default=5, type=int, help='length of vision prompt')
    parser.add_argument('--n_high', default=10, type=int)
    parser.add_argument('--n_topk', default=5, type=int, help='types of imgs')
    parser.add_argument('--coop_scale', default=-1, type=int, help='scale of coop')
    parser.add_argument('--LLM', default='GPT_4', type=str, help='GPT_4, claude3, gemini, llama3' )

    # Add pytorch lightning's args to parser as a group.
    parser = Trainer.add_argparse_args(parser)


    args = parser.parse_args()
    if args.dataset_name == 'RCC':
        args.csv_dir = './tcga_rcc.csv'
        args.task = 'RCC_subtyping'
        args.logger_name = 'mscpt_rcc'
    elif args.dataset_name == 'Lung':
        args.csv_dir = './tcga_lung.csv'
        args.task = 'NSCLC_subtyping'
        args.logger_name = 'mscpt_lung'
    elif args.dataset_name == 'BRCA':
        args.csv_dir = './tcga_brca.csv'
        args.task = 'BRCA_subtyping'
        args.logger_name = 'mscpt_brca'

    if args.base_model == 'clip':
        args.patience = 20
    elif args.base_model == 'plip':
        args.patience = 10
    args.max_epochs = args.total_epochs
    if args.use_gpu:
        args.accelerator = 'gpu'
        args.devices = args.device_ids
    else:
        args.accelerator = 'cpu'
    if args.trainer_perc != 32:
        args.precision = args.trainer_perc
    args.mean_sen = [0.485, 0.456, 0.406]
    args.std_sen = [0.229, 0.224, 0.225]
    args.num_sanity_val_steps = 0
    args.label_dicts = label_dicts[args.task]
    
    main(args)
