import argparse
import pandas as pd
import time
import torch
import torch.multiprocessing
from pathlib import Path
import torch as th
from torch.utils.data import DataLoader
from utils.data.dataset import read_dataset, AugmentedDataset
from utils.data.collate import (
    seq_to_eop_multigraph,
    seq_to_shortcut_graph,
    collate_fn_factory,
)
from utils.train import TrainRunner
from PASBR import PILL
from multiprocessing import freeze_support
# En PyTorch 2.3, solo 'file_descriptor' es válido como sharing_strategy
torch.multiprocessing.set_sharing_strategy('file_descriptor')

torch.cuda.set_per_process_memory_fraction(0.6, 0)
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()



category_dict = {}
price_dict = {}

def get_collate_fn(num_layers):
    if num_layers > 1:
        return collate_fn_factory(seq_to_eop_multigraph, seq_to_shortcut_graph, cate_dict=category_dict, price_dict=price_dict)
    else:
        return collate_fn_factory(seq_to_eop_multigraph, cate_dict=category_dict, price_dict=price_dict)



def ayno():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
    '--dataset-dir', default='./datasets/yoochoose1_10', help='the dataset directory'
    )
    parser.add_argument('--embedding-dim', type=int, default=32, help='the embedding size')
    parser.add_argument('--num-layers', type=int, default=4, help='the number of layers')
    parser.add_argument('--without_intent', type=int, default=0, help='0: including intention; 1: No intention')
    parser.add_argument('--without_price', type=int, default=0, help='0: including price; 1: No price')
    # TODO:新增自注意力层的参数
    parser.add_argument('--nhead', type=int, default=2, help='the number of heads of multi-head attention')
    parser.add_argument('--layer', type=int, default=1, help='number of SAN layers')
    parser.add_argument('--feedforward', type=int, default=4, help='the multipler of hidden state size')

    parser.add_argument(
       '--feat-drop', type=float, default=0.5, help='the dropout ratio for features'
    )
    parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate')

    parser.add_argument(
     '--batch-size', type=int, default=512, help='the batch size for training'
    )
    parser.add_argument(
        '--epochs', type=int, default=30, help='the number of training epochs'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-4,
        help='the parameter for L2 regularization',
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=5,
        help='the number of epochs that the performance does not improves after which the training stops',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='the number of processes to load the input graphs',
    )
    parser.add_argument(
        '--valid-split',
        type=float,
        default=None,
        help='the fraction for the validation set',
    )
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='print the loss after this number of iterations',
    )
    parser.add_argument(
        '--save-recommendations',
        action='store_true',
        default=True,
        help='whether to save recommendations'
    )
    parser.add_argument(
        '--recommendation-dir',
        type=str,
        default='recommendations',
        help='directory to save recommendations'
    )
    parser.add_argument(
        '--save-best-only',
        action='store_true',
        default=False,
        help='only save recommendations for best model'
    )
    parser.add_argument(
        '--save-metrics',
        action='store_true',
        default=True,
        help='whether to save metrics per epoch'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='directory to save metrics results'
    )
    parser.add_argument(
        '--cutoffs',
        type=str,
        default='1,5,10,20',
        help='valores de cutoff separados por comas (ej: 5,10,20)'
    )
    parser.add_argument(
        '--use-experiment-logger',
        action='store_true',
        default=True,
        help='usar nuevo sistema de registro de experimentos (estilo fpar-CoHHN)'
    )
    parser.add_argument(
        '--config-name',
        type=str,
        default=None,
        help='nombre de la configuración para el directorio de experimentos (si es None, se usará "default")'
    )

    args = parser.parse_args()
    print(args)

    dataset_dir = Path(args.dataset_dir)
    print(dataset_dir)
    # --dataset-dir datasets/diginetica --embedding-dim 32 --num-layers 4
    # --dataset-dir datasets/yoochoose --embedding-dim 32 --num-layers 4
    print('reading dataset')
    train_sessions, test_sessions, num_items = read_dataset(dataset_dir)

    if args.valid_split is not None:
        num_valid = int(len(train_sessions) * args.valid_split)
        test_sessions = train_sessions[-num_valid:]
        train_sessions = train_sessions[:-num_valid]

    # 将一条session划分成多条子session(item_id_seq next_item_label)
    train_set = AugmentedDataset(train_sessions)
    test_set = AugmentedDataset(test_sessions)

    # 获取新商品和类别ID、价格ID对应字典
    if "diginetica" in args.dataset_dir:
        with open("./PILL/utils/data/niid_2_ncid.txt", 'r') as item_category_f:
            # niid_id : n_category_id
            item_category_lines = item_category_f.readlines()
            for each_line in item_category_lines:
                each_id_line_2_list = each_line.split(',')
                category_dict[each_id_line_2_list[0]] = each_id_line_2_list[1].strip()
        with open("./PILL/utils/data/niid_2_priceid.txt", 'r') as item_price_f:
            item_price_lines = item_price_f.readlines()
            for each_line in item_price_lines:
                each_line_2_list = each_line.split(',')
                price_dict[each_line_2_list[0]] = each_line_2_list[1].strip()
        # 获取类别数量
        df = pd.read_csv("./PILL/utils/data/niid_2_ncid.txt", delimiter=',', names=['iid', 'cid'])
        # 获取价格数量
        df_price = pd.read_csv("./PILL/utils/data/niid_2_priceid.txt", delimiter=',', names=['iid', 'pid'])
    elif 'yoochoose1_10' in args.dataset_dir or 'yoochoose1_4' in args.dataset_dir:
        with open("./PILL/utils/data/yoochoose_process/input_data/renew_yoo_niid_2_cid.txt", 'r') as item_category_f:
            # niid_id : n_category_id
            item_category_lines = item_category_f.readlines()
            for each_line in item_category_lines:
                each_id_line_2_list = each_line.split(',')
                category_dict[each_id_line_2_list[0]] = each_id_line_2_list[1].strip()
        with open("./PILL/utils/data/yoochoose_process/input_data/renew_yoo_niid_2_priceid_dispersed_50.txt", 'r') as item_price_f:
            item_price_lines = item_price_f.readlines()
            for each_line in item_price_lines:
                each_line_2_list = each_line.split(',')
                price_dict[each_line_2_list[0]] = each_line_2_list[1].strip()
        # 获取类别数量
        df = pd.read_csv("./PILL/utils/data/yoochoose_process/input_data/renew_yoo_niid_2_cid.txt", delimiter=',', names=['iid', 'cid'])
        # 获取价格数量
        df_price = pd.read_csv("./PILL/utils/data/yoochoose_process/input_data/renew_yoo_niid_2_priceid_dispersed_50.txt", delimiter=',', names=['iid', 'pid'])
    elif 'amazon' in args.dataset_dir:
        with open("./datasets/amazon/helpid.txt", 'r') as item_category_f:
            # niid_id : n_category_id
            item_category_lines = item_category_f.readlines()
            for each_line in item_category_lines:
                each_id_line_2_list = each_line.split(',')
                category_dict[each_id_line_2_list[0]] = each_id_line_2_list[1].strip()
        with open("./datasets/amazon/helpprice.txt", 'r') as item_price_f:
            item_price_lines = item_price_f.readlines()
            for each_line in item_price_lines:
                each_line_2_list = each_line.split(',')
                price_dict[each_line_2_list[0]] = each_line_2_list[1].strip()
        # 获取类别数量
        df = pd.read_csv("./datasets/amazon/helpid.txt", delimiter=',', names=['iid', 'cid'])
        # 获取价格数量
        df_price = pd.read_csv("./datasets/amazon/helpprice.txt", delimiter=',', names=['iid', 'pid'])
    elif 'yelp' in args.dataset_dir:
        with open("./datasets/yelp/yelp_nid_to_category.csv", 'r') as item_category_f:
            item_category_lines = item_category_f.readlines()
            for each_line in item_category_lines:
                each_id_line_2_list = each_line.split(',')
                category_dict[each_id_line_2_list[0]] = each_id_line_2_list[1].strip()
        with open("./datasets/yelp/yelp_nid_to_price.csv", 'r') as item_price_f:
            item_price_lines = item_price_f.readlines()
            for each_line in item_price_lines:
                each_line_2_list = each_line.split(',')
                price_dict[each_line_2_list[0]] = each_line_2_list[1].strip()
        # Get category counts
        df = pd.read_csv("./datasets/yelp/yelp_nid_to_category.csv", delimiter=',', names=['iid', 'cid'])
        # Get price counts
        df_price = pd.read_csv("./datasets/yelp/yelp_nid_to_price.csv", delimiter=',', names=['iid', 'pid'])
    else:
        print("请正确输入数据集文件名称(diginetica/yoochoose1_10/yoochoose1_4)")
        exit(0)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=2,
        collate_fn = get_collate_fn(args.num_layers),
)

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=2,
        collate_fn = get_collate_fn(args.num_layers),
)
    # 获取类别数量
    print(df['cid'].max())
    num_category = df['cid'].max() + 1
    print(num_category)
    # 获取价格数量
    print("price max")
    print(df_price['pid'].max())
    num_price = df_price['pid'].max() + 1
    print(num_price)
    start = time.time()

    model = PILL(args, num_items, num_category, num_price, args.embedding_dim, args.num_layers, feat_drop=args.feat_drop)
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    model = model.to(device)
    print('cuda' if th.cuda.is_available() else 'cpu')
    print(model)
    runner = TrainRunner(
        model,
        train_loader,
        test_loader,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        save_recommendations=args.save_recommendations,
        recommendation_dir=args.recommendation_dir,
        dataset_dir=str(dataset_dir),
        save_metrics=args.save_metrics,
        results_dir=args.results_dir,
        use_experiment_logger=args.use_experiment_logger,
        config_name=args.config_name,
        args=args
    )
    print(args.num_layers)
    print(get_collate_fn(args.num_layers))
    print('start training')
    #metrics = runner.train(args.epochs, args.log_interval)
    # Convertir string de cutoffs a lista de enteros
    cutoffs = [int(x) for x in args.cutoffs.split(',')]
    print(f"Usando valores de cutoff: {cutoffs}")
    
    metrics = runner.train(
        args.epochs,
        args.log_interval,
        save_best=args.save_best_only,
        cutoffs=cutoffs
    )
    
    # Imprimir las métricas finales para cada cutoff
    print('\nResultados Finales:')
    for cutoff in cutoffs:
        print(f'\nMétricas para K@{cutoff}:')
        print(f'MRR@{cutoff}\tHR@{cutoff}\tP@{cutoff}\tRecall@{cutoff}\tNDCG@{cutoff}')
        print(f'{metrics[cutoff]["mrr"] * 100:.3f}%\t{metrics[cutoff]["hr"] * 100:.3f}%\t{metrics[cutoff]["precision"] * 100:.3f}%\t{metrics[cutoff]["recall"] * 100:.3f}%\t{metrics[cutoff]["ndcg"] * 100:.3f}%')
        print(f'MAP@{cutoff}\tF1@{cutoff}\tDiversity@{cutoff}')
        print(f'{metrics[cutoff]["map"] * 100:.3f}%\t{metrics[cutoff]["f1"] * 100:.3f}%\t{metrics[cutoff]["diversity"] * 100:.3f}%')
    end = time.time()
    print("run time: %f s" % (end - start))

if __name__ == '__main__':
    freeze_support()  # This is optional unless you are creating an executable.
    ayno()
