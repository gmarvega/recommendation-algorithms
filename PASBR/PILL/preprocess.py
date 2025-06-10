from pathlib import Path
import argparse
import sys as system

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

optional = parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
required.add_argument(
    '-d',
    '--dataset',
    choices=['diginetica', 'gowalla', 'lastfm', 'yelp'],
    required=True,
    help='the dataset name',
)
required.add_argument(
    '-f',
    '--filepath',
    required=True,
    help='the file for the dataset, i.e., "train-item-views.csv" for diginetica, '
    '"loc-gowalla_totalCheckins.txt" for gowalla, '
    '"userid-timestamp-artid-artname-traid-traname.tsv" for lastfm',
)
optional.add_argument(
    '-t',
    '--dataset-dir',
    default='datasets/{dataset}',
    help='the folder to save the preprocessed dataset',
)
optional.add_argument(
    '--discretization_method',
    choices=['quantiles', 'equal_width', 'logistic', 'custom_logistic', 'custom_logistic_per_category'],
    default='quantiles',
    help="Método de discretización de precios a utilizar cuando el dataset es 'diginetica'."
)
optional.add_argument(
    '--num_price_ranges',
    type=int,
    default=50,
    help="Número de rangos de precios para la discretización cuando el dataset es 'diginetica'."
)
parser._action_groups.append(optional)
args = parser.parse_args()

dataset_dir = Path(args.dataset_dir.format(dataset=args.dataset))

if args.dataset == 'diginetica':
    from utils.data.preprocess import preprocess_diginetica
    preprocess_diginetica(
        dataset_dir,
        args.filepath,
        args.discretization_method,
        args.num_price_ranges
    )
elif args.dataset == 'yelp':
    from utils.data.preprocess import preprocess_yelp
    preprocess_yelp(dataset_dir, args.filepath)
    #print("Yelp")
    #system.exit(0)
else:
    from pandas import Timedelta
    from utils.data.preprocess import preprocess_gowalla_lastfm

    csv_file = args.filepath
    if args.dataset == 'gowalla':
        usecols = [0, 1, 4]
        interval = Timedelta(days=1)
        n = 30000
    else:
        usecols = [0, 1, 2]
        interval = Timedelta(hours=8)
        n = 40000
    preprocess_gowalla_lastfm(dataset_dir, csv_file, usecols, interval, n)
