import pandas as pd
import numpy as np
import csv
import math
from pathlib import Path

# Funciones auxiliares para discretización de precios (copiadas de preprocess_diginetica_pasbr_v2.py)
def logistic_custom(t, u, s):
    if s == 0: # Evitar división por cero si la desviación estándar es 0
        return 0.5 if t == u else (1.0 if t > u else 0.0) # Comportamiento sigmoide degenerado
    gama = s * (3**0.5) / math.pi
    if gama == 0: # Evitar división por cero si gama resulta ser 0
         return 0.5 if t == u else (1.0 if t > u else 0.0)
    try:
        exponent = (t - u) / gama
        if exponent > 700:
            return 0.0
        elif exponent < -700:
            return 1.0
        return 1 / (1 + math.exp(exponent))
    except OverflowError:
        return 0.0 if (t-u)/gama > 0 else 1.0

def get_price_level_custom(price, p_min, p_max, mean, std, n_ranges):
    if std == 0:
        return -1
    log_price = logistic_custom(price, mean, std)
    log_p_min = logistic_custom(p_min, mean, std)
    log_p_max = logistic_custom(p_max, mean, std)
    fenmu = log_p_max - log_p_min
    if fenmu == 0:
        return -1
    # if price == 0:
    #     return -1
    fenzi = log_price - log_p_min
    level = int((fenzi / fenmu) * n_ranges)
    if level < 0:
        level = 0
    elif level >= n_ranges:
        level = n_ranges - 1
    return level + 1

def _discretize_prices_diginetica(df_filtered_items, dataset_dir_for_raw_data, discretization_method_param, num_price_ranges_param):
    """
    df_filtered_items: DataFrame con columna 'itemId' (OIDs originales)
    dataset_dir_for_raw_data: Path al directorio datasets/diginetica
    discretization_method_param: str, método de discretización
    num_price_ranges_param: int, número de rangos
    """
    import pandas as pd
    products_csv_path = Path(dataset_dir_for_raw_data) / 'dataset' / 'products.csv'
    categories_csv_path = Path(dataset_dir_for_raw_data) / 'dataset' / 'product-categories.csv'

    # Cargar productos
    products_df = pd.read_csv(products_csv_path, delimiter=';')
    products_df['pricelog2'] = pd.to_numeric(products_df['pricelog2'], errors='coerce')
    products_df = products_df.dropna(subset=['pricelog2'])
    products_df['itemId'] = products_df['itemId'].astype(int)

    # Filtrar productos por OIDs presentes en el df filtrado
    oids = df_filtered_items['itemId'].unique()
    products_df = products_df[products_df['itemId'].isin(oids)]

    # Si se requiere, cargar categorías
    if discretization_method_param == 'custom_logistic_per_category':
        categories_df = pd.read_csv(categories_csv_path, delimiter=';')
        categories_df['itemId'] = categories_df['itemId'].astype(int)
        item_data_full = pd.merge(products_df, categories_df, on='itemId', how='left')
        # Rellenar NaN en categoryId con 'no category' para evitar filtrado
        item_data_full['categoryId'] = item_data_full['categoryId'].fillna('no category')
    else:
        item_data_full = products_df.copy()

    all_prices = item_data_full['pricelog2'].values

    thresholds = None
    category_price_stats = None

    if discretization_method_param == 'quantiles':
        thresholds = np.quantile(all_prices, np.linspace(0, 1, num_price_ranges_param + 1))
    elif discretization_method_param == 'equal_width':
        min_price, max_price = np.min(all_prices), np.max(all_prices)
        thresholds = np.linspace(min_price, max_price, num_price_ranges_param + 1)
    elif discretization_method_param == 'logistic':
        mean, std = np.mean(all_prices), np.std(all_prices)
    elif discretization_method_param == 'custom_logistic':
        mean, std = np.mean(all_prices), np.std(all_prices)
    elif discretization_method_param == 'custom_logistic_per_category':
        # Calcular media y std por categoría
        category_price_stats = {}
        #print("Columnas disponibles en item_data_full:", item_data_full.columns)
        #print("Primeras filas de item_data_full:")
        #print(item_data_full.head())
        for cat_id, group in item_data_full.groupby('categoryId'):
            prices_cat = group['pricelog2'].values
            if len(prices_cat) > 0:
                category_price_stats[cat_id] = {
                    'mean': np.mean(prices_cat),
                    'std': np.std(prices_cat),
                    'min': np.min(prices_cat),
                    'max': np.max(prices_cat)
                }

    oid_to_price_level_map = {}
    for oid in oids:
        item_info = item_data_full[item_data_full['itemId'] == oid]
        if item_info.empty:
            oid_to_price_level_map[oid] = -1
            continue
        price_val = item_info['pricelog2'].iloc[0]
        if discretization_method_param == 'quantiles' or discretization_method_param == 'equal_width':
            # Buscar el rango correspondiente
            price_level = -1
            for i in range(len(thresholds) - 1):
                if thresholds[i] <= price_val < thresholds[i + 1]:
                    price_level = i + 1
                    break
            if price_val == thresholds[-1]:
                price_level = len(thresholds) - 1
            oid_to_price_level_map[oid] = price_level
        elif discretization_method_param == 'logistic':
            mean, std = np.mean(all_prices), np.std(all_prices)
            p_min, p_max = np.min(all_prices), np.max(all_prices)
            price_level = get_price_level_custom(price_val, p_min, p_max, mean, std, num_price_ranges_param)
            oid_to_price_level_map[oid] = price_level
        elif discretization_method_param == 'custom_logistic':
            mean, std = np.mean(all_prices), np.std(all_prices)
            p_min, p_max = np.min(all_prices), np.max(all_prices)
            price_level = get_price_level_custom(price_val, p_min, p_max, mean, std, num_price_ranges_param)
            oid_to_price_level_map[oid] = price_level
        elif discretization_method_param == 'custom_logistic_per_category':
            cat_id = item_info['categoryId'].iloc[0]
            stats = category_price_stats.get(cat_id)
            if stats is None:
                oid_to_price_level_map[oid] = -1
                continue
            price_level = get_price_level_custom(
                price_val,
                stats['min'],
                stats['max'],
                stats['mean'],
                stats['std'],
                num_price_ranges_param
            )
            oid_to_price_level_map[oid] = price_level
        else:
            oid_to_price_level_map[oid] = -1
    return oid_to_price_level_map

def get_session_id(df, interval):
    df_prev = df.shift()
    is_new_session = (df.userId != df_prev.userId) | (
        df.timestamp - df_prev.timestamp > interval
    )
    session_id = is_new_session.cumsum() - 1
    return session_id.astype(int)

def group_sessions(df, interval):
    sessionId = get_session_id(df, interval)
    df = df.assign(sessionId=sessionId.astype(int))
    return df

def filter_short_sessions(df, min_len=2):
    session_len = df.groupby('sessionId', sort=False).size()
    long_sessions = session_len[session_len >= min_len].index
    df_long = df[df.sessionId.isin(long_sessions)]
    return df_long

def filter_infreq_items(df, min_support=5):
    item_support = df.groupby('itemId', sort=False).size()
    freq_items = item_support[item_support >= min_support].index
    df_freq = df[df.itemId.isin(freq_items)]
    return df_freq

def filter_until_all_long_and_freq(df, min_len=2, min_support=5):
    while True:
        df_long = filter_short_sessions(df, min_len)
        df_freq = filter_infreq_items(df_long, min_support)
        if len(df_freq) == len(df):
            break
        df = df_freq
    return df

def truncate_long_sessions(df, max_len=20, is_sorted=False):
    if not is_sorted:
        df = df.sort_values(['sessionId', 'timestamp'])
    itemIdx = df.groupby('sessionId').cumcount()
    df_t = df[itemIdx < max_len]
    return df_t

def update_id(df, field):
    labels = pd.factorize(df[field])[0]
    kwargs = {field: labels.astype(int)}
    df = df.assign(**kwargs)
    return df

def remove_immediate_repeats(df):
    df_prev = df.shift()
    is_not_repeat = (df.sessionId != df_prev.sessionId) | (df.itemId != df_prev.itemId)
    df_no_repeat = df[is_not_repeat]
    return df_no_repeat

def reorder_sessions_by_endtime(df):
    endtime = df.groupby('sessionId', sort=False).timestamp.max()
    df_endtime = endtime.sort_values().reset_index()
    # Ensure session IDs are integers
    df_endtime['sessionId'] = df_endtime['sessionId'].astype(int)
    oid2nid = {int(oid): int(nid) for oid, nid in zip(df_endtime.sessionId, df_endtime.index)}
    sessionId_new = df.sessionId.map(oid2nid)
    df = df.assign(sessionId=sessionId_new.astype(int))
    df = df.sort_values(['sessionId', 'timestamp'])
    return df

def keep_top_n_items(df, n):
    item_support = df.groupby('itemId', sort=False).size()
    top_items = item_support.nlargest(n).index
    df_top = df[df.itemId.isin(top_items)]
    return df_top

def split_by_time(df, timedelta):
    max_time = df.timestamp.max()
    end_time = df.groupby('sessionId').timestamp.max()
    split_time = max_time - timedelta
    train_sids = end_time[end_time < split_time].index
    df_train = df[df.sessionId.isin(train_sids)]
    df_test = df[~df.sessionId.isin(train_sids)]
    return df_train, df_test

def train_test_split(df, test_split=0.2):
    endtime = df.groupby('sessionId', sort=False).timestamp.max()
    endtime = endtime.sort_values()
    num_tests = int(len(endtime) * test_split)
    test_session_ids = endtime.index[-num_tests:]
    df_train = df[~df.sessionId.isin(test_session_ids)]
    df_test = df[df.sessionId.isin(test_session_ids)]
    return df_train, df_test

def save_sessions(df, filepath):
    df = reorder_sessions_by_endtime(df)
    # Ensure itemId is integer before saving
    df['itemId'] = df['itemId'].astype(int)
    sessions = df.groupby('sessionId').itemId.apply(lambda x: ','.join(map(lambda i: str(int(i)), x)))
    sessions.to_csv(filepath, sep='\t', header=False, index=False)

def save_dataset(dataset_dir, df_train, df_test, oid_to_price_level_map):
    # filter items in test but not in train
    df_test = df_test[df_test.itemId.isin(df_train.itemId.unique())]
    df_test = filter_short_sessions(df_test)

    print(f'No. of Clicks: {len(df_train) + len(df_test)}')
    print(f'No. of Items: {df_train.itemId.nunique()}')

    # update itemId and ensure integer types
    train_itemId_new, uniques = pd.factorize(df_train.itemId)
    df_train = df_train.assign(itemId=train_itemId_new.astype(int))
    oid2nid = {int(oid): int(i) for i, oid in enumerate(uniques)}

    # Guardar mapeo NID a price_level
    niid_priceid_filepath = Path('PILL') / 'utils' / 'data' / 'niid_2_priceid.txt'
    niid_priceid_filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(niid_priceid_filepath, 'w', newline='') as f_price:
        writer_price = csv.writer(f_price)
        for oid_original, nid_final in oid2nid.items():
            price_level = oid_to_price_level_map.get(oid_original)
            if price_level is not None and price_level != -1:
                writer_price.writerow([nid_final, price_level])
    print(f"Generated price mapping: {niid_priceid_filepath}")

    # 将新旧ID的对应关系字典保存成oid2nid.csv
    with open(dataset_dir / "oid2nid.csv", 'w') as csv_f:
        writer = csv.writer(csv_f)
        for key, value in oid2nid.items():
            writer.writerow([key, value])
    df_test['itemId'] = df_test['itemId'].astype(int)
    test_itemId_new = df_test.itemId.map(oid2nid)
    df_test = df_test.assign(itemId=test_itemId_new.astype(int))

    print(f'saving dataset to {dataset_dir}')
    dataset_dir.mkdir(parents=True, exist_ok=True)
    save_sessions(df_train, dataset_dir / 'train.txt')
    save_sessions(df_test, dataset_dir / 'test.txt')
    num_items = len(uniques)
    with open(dataset_dir / 'num_items.txt', 'w') as f:
        f.write(str(num_items))

def preprocess_diginetica(dataset_dir, csv_file, discretization_method, num_price_ranges):
    print(f'reading {csv_file}...')
    df = pd.read_csv(
        csv_file,
        usecols=[0, 2, 3, 4],
        delimiter=';',
        parse_dates=['eventdate'],
        dtype={'sessionId': int, 'itemId': int}  # Ensure integer types on load
    )
    print('start preprocessing')
    # timeframe (time since the first query in a session, in milliseconds)
    df['timestamp'] = pd.to_timedelta(df.timeframe, unit='ms') + df.eventdate
    df = df.drop(columns=['eventdate', 'timeframe'])
    df = df.sort_values(['sessionId', 'timestamp'])
    df = filter_short_sessions(df)
    df = truncate_long_sessions(df, is_sorted=True)
    df = filter_infreq_items(df)
    df = filter_short_sessions(df)

    # Discretización de precios antes de split
    oid_to_price_level_map = _discretize_prices_diginetica(
        df.copy(), dataset_dir, discretization_method, num_price_ranges
    )

    df_train, df_test = split_by_time(df, pd.Timedelta(days=7))
    save_dataset(dataset_dir, df_train, df_test, oid_to_price_level_map)

def save_yelp_mappings(dataset_dir, oid_to_nid, nid_to_category, nid_to_price):
    """Saves the Yelp mapping dictionaries to CSV files."""
    with open(dataset_dir / "yelp_oid_to_nid.csv", 'w') as csv_f:
        writer = csv.writer(csv_f)
        for key, value in oid_to_nid.items():
            writer.writerow([key, value])

    with open(dataset_dir / "yelp_nid_to_category.csv", 'w') as csv_f:
        writer = csv.writer(csv_f)
        for key, value in nid_to_category.items():
            writer.writerow([key, value])

    with open(dataset_dir / "yelp_nid_to_price.csv", 'w') as csv_f:
        writer = csv.writer(csv_f)
        for key, value in nid_to_price.items():
            writer.writerow([key, value])

def preprocess_yelp(dataset_dir, csv_file):
    print(f'reading {csv_file}...')
    df = pd.read_csv(
        csv_file,
        usecols=[0, 1, 2, 3, 4, 5],  # Include price and categories columns
        delimiter=';',
        parse_dates=['eventdate'],
        infer_datetime_format=True,
    )
    print('start preprocessing')
    # Ensure itemId is integer
    df['itemId'] = df['itemId'].astype(int)
    # timestamp (time of the review)
    df['timestamp'] = pd.to_timedelta(df['timeframe'], unit='ms')
    df = df.drop(['eventdate', 'timeframe'], axis=1)
    df = df.sort_values(['sessionId', 'timestamp'])
    df = filter_short_sessions(df)
    df = truncate_long_sessions(df, is_sorted=True)
    df = filter_infreq_items(df)
    df = filter_short_sessions(df)
    df = df.dropna()
    df_train, df_test = train_test_split(df, test_split=0.2)
    #print(df_train.head(10))
    # update itemId and create mappings
    train_itemId_new, uniques = pd.factorize(df_train.itemId)
    df_train = df_train.assign(itemId=train_itemId_new.astype(int))
    oid_to_nid = {int(oid): int(i) for i, oid in enumerate(uniques)}

    # Create category and price mappings
    original_items = df_train.itemId.unique()
    nid_to_category = {}
    nid_to_price = {}
    
    # Create a temporary mapping of original IDs to categories and prices
    temp_df = df[['itemId', 'categories', 'price']].drop_duplicates()
    
    # Map new IDs to categories and prices
    for oid, nid in oid_to_nid.items():
        item_data = temp_df[temp_df.itemId == oid].iloc[0]
        nid_to_category[nid] = item_data['categories']
        nid_to_price[nid] = item_data['price']

    save_yelp_mappings(dataset_dir, oid_to_nid, nid_to_category, nid_to_price)
    
    # 将新旧ID的对应关系字典保存成oid2nid.csv
    with open(dataset_dir / "oid2nid.csv", 'w') as csv_f:
        writer = csv.writer(csv_f)
        for key, value in oid_to_nid.items():
            writer.writerow([key, value])
    # Ensure itemId remains integer throughout the pipeline and handle NA/inf values
    df_test = df_test[df_test['itemId'].notna() & ~df_test['itemId'].isin([np.inf, -np.inf])]
    df_test['itemId'] = df_test['itemId'].astype(float).astype(int)
    
    # Handle missing mappings
    valid_items = df_test['itemId'].isin(oid_to_nid.keys())
    df_test = df_test[valid_items]
    
    test_itemId_new = df_test.itemId.map(oid_to_nid)
    # Verify no NaN values were introduced by the mapping
    df_test = df_test[test_itemId_new.notna()]
    test_itemId_new = test_itemId_new[test_itemId_new.notna()]
    df_test = df_test.assign(itemId=test_itemId_new.astype(int))
    # print(df_test.head(10))
    print(f'saving dataset to {dataset_dir}')
    dataset_dir.mkdir(parents=True, exist_ok=True)
    save_sessions(df_train, dataset_dir / 'train.txt')
    save_sessions(df_test, dataset_dir / 'test.txt')
    num_items = len(uniques)
    with open(dataset_dir / 'num_items.txt', 'w') as f:
        f.write(str(num_items))

def preprocess_gowalla_lastfm(dataset_dir, csv_file, usecols, interval, n):
    print(f'reading {csv_file}...')
    df = pd.read_csv(
        csv_file,
        sep='\t',
        header=None,
        names=['userId', 'timestamp', 'itemId'],
        usecols=usecols,
        parse_dates=['timestamp'],
        infer_datetime_format=True,
        dtype={'userId': int, 'itemId': int}  # Ensure integer types on load
    )
    print('start preprocessing')
    df = df.dropna()
    df = update_id(df, 'userId')
    df = update_id(df, 'itemId')
    df = df.sort_values(['userId', 'timestamp'])

    df = group_sessions(df, interval)
    df = remove_immediate_repeats(df)
    df = truncate_long_sessions(df, is_sorted=True)
    df = keep_top_n_items(df, n)
    df = filter_until_all_long_and_freq(df)
    df_train, df_test = train_test_split(df, test_split=0.2)
    save_dataset(dataset_dir, df_train, df_test)
