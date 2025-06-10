# -*- coding: utf-8 -*-
"""
===============================================================================
Script: preprocess_amazon_pasbr_equivalent.py
===============================================================================

Este script adapta la lógica de CoHHNpreprocess-Amazon_withMaps.py para generar
archivos de salida estructuralmente equivalentes a los producidos por
preprocess_diginetica_pasbr_v2.py, pero usando datos de Amazon.

Entradas:
    - <datasets_name>.json.gz: Interacciones usuario-ítem.
    - meta_<datasets_name>.json.gz: Metadatos de productos.

Salidas:
    - Archivos en output/Amazon/<datasets_name>/... (ver plan detallado).

Configuración:
    - Modifica las variables 'datasets_name' y 'price_level_num' al inicio.

===============================================================================
"""

import os
import gzip
import json
import math
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime

# =======================
# CONFIGURACIÓN PRINCIPAL
# =======================
datasets_name = 'Grocery_and_Gourmet_Food'  # Cambia por el dataset de Amazon deseado
price_level_num = 2                         # Número de niveles de precio (ajusta según necesidad)
max_seq_len = 20                            # Longitud máxima de secuencia por sesión
min_session_length = 2                      # Mínimo de interacciones por sesión
min_item_interactions = 1                   # Mínimo de interacciones por ítem (ajusta si quieres filtrar ítems raros)
train_split_ratio = 0.9                     # Proporción de sesiones para train

# Ruta a los datos crudos
originalData = os.path.join('../datasets', 'amazon')
data_path_interactions = os.path.join(originalData, f'{datasets_name}.json.gz')
data_path_meta = os.path.join(originalData, f'meta_{datasets_name}.json.gz')

# =======================
# FUNCIONES AUXILIARES
# =======================

def parse_gz_json(path):
    with gzip.open(path, 'rb') as g:
        for l in g:
            yield json.loads(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse_gz_json(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def reg_price(price):
    price_num_str = str(price).replace('$','').replace(',','')
    if is_number(price_num_str):
        return float(price_num_str)
    return ''

def reg_category(cate):
    if isinstance(cate, list) and len(cate) > 0:
        return cate
    return []

def get_last_cate(cate_list):
    if cate_list:
        return cate_list[-1]
    return ''

def logistic(t, u, s):
    if s == 0:
        return 0.5
    gama = s * (3**0.5) / math.pi
    if gama == 0:
        return 0.5
    try:
        exponent = (t-u)/gama
        if exponent > 700: return 0.0
        if exponent < -700: return 1.0
        return 1 / (1 + math.exp(exponent))
    except OverflowError:
        return 0.0 if (t-u)/gama > 0 else 1.0

def get_price_level(price, p_min, p_max, mean, std, num_levels):
    if std == 0 or p_min == p_max:
        return int(num_levels / 2) + 1 if num_levels > 0 else 1
    log_price = logistic(price, mean, std)
    log_p_min = logistic(p_min, mean, std)
    log_p_max = logistic(p_max, mean, std)
    fenmu = log_p_max - log_p_min
    if fenmu == 0:
        if price <= p_min: return 1
        if price >= p_max: return num_levels
        return int(num_levels / 2) + 1
    if price < p_min: return 1
    if price > p_max: return num_levels
    fenzi = log_price - log_p_min
    level = int((fenzi / fenmu) * num_levels)
    level = max(0, min(level, num_levels - 1))
    return level + 1

def format_for_pkl(item_seqs, cat_seqs, price_seqs):
    combined = []
    for i_s, c_s, p_s in zip(item_seqs, cat_seqs, price_seqs):
        combined.append((list(map(str, i_s)), list(map(str, c_s)), list(map(str, p_s))))
    return combined

def save_structured_sessions(filename, session_ids, item_seqs, price_seqs, cat_seqs, path_dir):
    df_data = []
    for i, (s_id, items, prices, cats) in enumerate(zip(session_ids, item_seqs, price_seqs, cat_seqs)):
        df_data.append({
            'session_id': s_id,
            'item_sequence': ','.join(map(str,items)),
            'price_level_sequence': ','.join(map(str,prices)),
            'category_sequence': ','.join(map(str,cats))
        })
    df = pd.DataFrame(df_data)
    df.to_csv(os.path.join(path_dir, filename), index=False)

# =======================
# 1. CARGA Y PREPROCESAMIENTO BASE
# =======================

print(f'Leyendo interacciones de {data_path_interactions} ...')
df_interactions = getDF(data_path_interactions)
interaction = df_interactions[['reviewerID', 'asin', 'unixReviewTime']]

print(f'Leyendo metadatos de {data_path_meta} ...')
df_item_meta = getDF(data_path_meta)
item_property = df_item_meta[['asin', 'price', 'category', 'also_buy', 'also_view']]

item_property['price_num'] = item_property['price'].apply(reg_price)
item_property['category_list'] = item_property['category'].apply(reg_category)

# Filtrar ítems sin precio o sin categoría
item_property = item_property[item_property['price_num'] != '']
item_property['price_num'] = item_property['price_num'].astype(float)
item_property = item_property[item_property['category_list'].apply(len) > 0]
item_property.drop_duplicates(subset=['asin'],keep='first',inplace=True)

item_property['cate_final'] = item_property['category_list'].apply(get_last_cate)
item_property = item_property[item_property['cate_final'] != '']

item_all_features = item_property[['asin', 'price_num', 'cate_final', 'also_buy', 'also_view']]

# Estadísticas de precios por categoría
group_cate_stats = item_all_features.groupby('cate_final')['price_num'].agg(['count', 'min', 'max', 'mean', 'std']).reset_index()
item_data_merged = pd.merge(item_all_features, group_cate_stats, how='left', on = 'cate_final')
item_data_merged.fillna({'std': 0}, inplace=True)

# Filtrar ítems con pocas instancias por categoría o std=0
item_data_filtered = item_data_merged[(item_data_merged['count'] > 9) & (item_data_merged['std'] != 0)]

# Calcular price_level
item_data_filtered['price_level_calculated'] = item_data_filtered.apply(
    lambda row: get_price_level(row['price_num'], row['min'], row['max'], row['mean'], row['std'], price_level_num), axis=1
)

item_final_processed = item_data_filtered[item_data_filtered['price_level_calculated'] != -1]
item_final_processed = item_final_processed[['asin', 'price_num', 'cate_final', 'price_level_calculated']]

# Unir interacciones con ítems procesados
user_item_interactions = pd.merge(interaction, item_final_processed, how='inner', on = 'asin')
user_item_interactions.sort_values(by=["reviewerID","unixReviewTime"],inplace=True,ascending=[True,True])

# Filtrar sesiones cortas
user_click_counts = user_item_interactions.groupby('reviewerID')['asin'].count().reset_index(name='click_num')
user_item_interactions = pd.merge(user_item_interactions, user_click_counts, how='left', on='reviewerID')
user_item_interactions_filtered = user_item_interactions[user_item_interactions['click_num'] >= min_session_length]

data_all_sessions = user_item_interactions_filtered

# =======================
# 2. MAPEOS DE IDS Y FILTRADO FINAL
# =======================

# OID (asin) a NID (numeric_itemID)
unique_asins = data_all_sessions['asin'].unique()
oid_to_nid = {asin: i+1 for i, asin in enumerate(unique_asins)}
nid_to_oid = {i+1: asin for i, asin in enumerate(unique_asins)}
num_items = len(unique_asins)

# Original Category String (cate_final) a NCID (numeric_categoryID)
unique_categories = data_all_sessions['cate_final'].unique()
ocid_to_ncid = {cat_str: i+1 for i, cat_str in enumerate(unique_categories)}
ncid_to_ocid = {i+1: cat_str for i, cat_str in enumerate(unique_categories)}
num_categories = len(unique_categories)

# Price Level (price_level_calculated) a PID (price_levelID)
data_all_sessions.rename(columns={'price_level_calculated': 'pid'}, inplace=True)
num_price_levels = price_level_num

# Aplicar mapeos a las sesiones
data_all_sessions['nid'] = data_all_sessions['asin'].map(oid_to_nid)
data_all_sessions['ncid'] = data_all_sessions['cate_final'].map(ocid_to_ncid)
# 'pid' ya está presente

# Mapeo de reviewerID a sessionID_numeric
unique_reviewerIDs = data_all_sessions['reviewerID'].unique()
reviewerID_to_sessionID_numeric = {revID: i+1 for i, revID in enumerate(unique_reviewerIDs)}
data_all_sessions['sessionID_numeric'] = data_all_sessions['reviewerID'].map(reviewerID_to_sessionID_numeric)

# =======================
# 3. GENERACIÓN DE SECUENCIAS Y SPLIT TRAIN/TEST
# =======================

session_groups = data_all_sessions.groupby('sessionID_numeric')

all_item_seqs = []
all_cat_seqs = []
all_price_seqs = []
session_ids_list = []

for session_id, group in session_groups:
    group = group.sort_values(by='unixReviewTime')
    item_seq = group['nid'].astype(str).tolist()
    cat_seq = group['ncid'].astype(str).tolist()
    price_seq = group['pid'].astype(str).tolist()
    if len(item_seq) < min_session_length:
        continue
    all_item_seqs.append(item_seq[:max_seq_len])
    all_cat_seqs.append(cat_seq[:max_seq_len])
    all_price_seqs.append(price_seq[:max_seq_len])
    session_ids_list.append(session_id)

num_sessions = len(session_ids_list)
split_point = int(num_sessions * train_split_ratio)

train_item_seqs = all_item_seqs[:split_point]
train_cat_seqs = all_cat_seqs[:split_point]
train_price_seqs = all_price_seqs[:split_point]
train_session_ids = session_ids_list[:split_point]

test_item_seqs = all_item_seqs[split_point:]
test_cat_seqs = all_cat_seqs[split_point:]
test_price_seqs = all_price_seqs[split_point:]
test_session_ids = session_ids_list[split_point:]

# =======================
# 4. DIRECTORIOS DE SALIDA
# =======================

BASE_DIR = os.getcwd()
PILL_DIR = os.path.join(BASE_DIR, 'PILL')
PILL_UTILS_DATA_DIR = os.path.join(PILL_DIR, 'utils', 'data')

AMAZON_PROCESSED_DIR = os.path.join(BASE_DIR, 'datasets', 'amazon_processed', datasets_name)
FINAL_STRUCTURED_DIR_AMAZON = os.path.join(BASE_DIR, f'amazon_pasbr_pricelevel_{price_level_num}_{datasets_name}')

os.makedirs(AMAZON_PROCESSED_DIR, exist_ok=True)
os.makedirs(PILL_UTILS_DATA_DIR, exist_ok=True)
os.makedirs(FINAL_STRUCTURED_DIR_AMAZON, exist_ok=True)

# =======================
# 5. GUARDADO DE ARCHIVOS
# =======================

# --- a) processed_base/ ---
# oid2nid.csv
oid_nid_df = pd.DataFrame(list(oid_to_nid.items()), columns=['original_itemID', 'numeric_itemID'])
oid_nid_path = os.path.join(AMAZON_PROCESSED_DIR, 'oid2nid.csv')
oid_nid_df.to_csv(oid_nid_path, index=False)
print(f"  Guardado: {oid_nid_path}")

# num_items.txt
num_items_path = os.path.join(AMAZON_PROCESSED_DIR, 'num_items.txt')
with open(num_items_path, 'w') as f:
    f.write(str(num_items))
print(f"  Guardado: {num_items_path}")

# train.txt
train_txt_path = os.path.join(AMAZON_PROCESSED_DIR, 'train.txt')
with open(train_txt_path, 'w') as f:
    for seq in train_item_seqs:
        f.write(','.join(seq) + '\n')
print(f"  Guardado: {train_txt_path}")

# test.txt
test_txt_path = os.path.join(AMAZON_PROCESSED_DIR, 'test.txt')
with open(test_txt_path, 'w') as f:
    for seq in test_item_seqs:
        f.write(','.join(seq) + '\n')
print(f"  Guardado: {test_txt_path}")

# --- b) intermediate_mappings/ ---
# oid2nid_category.csv
oid2nid_category_path = os.path.join(PILL_UTILS_DATA_DIR, f'oid2nid_category_amazon_{datasets_name}.csv')
ocid_ncid_df = pd.DataFrame(list(ocid_to_ncid.items()), columns=['original_categoryID', 'numeric_categoryID'])
ocid_ncid_df.to_csv(oid2nid_category_path, index=False)
print(f"  Guardado: {oid2nid_category_path}")

# new_oid2nid_category.csv (solo categorías activas)
new_oid2nid_category_path = os.path.join(PILL_UTILS_DATA_DIR, f'new_oid2nid_category_amazon_{datasets_name}.csv')
active_ncids = data_all_sessions['ncid'].unique()
active_ocid_ncid_df = ocid_ncid_df[ocid_ncid_df['numeric_categoryID'].isin(active_ncids)]
active_ocid_ncid_df.to_csv(new_oid2nid_category_path, index=False)
print(f"  Guardado: {new_oid2nid_category_path}")

# niid_2_ncid.txt y niid_2_priceid.txt
item_props_for_map = item_final_processed[['asin', 'cate_final', 'price_level_calculated']].copy()
item_props_for_map['nid'] = item_props_for_map['asin'].map(oid_to_nid)
item_props_for_map['ncid'] = item_props_for_map['cate_final'].map(ocid_to_ncid)
item_props_for_map.rename(columns={'price_level_calculated': 'pid'}, inplace=True)
item_props_for_map.dropna(subset=['nid'], inplace=True)

nid_to_ncid_map_df = item_props_for_map[['nid', 'ncid']].drop_duplicates().sort_values(by='nid')
nid_to_pid_map_df = item_props_for_map[['nid', 'pid']].drop_duplicates().sort_values(by='nid')

niid_2_ncid_path = os.path.join(PILL_UTILS_DATA_DIR, f'niid_2_ncid_amazon_{datasets_name}.txt')
nid_to_ncid_map_df.to_csv(niid_2_ncid_path, header=False, index=False)
print(f"  Guardado: {niid_2_ncid_path}")

niid_2_priceid_path = os.path.join(PILL_UTILS_DATA_DIR, f'niid_2_priceid_amazon_{datasets_name}.txt')
nid_to_pid_map_df.to_csv(niid_2_priceid_path, header=False, index=False)
print(f"  Guardado: {niid_2_priceid_path}")

# --- c) amazon_pasbr_pricelevel_{price_level_num}/ ---
# train.pkl & test.pkl
train_data_pkl = format_for_pkl(train_item_seqs, train_cat_seqs, train_price_seqs)
test_data_pkl = format_for_pkl(test_item_seqs, test_cat_seqs, test_price_seqs)

train_pkl_path = os.path.join(FINAL_STRUCTURED_DIR_AMAZON, 'train.pkl')
with open(train_pkl_path, 'wb') as f:
    pickle.dump(train_data_pkl, f)
print(f"  Guardado: {train_pkl_path}")

test_pkl_path = os.path.join(FINAL_STRUCTURED_DIR_AMAZON, 'test.pkl')
with open(test_pkl_path, 'wb') as f:
    pickle.dump(test_data_pkl, f)
print(f"  Guardado: {test_pkl_path}")

# mapping_dicts.pkl
nid_to_ncid_dict = pd.Series(nid_to_ncid_map_df.ncid.values, index=nid_to_ncid_map_df.nid).to_dict()
nid_to_pid_dict = pd.Series(nid_to_pid_map_df.pid.values, index=nid_to_pid_map_df.nid).to_dict()

mapping_dicts_path = os.path.join(FINAL_STRUCTURED_DIR_AMAZON, 'mapping_dicts.pkl')
mapping_content = {
    'nid_to_oid': nid_to_oid,
    'oid_to_nid': oid_to_nid,
    'nid_to_ncid': nid_to_ncid_dict,
    'nid_to_pid': nid_to_pid_dict,
    'ncid_to_ocid': ncid_to_ocid,
    'ocid_to_ncid': ocid_to_ncid,
    'num_items_numeric': num_items,
    'num_categories_numeric': num_categories,
    'num_price_levels': num_price_levels,
    'discretization_method_used': "logistic_amazon",
    'num_price_ranges_config': price_level_num
}
with open(mapping_dicts_path, 'wb') as f:
    pickle.dump(mapping_content, f)
print(f"  Guardado: {mapping_dicts_path}")

# item_mapping_table_with_sequences.csv
item_mapping_table_path = os.path.join(FINAL_STRUCTURED_DIR_AMAZON, 'item_mapping_table_with_sequences.csv')
item_table_df = item_final_processed[['asin', 'price_num']].copy()
item_table_df.rename(columns={'asin': 'original_itemID', 'price_num': 'original_price'}, inplace=True)
item_table_df['numeric_itemID'] = item_table_df['original_itemID'].map(oid_to_nid)
item_table_df = pd.merge(item_table_df, nid_to_ncid_map_df.rename(columns={'nid':'numeric_itemID', 'ncid':'numeric_categoryID'}), on='numeric_itemID', how='left')
item_table_df = pd.merge(item_table_df, nid_to_pid_map_df.rename(columns={'nid':'numeric_itemID', 'pid':'price_levelID'}), on='numeric_itemID', how='left')
item_table_df = pd.merge(item_table_df, ocid_ncid_df.rename(columns={'numeric_categoryID':'numeric_categoryID_temp', 'original_categoryID':'original_categoryID_val'}), left_on='numeric_categoryID', right_on='numeric_categoryID_temp', how='left')
item_table_df.rename(columns={'original_categoryID_val': 'original_categoryID'}, inplace=True)
item_table_df.drop(columns=['numeric_categoryID_temp'], inplace=True, errors='ignore')
item_table_df = item_table_df[['numeric_itemID', 'original_itemID', 'numeric_categoryID', 'original_categoryID', 'price_levelID', 'original_price']]
item_table_df.dropna(subset=['numeric_itemID'], inplace=True)
item_table_df = item_table_df.astype({'numeric_itemID': int, 'numeric_categoryID': 'Int64', 'price_levelID': 'Int64'})
item_table_df.sort_values(by='numeric_itemID', inplace=True)
item_table_df.to_csv(item_mapping_table_path, index=False)
print(f"  Guardado: {item_mapping_table_path}")

# categoryID_to_category_mapping.csv
category_mapping_path = os.path.join(FINAL_STRUCTURED_DIR_AMAZON, 'categoryID_to_category_mapping.csv')
cat_map_df = pd.DataFrame(list(ncid_to_ocid.items()), columns=['numeric_categoryID', 'original_categoryID'])
cat_map_df.sort_values(by='numeric_categoryID', inplace=True)
cat_map_df.to_csv(category_mapping_path, index=False)
print(f"  Guardado: {category_mapping_path}")

# itemID_numeric_to_originalItemID_mapping.csv
item_id_numeric_to_original_path = os.path.join(FINAL_STRUCTURED_DIR_AMAZON, 'itemID_numeric_to_originalItemID_mapping.csv')
item_map_df_simple = pd.DataFrame(list(nid_to_oid.items()), columns=['numeric_itemID', 'original_itemID'])
item_map_df_simple.sort_values(by='numeric_itemID', inplace=True)
item_map_df_simple.to_csv(item_id_numeric_to_original_path, index=False)
print(f"  Guardado: {item_id_numeric_to_original_path}")

# priceLevel_summary.csv
price_level_summary_path = os.path.join(FINAL_STRUCTURED_DIR_AMAZON, 'priceLevel_summary.csv')
price_summary_df = nid_to_pid_map_df.groupby('pid').size().reset_index(name='item_count')
price_summary_df.rename(columns={'pid': 'price_levelID'}, inplace=True)
price_summary_df.sort_values(by='price_levelID', inplace=True)
price_summary_df.to_csv(price_level_summary_path, index=False)
print(f"  Guardado: {price_level_summary_path}")

# structured_train_sessions.csv & structured_test_sessions.csv
train_sessions_path = os.path.join(FINAL_STRUCTURED_DIR_AMAZON, 'structured_train_sessions.csv')
save_structured_sessions('structured_train_sessions.csv', train_session_ids, train_item_seqs, train_price_seqs, train_cat_seqs, FINAL_STRUCTURED_DIR_AMAZON)
print(f"  Guardado: {train_sessions_path}")

test_sessions_path = os.path.join(FINAL_STRUCTURED_DIR_AMAZON, 'structured_test_sessions.csv')
save_structured_sessions('structured_test_sessions.csv', test_session_ids, test_item_seqs, test_price_seqs, test_cat_seqs, FINAL_STRUCTURED_DIR_AMAZON)
print(f"  Guardado: {test_sessions_path}")

print(f"Preprocesamiento completado para {datasets_name}.")
print(f"Archivos generados en:")
print(f"  - Base procesada: {AMAZON_PROCESSED_DIR}")
print(f"  - Mapeos intermedios (PILL): {PILL_UTILS_DATA_DIR}")
print(f"  - Estructurados finales: {FINAL_STRUCTURED_DIR_AMAZON}")
