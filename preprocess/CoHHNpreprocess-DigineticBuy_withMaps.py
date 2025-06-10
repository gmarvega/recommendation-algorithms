'''

creat by kun at Sep 2021
Reference: 
https://github.com/xiaxin1998/DHCN https://github.com/Wang-Shuo/Neural-Attentive-Session-Based-Recommendation-PyTorch/blob/master/main.py

'''

import os
import pickle
import time
import pandas as pd
import numpy as np
import math

# =========================
# 1. Definición de rutas de salida
# =========================

datasets_name = 'digineticaBuy'
# Corregido para que la ruta base sea relativa al script actual si es necesario,
# o absoluta si se prefiere. Asumiendo que 'output' está al mismo nivel que 'preprocess'.
# Si el script se ejecuta desde 'preprocess', la ruta sería '../output/...'
# Para mantener consistencia con el script de Amazon, se usará una ruta desde la raíz del proyecto.
# Asumiendo que el script se ejecuta desde la raíz del proyecto o que las rutas son relativas a ella.
# Si el script está en 'preprocess/', y 'output/' está en la raíz:
output_base_path = 'output' # O '../output' si se ejecuta desde preprocess/
train_data_path = os.path.join(output_base_path, 'Diginetica', datasets_name)
os.makedirs(train_data_path, exist_ok=True)

# Rutas a los datos originales de Diginetica (ajustar si es necesario)
original_data_base_path = '../../datasets/diginetica' # Asumiendo que está en la raíz o accesible así
products = 'products'
path_products = os.path.join(original_data_base_path, products + '.csv')

price_level_num = 10

print(f"Intentando leer {path_products} ...")
try:
    items = pd.read_csv(path_products, sep=';', on_bad_lines='skip')
    print(f"Leído {path_products} con shape: {items.shape}")
    print("Columnas detectadas:", items.columns)
    print("Primeras filas:")
    print(items.head())
except Exception as e:
    print(f"Error leyendo {path_products}: {e}")
    raise
# Ahora las columnas son: 'itemId', 'pricelog2', 'product.name.tokens'
# Creamos la columna 'inf' como string combinando los tres campos separados por ';'
items['inf'] = items['itemId'].astype(str) + ';' + items['pricelog2'].astype(str) + ';' + items['product.name.tokens'].astype(str)
items = items[['inf']]
print("Columna 'inf' creada correctamente. Primeras filas:")
print(items.head())

def reg_itemID_original(strlist): # Renombrado para evitar confusión con la función interna re_itemID
    results = int(strlist.split(';')[0])
    return results
def reg_price(strlist):
    price_str = strlist.split(';')[1]
    # Manejar posibles errores si el valor no es un número válido para int()
    try:
        price = int(price_str)
        results = 2 ** price
    except ValueError:
        results = np.nan # O algún otro valor para indicar precio inválido/faltante
    return results

items['itemID'] = items.inf.map(reg_itemID_original)
items['price'] = items.inf.map(reg_price)
items.dropna(subset=['price'], inplace=True) # Eliminar filas donde el precio no pudo ser parseado
items = items[['itemID', 'price']]

products_cate = 'product-categories'
path_products_cate = os.path.join(original_data_base_path, products_cate + '.csv')

print(f"Intentando leer {path_products_cate} ...")
try:
    item_cate_df = pd.read_csv(path_products_cate)
    print(f"Leído {path_products_cate} con shape: {item_cate_df.shape}")
    print("Primeras filas:")
    print(item_cate_df.head())
except Exception as e:
    print(f"Error leyendo {path_products_cate}: {e}")
    raise
item_cate_df = item_cate_df.rename(columns={'itemId;categoryId': 'inf'})[['inf']]
def re_categoryId(strlist):
    results = int(strlist.split(';')[1])
    return results
item_cate_df['itemID'] = item_cate_df.inf.map(reg_itemID_original) # Usar la función renombrada
item_cate_df['cate'] = item_cate_df.inf.map(re_categoryId)
item_cate_df = item_cate_df[['itemID', 'cate']]
item_all = pd.merge(item_cate_df, items, how='left', on='itemID')
item_all.drop_duplicates(subset=['itemID'], keep='first', inplace=True)
item_all.dropna(inplace=True) # Eliminar ítems sin precio o categoría después del merge

group_cate_num = pd.DataFrame(item_all.groupby(item_all['cate']).count())
group_num = group_cate_num.reset_index()[['cate', 'itemID']].rename(columns={'itemID': 'count'})

group_cate_min = pd.DataFrame(item_all['price'].groupby(item_all['cate']).min())
group_min = group_cate_min.reset_index()[['cate', 'price']].rename(columns={'price': 'min'})

group_cate_max = pd.DataFrame(item_all['price'].groupby(item_all['cate']).max())
group_max = group_cate_max.reset_index()[['cate', 'price']].rename(columns={'price': 'max'})

group_cate_mean = pd.DataFrame(item_all['price'].groupby(item_all['cate']).mean())
group_mean = group_cate_mean.reset_index()[['cate', 'price']].rename(columns={'price': 'mean'})

group_cate_std = pd.DataFrame(item_all['price'].groupby(item_all['cate']).std())
group_std = group_cate_std.reset_index()[['cate', 'price']].rename(columns={'price': 'std'})
group_std.fillna(0, inplace=True) # Rellenar std Nan con 0 para categorías con un solo item

item_data1 = pd.merge(item_all, group_num, how='left', on='cate')
item_data2 = pd.merge(item_data1, group_min, how='left', on='cate')
item_data3 = pd.merge(item_data2, group_max, how='left', on='cate')
item_data4 = pd.merge(item_data3, group_mean, how='left', on='cate')
item_data5 = pd.merge(item_data4, group_std, how='left', on='cate')

item_data = item_data5[item_data5['count'] > 9]
item_data = item_data[item_data['std'] != 0].copy() # Usar .copy() para evitar SettingWithCopyWarning

def logistic(t, u, s):
    gama = s * 3 ** (0.5) / math.pi
    results = 1 / (1 + math.exp((t - u) / gama))
    return results
def get_price_level(price, p_min, p_max, mean, std):
    if std == 0:
        return -1
    fenzi = logistic(price, mean, std) - logistic(p_min, mean, std)
    fenmu = logistic(p_max, mean, std) - logistic(p_min, mean, std)
    if fenmu == 0 or price == 0:
        return -1
    results = int(fenzi / fenmu * price_level_num) + 1
    return results


item_data.loc[:, 'price_level'] = item_data.apply(lambda row: get_price_level(row['price'], row['min'], row['max'], row['mean'], row['std']), axis=1)
item_final = item_data[item_data['price_level'] != -1].copy() # .copy() para evitar warnings

# group_price_num = pd.DataFrame(item_final.groupby(item_final['price_level']).count()) # No se usa group_num después
# group_num = group_price_num.reset_index()[['price_level', 'itemID']].rename(columns={'itemID': 'count'})

item_views_file = 'train-item-views' # Renombrado para claridad
path_item_views = os.path.join(original_data_base_path, item_views_file + '.csv')

print(f"Intentando leer {path_item_views} ...")
try:
    interaction = pd.read_csv(path_item_views, sep=';')
    print(f"Leído {path_item_views} con shape: {interaction.shape}")
    print("Primeras filas:")
    print(interaction.head())
except Exception as e:
    print(f"Error leyendo {path_item_views}: {e}")
    raise
# Renombrar columnas directamente si el CSV tiene cabeceras, o ajustar parseo si no.
# Asumiendo que las columnas son: sessionId, userId, itemId, timeframe, eventdate
# interaction.columns = ['sessionId', 'userId', 'itemId', 'timeframe', 'eventdate'] # Si no tiene header
interaction = interaction[['sessionId', 'itemId', 'timeframe']].rename(columns={'sessionId':'sessionID', 'itemId':'itemID_original', 'timeframe':'time'})


# Asegurar que item_final tiene 'itemID' como el ID original para el merge
item_final_for_merge = item_final.rename(columns={'itemID': 'itemID_original'})
user_item1 = pd.merge(interaction, item_final_for_merge, how='inner', on='itemID_original') # inner para asegurar que solo queden items con info de precio/cat
user_item2 = user_item1.dropna(axis=0)

user_item2.sort_values(by=["sessionID", "time"], inplace=True, ascending=[True, True])

user_click_num = pd.DataFrame(user_item2.groupby(user_item2['sessionID'])['itemID_original'].count()) # Contar cualquier columna no nula
click_num = user_click_num.reset_index().rename(columns={'itemID_original': 'click_num'})
item_data3 = pd.merge(user_item2, click_num, how='left', on='sessionID')
item_data4 = item_data3[item_data3['click_num'] > 1]
# 'cate' en item_final es el original, 'price_level' es el calculado.
data_all_raw = item_data4[['sessionID', 'itemID_original', 'time', 'price', 'cate', 'price_level']].copy()


data_all_raw = data_all_raw.rename(columns={'price_level': 'priceLevel', 'cate': 'category_original', 'itemID_original': 'itemID'})
# Columnas finales antes de mapear IDs: sessionID, itemID (original), time, price, priceLevel, category_original

# Mapeo de IDs
reviewerID2sessionID_map = {} # Renombrado para evitar conflicto con la función
originalItemID2numericItemID_map = {} # Renombrado (antes asin2itemID)
originalCategoryID2numericCategoryID_map = {} # Renombrado (antes category2categoryID)

current_session_num = 0
current_item_num = 0
current_category_num = 0

data_all_processed = data_all_raw.copy()

# Aplicar mapeos
unique_sessions = data_all_processed['sessionID'].unique()
for session_orig in unique_sessions:
    if session_orig not in reviewerID2sessionID_map:
        current_session_num += 1
        reviewerID2sessionID_map[session_orig] = current_session_num
data_all_processed['sessionID_numeric'] = data_all_processed['sessionID'].map(reviewerID2sessionID_map)

unique_items = data_all_processed['itemID'].unique()
for item_orig in unique_items:
    if item_orig not in originalItemID2numericItemID_map:
        current_item_num += 1
        originalItemID2numericItemID_map[item_orig] = current_item_num
data_all_processed['itemID_numeric'] = data_all_processed['itemID'].map(originalItemID2numericItemID_map)

unique_categories = data_all_processed['category_original'].unique()
for cat_orig in unique_categories:
    if cat_orig not in originalCategoryID2numericCategoryID_map:
        current_category_num += 1
        originalCategoryID2numericCategoryID_map[cat_orig] = current_category_num
data_all_processed['category_numeric'] = data_all_processed['category_original'].map(originalCategoryID2numericCategoryID_map)


print('#session: ', current_session_num)
print('#item: ', current_item_num)
print('#category: ', current_category_num)


item2price_level_map = {} # Mapeo de itemID_numeric a priceLevel
for _, row in data_all_processed.iterrows():
    if row['itemID_numeric'] not in item2price_level_map:
        item2price_level_map[row['itemID_numeric']] = row['priceLevel']

# Seleccionar columnas finales para 'data'
data = data_all_processed[['sessionID_numeric', 'itemID_numeric', 'priceLevel', 'category_numeric']].copy()
data.rename(columns={'sessionID_numeric':'sessionID', 'itemID_numeric':'itemID', 'category_numeric':'category'}, inplace=True)


item_inter_num_df = pd.DataFrame(data.groupby(data['itemID'])['sessionID'].count()) # Contar sesiones por itemID
item_inter_num_df = item_inter_num_df.reset_index().rename(columns={'sessionID': 'item_num'})
data = pd.merge(data, item_inter_num_df, how='left', on='itemID')

data = data[data['item_num'] > 9]
data = data[['sessionID', 'itemID', 'priceLevel', 'category']] # Columnas finales para construir secuencias

sess_all_items = {}
price_all_levels = {}
cate_all_ids = {}

for _, row in data.iterrows():
    sess_id = row['sessionID']
    item_id = row['itemID']
    price_level_val = row['priceLevel'] # Renombrado para evitar conflicto
    cate_id_val = row['category'] # Renombrado

    if sess_id not in sess_all_items:
        sess_all_items[sess_id] = []
        price_all_levels[sess_id] = []
        cate_all_ids[sess_id] = []
    
    sess_all_items[sess_id].append(item_id)
    price_all_levels[sess_id].append(price_level_val)
    cate_all_ids[sess_id].append(cate_id_val)


sess_total_numeric = data['sessionID'].max() if not data.empty else 0
split_num_sessions = int(sess_total_numeric / 10 * 9) if sess_total_numeric > 0 else 0


tra_sess_dict = dict() 
tes_sess_dict = dict()
tra_price_dict = dict()
tes_price_dict = dict()
tra_cate_dict = dict()
tes_cate_dict = dict()

for sess_temp_id in sess_all_items.keys():
    all_item_seqs = sess_all_items[sess_temp_id]
    all_price_level_seqs = price_all_levels[sess_temp_id]
    all_cate_id_seqs = cate_all_ids[sess_temp_id]

    if len(all_item_seqs) < 2:
        continue
    if len(all_item_seqs) > 20: # Truncar secuencias largas
        all_item_seqs = all_item_seqs[:20]
        all_price_level_seqs = all_price_level_seqs[:20]
        all_cate_id_seqs = all_cate_id_seqs[:20]
        
    if int(sess_temp_id) < split_num_sessions:
        tra_sess_dict[sess_temp_id] = all_item_seqs
        tra_price_dict[sess_temp_id] = all_price_level_seqs
        tra_cate_dict[sess_temp_id] = all_cate_id_seqs
    else:
        tes_sess_dict[sess_temp_id] = all_item_seqs
        tes_price_dict[sess_temp_id] = all_price_level_seqs
        tes_cate_dict[sess_temp_id] = all_cate_id_seqs

# item_dict_seq: Mapeo de itemID numérico global a ID local de secuencia de train
item_dict_seq = {} 
cate_dict_seq = {} 
price_dict_seq = {} 


def obtian_tra_seqs():
    train_item_seqs_local = []
    train_price_level_seqs_local = []
    train_cate_id_seqs_local = []
    
    item_ctr_local = 1
    price_ctr_local = 1
    cate_ctr_local = 1
    
    # Asegurar que los diccionarios se limpian o se definen aquí si son globales y modificados
    global item_dict_seq, price_dict_seq, cate_dict_seq
    item_dict_seq = {}
    price_dict_seq = {}
    cate_dict_seq = {}

    sorted_train_session_ids = sorted(tra_sess_dict.keys())

    for s_id in sorted_train_session_ids:
        global_item_id_seq = tra_sess_dict[s_id]
        global_price_level_seq = tra_price_dict[s_id]
        global_cate_id_seq = tra_cate_dict[s_id]
        
        out_item_seq_local = []
        out_price_seq_local = []
        out_cate_seq_local = []
        
        for item_global_id, price_global_id, cate_global_id in zip(global_item_id_seq, global_price_level_seq, global_cate_id_seq):
            if item_global_id not in item_dict_seq:
                item_dict_seq[item_global_id] = item_ctr_local
                item_ctr_local += 1
            out_item_seq_local.append(item_dict_seq[item_global_id])
            
            if price_global_id not in price_dict_seq:
                price_dict_seq[price_global_id] = price_ctr_local
                price_ctr_local += 1
            out_price_seq_local.append(price_dict_seq[price_global_id])
            
            if cate_global_id not in cate_dict_seq:
                cate_dict_seq[cate_global_id] = cate_ctr_local
                cate_ctr_local += 1
            out_cate_seq_local.append(cate_dict_seq[cate_global_id])
            
        if len(out_item_seq_local) < 2:
            continue
        train_item_seqs_local.append(out_item_seq_local)
        train_price_level_seqs_local.append(out_price_seq_local)
        train_cate_id_seqs_local.append(out_cate_seq_local)
        
    print("#train_session", len(train_item_seqs_local))
    print("#train_items (local IDs)", item_ctr_local - 1)
    print("#train_price_levels (local IDs)", price_ctr_local - 1)
    print("#train_categories (local IDs)", cate_ctr_local - 1)
    return train_item_seqs_local, train_price_level_seqs_local, train_cate_id_seqs_local

def obtian_tes_seqs():
    test_item_seqs_local = []
    test_price_level_seqs_local = []
    test_cate_id_seqs_local = []

    sorted_test_session_ids = sorted(tes_sess_dict.keys())

    for s_id in sorted_test_session_ids:
        global_item_id_seq = tes_sess_dict[s_id]
        global_price_level_seq = tes_price_dict[s_id]
        global_cate_id_seq = tes_cate_dict[s_id]
        
        out_item_seq_local = []
        out_price_seq_local = []
        out_cate_seq_local = []
        
        for item_global_id, price_global_id, cate_global_id in zip(global_item_id_seq, global_price_level_seq, global_cate_id_seq):
            if item_global_id in item_dict_seq: # Solo ítems vistos en train
                out_item_seq_local.append(item_dict_seq[item_global_id])
                if price_global_id in price_dict_seq: # Solo price levels vistos en train
                     out_price_seq_local.append(price_dict_seq[price_global_id])
                else: # Manejar price levels no vistos en train (ej. asignar un ID especial o ignorar)
                    # Para este caso, si no está, no se añade, lo que podría acortar la secuencia de precios
                    pass
                if cate_global_id in cate_dict_seq: # Solo categorías vistas en train
                    out_cate_seq_local.append(cate_dict_seq[cate_global_id])
                else:
                    pass # Similar para categorías
        
        # Asegurar que todas las secuencias (item, price, cate) tengan la misma longitud después del filtrado
        min_len = min(len(out_item_seq_local), len(out_price_seq_local), len(out_cate_seq_local))
        out_item_seq_local = out_item_seq_local[:min_len]
        out_price_seq_local = out_price_seq_local[:min_len]
        out_cate_seq_local = out_cate_seq_local[:min_len]

        if len(out_item_seq_local) < 2:
            continue
        test_item_seqs_local.append(out_item_seq_local)
        test_price_level_seqs_local.append(out_price_seq_local)
        test_cate_id_seqs_local.append(out_cate_seq_local)
        
    return test_item_seqs_local, test_price_level_seqs_local, test_cate_id_seqs_local


def process_seqs_no_aug(item_seqs, price_seqs, cate_seqs): # Renombrado para claridad
    print("no data augment")
    out_item_sub_seqs = []
    out_price_sub_seqs = []
    out_cate_sub_seqs = []
    labels = []
    for i_seq, p_seq, c_seq in zip(item_seqs, price_seqs, cate_seqs):     
        labels.append(i_seq[-1])
        out_item_sub_seqs.append(i_seq[:-1])
        out_price_sub_seqs.append(p_seq[:-1])
        out_cate_sub_seqs.append(c_seq[:-1])
    return out_item_sub_seqs, out_price_sub_seqs, out_cate_sub_seqs, labels

# Obtener secuencias con IDs locales
train_item_seqs, train_price_seqs, train_cate_seqs = obtian_tra_seqs()
test_item_seqs, test_price_seqs, test_cate_seqs = obtian_tes_seqs()

# Procesar para obtener sub-secuencias y labels
tr_sub_items, tr_sub_prices, tr_sub_cates, tr_labs = process_seqs_no_aug(train_item_seqs, train_price_seqs, train_cate_seqs)
te_sub_items, te_sub_prices, te_sub_cates, te_labs = process_seqs_no_aug(test_item_seqs, test_price_seqs, test_cate_seqs)


print('train sequence (local IDs): ', tr_sub_items[:5])
print('train price (local IDs): ', tr_sub_prices[:5])
print('train category (local IDs): ', tr_sub_cates[:5])
print('train lab (local IDs): ', tr_labs[:5])


def to_matrix_relations(all_item_seqs, all_price_seqs, all_cate_seqs): # Renombrado
    # Estas secuencias deben tener IDs locales (los que genera obtian_tra/tes)
    price_to_items_dict = {}
    price_to_cates_dict = {}
    cate_to_items_dict = {}

    for i_s, p_s, c_s in zip(all_item_seqs, all_price_seqs, all_cate_seqs):
        for item_local_id, price_local_id, cate_local_id in zip(i_s, p_s, c_s):
            price_to_items_dict.setdefault(price_local_id, []).append(item_local_id)
            price_to_cates_dict.setdefault(price_local_id, []).append(cate_local_id)
            cate_to_items_dict.setdefault(cate_local_id, []).append(item_local_id)

    # Ordenar por clave (ID local) para consistencia si es necesario
    price_to_items_list = [price_to_items_dict[k] for k in sorted(price_to_items_dict.keys())]
    price_to_cates_list = [price_to_cates_dict[k] for k in sorted(price_to_cates_dict.keys())]
    cate_to_items_list = [cate_to_items_dict[k] for k in sorted(cate_to_items_dict.keys())]
    
    print("#price_level_keys (local)", len(price_to_items_dict))
    print("#category_keys (local)", len(cate_to_items_dict))
    
    return price_to_items_list, price_to_cates_list, cate_to_items_list


def create_data_masks(all_sessions_local_ids): # Renombrado
    indptr, indices, data_mask_values = [], [], [] # Renombrado data a data_mask_values
    indptr.append(0)
    for j in range(len(all_sessions_local_ids)):
        session_unique_ids = np.unique(all_sessions_local_ids[j]) 
        length = len(session_unique_ids)
        s = indptr[-1]
        indptr.append((s + length))
        for item_local_id_in_session in session_unique_ids:
            indices.append(item_local_id_in_session - 1) # Asumiendo que los IDs locales son 1-based
            data_mask_values.append(1)
    return (data_mask_values, indices, indptr)


# Usar las secuencias completas (no las sub-secuencias) para las matrices de relación
# train_item_seqs, test_item_seqs etc. contienen IDs locales
tra_pi_rel, tra_pc_rel, tra_ci_rel = to_matrix_relations(train_item_seqs + test_item_seqs, 
                                                      train_price_seqs + test_price_seqs, 
                                                      train_cate_seqs + test_cate_seqs)

# Crear tuplas de datos para pickle
# tr_sub_items, te_sub_items etc. son las secuencias de entrada al modelo (ya procesadas con _no_aug)
train_data_tuple = (tr_sub_items, tr_sub_prices, 
                    create_data_masks(tr_sub_items), create_data_masks(tr_sub_prices), 
                    create_data_masks(tra_pi_rel), create_data_masks(tra_pc_rel), create_data_masks(tra_ci_rel), 
                    tr_labs)
test_data_tuple = (te_sub_items, te_sub_prices, 
                   create_data_masks(te_sub_items), create_data_masks(te_sub_prices), 
                   create_data_masks(tra_pi_rel), create_data_masks(tra_pc_rel), create_data_masks(tra_ci_rel), 
                   te_labs)


interactions_count = sum(len(s) for s in train_item_seqs) + sum(len(s) for s in test_item_seqs)
sessions_count = len(train_item_seqs) + len(test_item_seqs)
avg_len = interactions_count / sessions_count if sessions_count > 0 else 0

print('#interactions: ', interactions_count)
print('#session: ', sessions_count)
print('sequence average length: ', avg_len)


# =========================
# Guardar archivos de datos principales (train/test)
# =========================
path_data_train_pickle = os.path.join(train_data_path, "train.txt")
path_data_test_pickle = os.path.join(train_data_path, "test.txt")

with open(path_data_train_pickle, 'wb') as f_train:
    pickle.dump(train_data_tuple, f_train)
with open(path_data_test_pickle, 'wb') as f_test:
    pickle.dump(test_data_tuple, f_test)
print(f"Saved train data to: {path_data_train_pickle}")
print(f"Saved test data to: {path_data_test_pickle}")


# =========================
# 2. Guardar mapping_dicts.pkl
# =========================
# reviewerID2sessionID_map: original sessionID -> numeric sessionID
# originalItemID2numericItemID_map: original itemID -> numeric itemID
# originalCategoryID2numericCategoryID_map: original category string/ID -> numeric categoryID
# item_dict_seq: numeric itemID -> local train sequence itemID
# price_dict_seq: priceLevel (numeric) -> local train sequence priceLevelID
# cate_dict_seq: numeric categoryID -> local train sequence categoryID
# item2price_level_map: numeric itemID -> priceLevel (numeric)

mapping_payload = {
    'sessionID_original_to_numeric': reviewerID2sessionID_map,
    'itemID_original_to_numeric': originalItemID2numericItemID_map,
    'categoryOriginal_to_numeric': originalCategoryID2numericCategoryID_map,
    'itemID_numeric_to_localTrainSeqID': item_dict_seq,
    'priceLevel_numeric_to_localTrainSeqID': price_dict_seq,
    'categoryNumeric_to_localTrainSeqID': cate_dict_seq,
    'itemID_numeric_to_priceLevel': item2price_level_map
}
mapping_dicts_path = os.path.join(train_data_path, 'mapping_dicts.pkl')
with open(mapping_dicts_path, 'wb') as f:
    pickle.dump(mapping_payload, f)
print(f"Saved mapping dictionaries to: {mapping_dicts_path}")

# =========================
# 3. Generar y guardar item_mapping_table_with_sequences.csv
# =========================
# item_final: 'itemID' (original), 'price', 'cate' (original), 'price_level'
# originalItemID2numericItemID_map: original itemID -> numeric itemID (global)
# item_dict_seq: numeric itemID (global) -> local train sequence itemID

# Mapeos inversos necesarios:
localTrainSeqID_to_numericItemID = {v: k for k, v in item_dict_seq.items()}
numericItemID_to_originalItemID = {v: k for k, v in originalItemID2numericItemID_map.items()}

original_item_id_to_train_seq_indices = {}
# tr_sub_items contiene secuencias de IDs locales de entrenamiento
for train_seq_idx, local_id_item_sequence in enumerate(train_item_seqs): # Usar train_item_seqs (completas, no sub)
    for local_item_id in local_id_item_sequence:
        if local_item_id in localTrainSeqID_to_numericItemID:
            numeric_item_id = localTrainSeqID_to_numericItemID[local_item_id]
            if numeric_item_id in numericItemID_to_originalItemID:
                original_item_id_val = numericItemID_to_originalItemID[numeric_item_id]
                original_item_id_to_train_seq_indices.setdefault(original_item_id_val, []).append(train_seq_idx)

# Usar item_final que tiene 'itemID' como original ID
item_mapping_table_df = item_final[['itemID', 'price', 'cate', 'price_level']].copy()
item_mapping_table_df.rename(columns={'price_level': 'priceLevel', 'cate': 'category_original_name'}, inplace=True)
item_mapping_table_df['train_seq_indices'] = item_mapping_table_df['itemID'].map(lambda x: original_item_id_to_train_seq_indices.get(x, []))

item_mapping_table_filepath = os.path.join(train_data_path, 'item_mapping_table_with_sequences.csv')
item_mapping_table_df.to_csv(item_mapping_table_filepath, index=False)
print(f"Saved item mapping table with sequences to: {item_mapping_table_filepath}")


# =========================
# 4. Generar y guardar categoryID_to_category_mapping.csv
# =========================
# originalCategoryID2numericCategoryID_map: original category string/ID -> numeric categoryID
numericCategoryID_to_originalCategory = {v: k for k, v in originalCategoryID2numericCategoryID_map.items()}
df_cat_map = pd.DataFrame(list(numericCategoryID_to_originalCategory.items()), columns=['categoryID_numeric', 'category_original_name'])
category_mapping_filepath = os.path.join(train_data_path, 'categoryID_to_category_mapping.csv')
df_cat_map.to_csv(category_mapping_filepath, index=False)
print(f"Saved categoryID to category mapping to: {category_mapping_filepath}")

# =========================
# 5. Generar y guardar itemID_to_originalItemID_mapping.csv
# =========================
# numericItemID_to_originalItemID ya está creado arriba
df_item_orig_map = pd.DataFrame(list(numericItemID_to_originalItemID.items()), columns=['itemID_numeric', 'original_itemID'])
itemid_original_mapping_filepath = os.path.join(train_data_path, 'itemID_numeric_to_originalItemID_mapping.csv') # Nombre de archivo más descriptivo
df_item_orig_map.to_csv(itemid_original_mapping_filepath, index=False)
print(f"Saved itemID_numeric to original_itemID mapping to: {itemid_original_mapping_filepath}")

# =========================
# 6. Generar y guardar priceLevel_summary.csv
# =========================
# item_final tiene 'price' y 'price_level'
price_summary_stats_df = item_final.groupby('price_level').agg(
    count=('price', 'count'),
    min_price=('price', 'min'),
    max_price=('price', 'max'),
    mean_price=('price', 'mean'),
    std_price=('price', 'std')
).reset_index().rename(columns={'price_level': 'priceLevel'})
price_summary_filepath = os.path.join(train_data_path, 'priceLevel_summary.csv')
price_summary_stats_df.to_csv(price_summary_filepath, index=False)
print(f"Saved priceLevel summary to: {price_summary_filepath}")

# =========================
# 7. Generar y guardar structured_train_sessions.csv
# =========================
# train_item_seqs, train_price_seqs, train_cate_seqs contienen IDs locales
df_struct_train_sessions = pd.DataFrame({
    'sessionID_numeric_train_index': range(1, len(train_item_seqs) + 1), 
    'item_sequence_localIDs': train_item_seqs,
    'price_sequence_localIDs': train_price_seqs,
    'category_sequence_localIDs': train_cate_seqs,
    'session_length': [len(s) for s in train_item_seqs]
})
structured_train_sessions_filepath = os.path.join(train_data_path, 'structured_train_sessions.csv')
df_struct_train_sessions.to_csv(structured_train_sessions_filepath, index=False)
print(f"Saved structured train session sequences to: {structured_train_sessions_filepath}")

# =========================
# 8. Generar y guardar structured_test_sessions.csv
# =========================
df_struct_test_sessions = pd.DataFrame({
    'sessionID_numeric_test_index': range(len(train_item_seqs) + 1, len(train_item_seqs) + len(test_item_seqs) + 1),
    'item_sequence_localIDs': test_item_seqs,
    'price_sequence_localIDs': test_price_seqs,
    'category_sequence_localIDs': test_cate_seqs,
    'session_length': [len(s) for s in test_item_seqs]
})
structured_test_sessions_filepath = os.path.join(train_data_path, 'structured_test_sessions.csv')
df_struct_test_sessions.to_csv(structured_test_sessions_filepath, index=False)
print(f"Saved structured test session sequences to: {structured_test_sessions_filepath}")

print(f"Dataset: {datasets_name}")
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print("Done processing and saving all files.")
