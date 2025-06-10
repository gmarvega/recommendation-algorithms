#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import subprocess
import numpy as np 
import pandas as pd 
import csv
import pickle
import argparse # Requisito 1: Importar argparse
import math # Asegúrate de que math esté importado al inicio del script
from scipy.stats import logistic # Para la discretización logística

# Definición de rutas base (relativas al directorio raíz del proyecto)
BASE_DIR = os.getcwd()

def logistic_custom(t, u, s):
    if s == 0: # Evitar división por cero si la desviación estándar es 0
        return 0.5 if t == u else (1.0 if t > u else 0.0) # Comportamiento sigmoide degenerado
    gama = s * (3**0.5) / math.pi
    if gama == 0: # Evitar división por cero si gama resulta ser 0
         return 0.5 if t == u else (1.0 if t > u else 0.0)
    # Corregido: el exponente debe ser negativo para una sigmoide estándar creciente
    try:
        exponent = (t - u) / gama
        # Prevenir overflow en math.exp()
        if exponent > 700: # math.exp(709) es aprox el límite antes de OverflowError
            return 0.0
        elif exponent < -700:
            return 1.0
        return 1 / (1 + math.exp(exponent))
    except OverflowError:
        # Si (t-u)/gama es muy negativo, exp() es cercano a 0, exp(muy negativo) es muy grande, resultado es cercano a 1
        # Si (t-u)/gama es muy positivo, exp(muy positivo) es muy grande, resultado es cercano a 0
        return 0.0 if (t-u)/gama > 0 else 1.0


def get_price_level_custom(price, p_min, p_max, mean, std, n_ranges):
    if std == 0:
        return -1
    
    # Usar la función logística personalizada
    log_price = logistic_custom(price, mean, std)
    log_p_min = logistic_custom(p_min, mean, std)
    log_p_max = logistic_custom(p_max, mean, std)
        
    fenmu = log_p_max - log_p_min
    
    if fenmu == 0:
        return -1

    # Lógica de manejo de precios fuera del rango de la categoría
    if price < p_min:
        return 1
    elif price > p_max:
        return n_ranges

    fenzi = log_price - log_p_min
    
    # Calcular el nivel, asegurando que esté en el rango [0, n_ranges - 1]
    # y luego ajustarlo a [1, n_ranges].
    level = int((fenzi / fenmu) * n_ranges)
    
    if level < 0:
        level = 0
    elif level >= n_ranges:
        level = n_ranges - 1
    
    # Ajustar el rango de 0 a n_ranges-1 a 1 a n_ranges
    return level + 1

DATASETS_ROOT = os.path.join(BASE_DIR, 'datasets')
DIGINETICA_RAW_DIR = os.path.join(DATASETS_ROOT, 'diginetica', 'dataset')
DIGINETICA_PROCESSED_DIR = os.path.join(DATASETS_ROOT, 'diginetica')
PILL_DIR = os.path.join(BASE_DIR, 'PILL')
PILL_UTILS_DATA_DIR = os.path.join(PILL_DIR, 'utils', 'data')

# Archivos fuente esperados
DIGINETICA_TRAIN_VIEWS = os.path.join(DIGINETICA_RAW_DIR, 'train-item-views.csv')
DIGINETICA_TRAIN_PURCHASES = os.path.join(DIGINETICA_RAW_DIR, 'train-purchases.csv')
DIGINETICA_PRODUCT_CATEGORIES = os.path.join(DIGINETICA_RAW_DIR, 'product-categories.csv')
DIGINETICA_PRODUCTS = os.path.join(DIGINETICA_RAW_DIR, 'products.csv')

# Archivos generados clave
NEW_OID_NID_CSV = os.path.join(DIGINETICA_PROCESSED_DIR, 'oid2nid.csv') # Corregido en Subtarea 7
NUM_ITEMS_TXT = os.path.join(DIGINETICA_PROCESSED_DIR, 'num_items.txt')
OID_NID_CATEGORY_CSV = os.path.join(PILL_UTILS_DATA_DIR, 'oid2nid_category.csv') 
NEW_OID_NID_CATEGORY_CSV = os.path.join(PILL_UTILS_DATA_DIR, 'new_oid2nid_category.csv')
NIID_NCID_TXT_SRC = os.path.join(PILL_UTILS_DATA_DIR, 'niid_2_ncid.txt') 
NIID_PRICEID_TXT_SRC = os.path.join(PILL_UTILS_DATA_DIR, 'niid_2_priceid.txt') 
NIID_NCID_TXT_DEST = os.path.join(DIGINETICA_PROCESSED_DIR, 'niid_2_ncid.txt')
NIID_PRICEID_TXT_DEST = os.path.join(DIGINETICA_PROCESSED_DIR, 'niid_2_priceid.txt')

# Configuración global para el número de rangos de precios
DEFAULT_NUM_PRICE_RANGES = 50

# --- INICIO: Funciones integradas para generación de new_oid2nid_category.csv ---

def itemid_2_categoryid_fast_integrated():
    """
    Genera la secuencia de categorías para los items de entrenamiento.
    Lee:
        - DIGINETICA_PRODUCT_CATEGORIES
        - NEW_OID_NID_CSV
        - train.txt (en DIGINETICA_PROCESSED_DIR)
    Escribe:
        - train_cate_seq.txt (en PILL_UTILS_DATA_DIR)
    """
    cate_dict = {}
    oid2nid = {}
    # Leer categorías
    with open(DIGINETICA_PRODUCT_CATEGORIES, 'r') as cate_f:
        cate_lines = cate_f.readlines()
        for each_cate_line in cate_lines:
            cate_line_2_list = each_cate_line.strip().split(';')
            if len(cate_line_2_list) > 1:
                cate_dict[cate_line_2_list[0]] = cate_line_2_list[1]
    # Leer oid2nid
    with open(NEW_OID_NID_CSV, 'r') as o2nid_f:
        oid2nid_lines = o2nid_f.readlines()
        for each_id_line in oid2nid_lines:
            id_line_2_list = each_id_line.strip().split(',')
            if len(id_line_2_list) > 1:
                oid2nid[id_line_2_list[0]] = id_line_2_list[1]
    # Leer secuencias de entrenamiento
    train_txt_path = os.path.join(DIGINETICA_PROCESSED_DIR, 'train.txt')
    with open(train_txt_path, 'r') as seq_f:
        lines = seq_f.readlines()
        each_cate_list = []
        for no, line in enumerate(lines, 1):
            if no % 10000 == 0:
                print(f"Procesando itemid_2_categoryid_fast: {len(lines)} líneas, procesando línea {no}")
            line_2_id_list = line.strip().split(',')
            cate_seq = []
            for each_id in line_2_id_list:
                oid = None
                for k, v in oid2nid.items():
                    if v == each_id.strip():
                        oid = k
                        break
                if oid is None:
                    cate_seq.append('0')
                else:
                    cate_seq.append(cate_dict.get(oid, '0'))
            each_cate_list.append(cate_seq)
    # Escribir secuencia de categorías
    write_category_seq_integrated(each_cate_list)

def write_category_seq_integrated(cate_list):
    """
    Escribe la secuencia de categorías en train_cate_seq.txt (en PILL_UTILS_DATA_DIR).
    """
    out_path = os.path.join(PILL_UTILS_DATA_DIR, 'train_cate_seq.txt')
    with open(out_path, 'w') as f:
        for cate_seq in cate_list:
            f.write(','.join(cate_seq) + '\n')

def train_cate_seq_2_format_integrated():
    """
    Convierte train_cate_seq.txt a train_cate_seq_format.csv (ambos en PILL_UTILS_DATA_DIR).
    """
    in_path = os.path.join(PILL_UTILS_DATA_DIR, 'train_cate_seq.txt')
    out_path = os.path.join(PILL_UTILS_DATA_DIR, 'train_cate_seq_format.csv')
    with open(in_path, 'r') as f:
        lines = f.readlines()
    with open(out_path, 'w') as f:
        for line in lines:
            cate_seq = line.strip().split(',')
            for cate in cate_seq:
                f.write(f"{cate},")
            f.write('\n')

def update_category_id_2_train_integrated():
    """
    Genera el archivo oid2nid_category.csv y new_categoryId_seq.txt.
    Lee:
        - train_cate_seq_format.csv (en PILL_UTILS_DATA_DIR)
    Escribe:
        - oid2nid_category.csv (OID_NID_CATEGORY_CSV)
        - new_categoryId_seq.txt (en PILL_UTILS_DATA_DIR)
    """
    in_path = os.path.join(PILL_UTILS_DATA_DIR, 'train_cate_seq_format.csv')
    out_path = OID_NID_CATEGORY_CSV
    new_cateid_seq_path = os.path.join(PILL_UTILS_DATA_DIR, 'new_categoryId_seq.txt')
    cate_set = set()
    cate_map = {}
    cate_cnt = 0
    with open(in_path, 'r') as f:
        lines = f.readlines()
    # Mapear categorías a nuevos IDs
    for line in lines:
        cates = line.strip().split(',')
        for cate in cates:
            if cate and cate not in cate_map:
                cate_map[cate] = str(cate_cnt)
                cate_cnt += 1
    # Escribir oid2nid_category.csv
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for k, v in cate_map.items():
            writer.writerow([k, v])
    # Escribir new_categoryId_seq.txt
    with open(new_cateid_seq_path, 'w') as f:
        for line in lines:
            cates = line.strip().split(',')
            new_cates = [cate_map.get(cate, '0') for cate in cates if cate]
            f.write(','.join(new_cates) + '\n')

def del_line_integrated():
    """
    Elimina líneas vacías de oid2nid_category.csv y escribe en new_oid2nid_category.csv.
    Lee:
        - oid2nid_category.csv (OID_NID_CATEGORY_CSV)
    Escribe:
        - new_oid2nid_category.csv (NEW_OID_NID_CATEGORY_CSV)
    """
    with open(OID_NID_CATEGORY_CSV, 'r') as f:
        lines = f.readlines()
    with open(NEW_OID_NID_CATEGORY_CSV, 'w') as f:
        for line in lines:
            if line.strip():
                f.write(line)

# --- FIN: Funciones integradas para generación de new_oid2nid_category.csv ---

# --- Funciones ya integradas para niid_2_ncid ---
def new_itemID_2_new_categoryID_integrated():
    nid2oid = {}
    cate_dict = {}
    oid2nid_category_dict = {}
    with open(DIGINETICA_PRODUCT_CATEGORIES, 'r') as cate_f:
        cate_lines = cate_f.readlines()
        for each_cate_line in cate_lines:
            cate_line_2_list = each_cate_line.split(';')
            if len(cate_line_2_list) > 1: 
                 cate_dict[cate_line_2_list[0]] = cate_line_2_list[1].strip()
    with open(NEW_OID_NID_CSV, 'r') as o2nid_f: # Usa la ruta corregida
        oid2nid_lines = o2nid_f.readlines()
        for each_id_line in oid2nid_lines:
            id_line_2_list = each_id_line.split(',')
            if len(id_line_2_list) > 1:
                nid2oid[id_line_2_list[1].strip()] = id_line_2_list[0]
    if not os.path.isfile(NEW_OID_NID_CATEGORY_CSV):
        print(f"ERROR CRÍTICO: El archivo de mapeo de categorías {NEW_OID_NID_CATEGORY_CSV} no existe y es requerido por new_itemID_2_new_categoryID_integrated.")
        print("Asegúrese de que la Fase 2 se complete correctamente o genere este archivo manualmente.")
        sys.exit(1)
    with open(NEW_OID_NID_CATEGORY_CSV, 'r') as o2nid_category_f:
        oid2nid_category_lines = o2nid_category_f.readlines()
        for each_line in oid2nid_category_lines:
            each_id_line_2_list = each_line.split(',')
            if len(each_id_line_2_list) > 1:
                oid2nid_category_dict[each_id_line_2_list[0]] = each_id_line_2_list[1].strip()
    train_txt_path = os.path.join(DIGINETICA_PROCESSED_DIR, 'train.txt')
    with open(train_txt_path, 'r') as seq_f:
        lines = seq_f.readlines()
        record_nid = []
        with open(NIID_NCID_TXT_SRC, 'w') as f: 
            for no, line in enumerate(lines, 1):
                if no % 10000 == 0:
                    print(f"Procesando new_itemID_2_new_categoryID: 共{len(lines)}行, 正在处理第{no}行")
                line_2_id_list = line.strip().split(',')
                for each_id in line_2_id_list:
                    current_nid = each_id.strip()
                    if not current_nid: continue 
                    if current_nid in record_nid:
                        continue
                    record_nid.append(current_nid)
                    get_oid = nid2oid.get(current_nid)
                    if get_oid is None: continue
                    get_category_id = cate_dict.get(get_oid)
                    if get_category_id is None: continue 
                    get_n_category_id = oid2nid_category_dict.get(get_category_id)
                    if get_n_category_id is None: continue 
                    content = f"{current_nid},{get_n_category_id}"
                    f.write(content + '\n')

# --- FIN de funciones para niid_2_ncid ---

# # --- Función de generación de niid_2_priceid eliminada en v2 ---
# def item_price_seq_integrated(n_ranges=5, discretization_method='quantiles'):
#     """
#     Esta función ha sido eliminada en la versión v2 del script.
#     La generación de niid_2_priceid.txt ahora es responsabilidad de PILL/preprocess.py.
#     """
#     pass

def ensure_dirs():
    os.makedirs(DIGINETICA_PROCESSED_DIR, exist_ok=True)
    os.makedirs(PILL_UTILS_DATA_DIR, exist_ok=True)

def fase_0_verificar_archivos_fuente():
    print("Fase 0: Verificando archivos fuente de Diginetica...")
    rutas_fuente = [
        DIGINETICA_TRAIN_VIEWS,
        DIGINETICA_TRAIN_PURCHASES,
        DIGINETICA_PRODUCT_CATEGORIES,
        DIGINETICA_PRODUCTS,
    ]
    faltantes = [ruta for ruta in rutas_fuente if not os.path.isfile(ruta)]
    if faltantes:
        print("ERROR: Faltan los siguientes archivos fuente de Diginetica:")
        for f in faltantes:
            print(f"  - {f}")
        sys.exit(1)
    print("Todos los archivos fuente están presentes.")

def fase_1_preparacion_oid_nid():
    print(f"Fase 1: Preparación para OID-NID y conteo de ítems.")
    print(f"NOTA: {NEW_OID_NID_CSV} y {NUM_ITEMS_TXT} serán generados por la Fase 3.")
    print("Fase 1 completada (informativa).")

def fase_2_generar_category_maps():
    print(f"Fase 2: Generando mapeos de categorías ({NEW_OID_NID_CATEGORY_CSV}) ...")
    # Verificar archivos de entrada críticos
    if not os.path.isfile(DIGINETICA_PRODUCT_CATEGORIES):
        print(f"ERROR: No se encuentra {DIGINETICA_PRODUCT_CATEGORIES}")
        sys.exit(1)
    if not os.path.isfile(NEW_OID_NID_CSV):
        print(f"ERROR: No se encuentra {NEW_OID_NID_CSV} (debe ser generado en Fase 3)")
        sys.exit(1)
    train_txt_path = os.path.join(DIGINETICA_PROCESSED_DIR, 'train.txt')
    if not os.path.isfile(train_txt_path):
        print(f"ERROR: No se encuentra {train_txt_path} (debe ser generado en Fase 3)")
        sys.exit(1)
    # Secuencia de funciones integradas
    print("  Ejecutando itemid_2_categoryid_fast_integrated ...")
    itemid_2_categoryid_fast_integrated()
    print("  Ejecutando train_cate_seq_2_format_integrated ...")
    train_cate_seq_2_format_integrated()
    print("  Ejecutando update_category_id_2_train_integrated ...")
    update_category_id_2_train_integrated()
    print("  Ejecutando del_line_integrated ...")
    del_line_integrated()
    print(f"Fase 2 completada. Archivo generado: {NEW_OID_NID_CATEGORY_CSV}")

def fase_3_ejecutar_preprocess_py(selected_discretization_method):
    """
    Ejecuta el script PILL/preprocess.py para generar los archivos procesados principales,
    incluyendo la discretización de precios y la generación de niid_2_priceid.txt.
    """
    print(f"Fase 3: Ejecutando PILL/preprocess.py para generar archivos procesados, incluyendo discretización de precios y niid_2_priceid.txt ...")
    preprocess_script_path = os.path.join(PILL_DIR, 'preprocess.py')
    cmd = [
        sys.executable, preprocess_script_path,
        '--dataset', 'diginetica',
        '--filepath', DIGINETICA_TRAIN_VIEWS, 
        '--dataset-dir', DIGINETICA_PROCESSED_DIR,
        '--discretization_method', selected_discretization_method,
        '--num_price_ranges', str(DEFAULT_NUM_PRICE_RANGES)
    ]
    try:
        print(f"Ejecutando comando: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=BASE_DIR)
        if result.stderr:
            print(f"STDERR de preprocess.py:\n{result.stderr}")
        print("Fase 3 completada exitosamente.")
    except subprocess.CalledProcessError as e:
        print(f"ERROR en Fase 3 al ejecutar {preprocess_script_path}:")
        print(f"Comando: {' '.join(e.cmd)}")
        print(f"Código de retorno: {e.returncode}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        sys.exit(1)

def fase_4_generar_niid2ncid():
    """
    Genera el archivo niid_2_ncid.txt a partir de los archivos procesados.
    La generación de niid_2_priceid.txt ahora es responsabilidad de PILL/preprocess.py.
    """
    print(f"Fase 4: Generando {NIID_NCID_TXT_SRC} ...")
    if not os.path.isfile(NEW_OID_NID_CSV): # Verifica el archivo generado por Fase 3
        print(f"ERROR CRÍTICO: El archivo de mapeo OID-NID {NEW_OID_NID_CSV} no existe. Debió ser generado por la Fase 3.")
        sys.exit(1)
    train_txt_path = os.path.join(DIGINETICA_PROCESSED_DIR, 'train.txt')
    if not os.path.isfile(train_txt_path):
        print(f"ERROR CRÍTICO: El archivo {train_txt_path} no existe. Debió ser generado por la Fase 3.")
        sys.exit(1)
    # La verificación de NEW_OID_NID_CATEGORY_CSV está dentro de new_itemID_2_new_categoryID_integrated
    print(f"  Llamando a new_itemID_2_new_categoryID_integrated (requiere {NEW_OID_NID_CATEGORY_CSV})...")
    new_itemID_2_new_categoryID_integrated()
    print("Fase 4 completada exitosamente.")

def fase_5_mover_archivos():
    print(f"Fase 5: Moviendo archivos de características a {DIGINETICA_PROCESSED_DIR} ...")
    archivos_a_mover = {
        # NIID_NCID_TXT_SRC: NIID_NCID_TXT_DEST,  # Se comenta para que niid_2_ncid.txt permanezca en PILL/utils/data/
        # NIID_PRICEID_TXT_SRC: NIID_PRICEID_TXT_DEST,  # Se comenta para que niid_2_priceid.txt permanezca en PILL/utils/data/
    }
    for src_file, dst_file in archivos_a_mover.items():
        try:
            if os.path.isfile(src_file):
                shutil.move(src_file, dst_file)
                print(f"  Movido: {src_file} -> {dst_file}")
            else:
                print(f"  ADVERTENCIA: Archivo fuente no encontrado para mover: {src_file}.")
        except Exception as e:
            print(f"ERROR al mover {src_file} a {dst_file}: {e}")
    print("Fase 5 completada.")

# El resto del script (main, argumentos, etc.) debe ser adaptado para usar las nuevas firmas de funciones.
# Asegúrate de actualizar la llamada a fase_3_ejecutar_preprocess_py para pasar el método de discretización deseado,
# y la llamada a fase_4_generar_niid2ncid() (sin parámetro de discretización).

# --- INICIO: Función para generación de archivos adicionales estructurados (portada de preprocess_diginetica_pasbr.py) ---

def generar_archivos_adicionales_diginetica_pricelevel2(discretization_method_param='quantiles'):
    """
    Genera archivos adicionales estructurados para pricelevel_2 en un nuevo directorio bajo BASE_DIR.
    Archivos generados:
        - train.pkl, test.pkl
        - mapping_dicts.pkl
        - item_mapping_table_with_sequences.csv
        - categoryID_to_category_mapping.csv
        - itemID_numeric_to_originalItemID_mapping.csv
        - priceLevel_summary.csv
        - structured_train_sessions.csv
        - structured_test_sessions.csv
    """
    print(f"Fase adicional: Generando archivos estructurados para pricelevel_{DEFAULT_NUM_PRICE_RANGES} usando discretización '{discretization_method_param}'...")

    # 1. Crear directorio destino si no existe
    output_dir = os.path.join(BASE_DIR, f"diginetica_pasbr_pricelevel_{DEFAULT_NUM_PRICE_RANGES}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Definición de rutas de archivos de entrada (generados por fases anteriores)
    train_txt = os.path.join(DIGINETICA_PROCESSED_DIR, 'train.txt')
    test_txt = os.path.join(DIGINETICA_PROCESSED_DIR, 'test.txt')
    oid2nid_csv = os.path.join(DIGINETICA_PROCESSED_DIR, 'oid2nid.csv')
    niid_2_ncid_txt = os.path.join(PILL_UTILS_DATA_DIR, 'niid_2_ncid.txt')
    niid_2_priceid_txt = os.path.join(PILL_UTILS_DATA_DIR, 'niid_2_priceid.txt')
    product_categories_csv = os.path.join(DIGINETICA_RAW_DIR, 'product-categories.csv')
    products_csv = os.path.join(DIGINETICA_RAW_DIR, 'products.csv')
    new_oid_nid_category_path = os.path.join(PILL_UTILS_DATA_DIR, 'new_oid2nid_category.csv')

    # Verificar existencia de archivos de entrada críticos
    critical_files = [train_txt, test_txt, oid2nid_csv, niid_2_ncid_txt, niid_2_priceid_txt,
                      product_categories_csv, products_csv, new_oid_nid_category_path]
    for f_path in critical_files:
        if not os.path.isfile(f_path):
            print(f"ERROR CRÍTICO para generar_archivos_adicionales: Falta el archivo {f_path}")
            return

    # 3. Carga de datos y mapeos
    print("  Cargando mapeos y datos...")
    oid_to_nid = pd.read_csv(oid2nid_csv, header=None, names=['original_itemID', 'numeric_itemID']).set_index('original_itemID')['numeric_itemID'].astype(str).to_dict()
    nid_to_oid = {v: k for k, v in oid_to_nid.items()}
    nid_to_ncid = pd.read_csv(niid_2_ncid_txt, header=None, names=['numeric_itemID', 'numeric_categoryID']).set_index('numeric_itemID')['numeric_categoryID'].astype(str).to_dict()
    nid_to_pid = pd.read_csv(niid_2_priceid_txt, header=None, names=['numeric_itemID', 'price_levelID']).set_index('numeric_itemID')['price_levelID'].astype(str).to_dict()
    ocid_to_ncid = pd.read_csv(new_oid_nid_category_path, header=None, names=['original_categoryID', 'numeric_categoryID']).set_index('original_categoryID')['numeric_categoryID'].astype(str).to_dict()
    ncid_to_ocid = {v: k for k, v in ocid_to_ncid.items()}
    oid_to_ocid = pd.read_csv(product_categories_csv, delimiter=';', header=None, names=['original_itemID', 'original_categoryID'], usecols=[0,1]).set_index('original_itemID')['original_categoryID'].astype(str).to_dict()
    oid_to_oprice = pd.read_csv(products_csv, delimiter=';', header=None, names=['original_itemID', 'original_price'], usecols=[0,1]).set_index('original_itemID')['original_price'].astype(str).to_dict()

    # 4. Carga de secuencias de entrenamiento y test
    def load_sequences(file_path):
        sequences = []
        with open(file_path, 'r') as f:
            for line in f:
                sequences.append(line.strip().split(','))
        return sequences

    print("  Cargando secuencias de train y test...")
    train_item_seqs = load_sequences(train_txt)
    test_item_seqs = load_sequences(test_txt)

    # 5. Construcción de secuencias de categorías y precios
    def construir_secuencias_aux(seqs, nid_to_feature_map, default_value='0'):
        feature_seqs = []
        for item_seq in seqs:
            current_feature_seq = []
            for nid in item_seq:
                feature = nid_to_feature_map.get(nid, default_value)
                current_feature_seq.append(str(feature))
            feature_seqs.append(current_feature_seq)
        return feature_seqs

    print("  Construyendo secuencias de categorías y precios...")
    train_cat_seqs = construir_secuencias_aux(train_item_seqs, nid_to_ncid)
    test_cat_seqs = construir_secuencias_aux(test_item_seqs, nid_to_ncid)
    train_price_seqs = construir_secuencias_aux(train_item_seqs, nid_to_pid)
    test_price_seqs = construir_secuencias_aux(test_item_seqs, nid_to_pid)

    # 6. Formateo para archivos .pkl (listas de tuplas de listas)
    def format_for_pkl(item_seqs, cat_seqs, price_seqs):
        combined_seqs = []
        for i in range(len(item_seqs)):
            combined_seqs.append(
                (item_seqs[i], cat_seqs[i], price_seqs[i])
            )
        return combined_seqs

    print("  Formateando datos para archivos .pkl...")
    train_data_pkl = format_for_pkl(train_item_seqs, train_cat_seqs, train_price_seqs)
    test_data_pkl = format_for_pkl(test_item_seqs, test_cat_seqs, test_price_seqs)

    # 7. Guardar archivos .pkl
    print("  Guardando archivos train.pkl y test.pkl...")
    with open(os.path.join(output_dir, 'train.pkl'), 'wb') as f_train:
        pickle.dump(train_data_pkl, f_train)
    with open(os.path.join(output_dir, 'test.pkl'), 'wb') as f_test:
        pickle.dump(test_data_pkl, f_test)

    # 8. Crear y guardar mapping_dicts.pkl
    print("  Creando y guardando mapping_dicts.pkl...")
    all_nids = set(nid_to_oid.keys())
    all_ncids = set(nid_to_ncid.values())
    all_pids = set(nid_to_pid.values())

    mapping_data = {
        'nid_to_oid': nid_to_oid,
        'oid_to_nid': oid_to_nid,
        'nid_to_ncid': nid_to_ncid,
        'nid_to_pid': nid_to_pid,
        'ncid_to_ocid': ncid_to_ocid,
        'ocid_to_ncid': ocid_to_ncid,
        'num_items_numeric': len(all_nids),
        'num_categories_numeric': len(all_ncids),
        'num_price_levels': len(all_pids),
        'discretization_method_used': discretization_method_param,
        'num_price_ranges_config': DEFAULT_NUM_PRICE_RANGES
    }
    with open(os.path.join(output_dir, 'mapping_dicts.pkl'), 'wb') as f_map:
        pickle.dump(mapping_data, f_map)

    # 9. Crear y guardar tablas de mapeo adicionales en CSV
    print("  Creando tablas de mapeo CSV adicionales...")
    item_mapping_list = []
    for nid in sorted(all_nids, key=lambda x: int(x) if x.isdigit() else x):
        oid = nid_to_oid.get(nid)
        ncid = nid_to_ncid.get(nid, 'N/A')
        ocid = ncid_to_ocid.get(ncid, 'N/A') if ncid != 'N/A' else oid_to_ocid.get(oid, 'N/A')
        pid = nid_to_pid.get(nid, 'N/A')
        oprice = oid_to_oprice.get(oid, 'N/A')
        item_mapping_list.append([nid, oid, ncid, ocid, pid, oprice])
    
    item_mapping_df = pd.DataFrame(item_mapping_list, columns=['numeric_itemID', 'original_itemID', 'numeric_categoryID', 'original_categoryID', 'price_levelID', 'original_price'])
    item_mapping_df.to_csv(os.path.join(output_dir, 'item_mapping_table_with_sequences.csv'), index=False)

    category_mapping_list = []
    for ncid in sorted(all_ncids, key=lambda x: int(x) if x.isdigit() else x):
        ocid = ncid_to_ocid.get(ncid, 'N/A')
        category_mapping_list.append([ncid, ocid])
    category_mapping_df = pd.DataFrame(category_mapping_list, columns=['numeric_categoryID', 'original_categoryID'])
    category_mapping_df.to_csv(os.path.join(output_dir, 'categoryID_to_category_mapping.csv'), index=False)
    
    nid_oid_list = [[nid, nid_to_oid.get(nid)] for nid in sorted(all_nids, key=lambda x: int(x) if x.isdigit() else x)]
    nid_oid_df = pd.DataFrame(nid_oid_list, columns=['numeric_itemID', 'original_itemID'])
    nid_oid_df.to_csv(os.path.join(output_dir, 'itemID_numeric_to_originalItemID_mapping.csv'), index=False)

    price_level_counts = pd.Series(nid_to_pid.values()).value_counts().reset_index()
    price_level_counts.columns = ['price_levelID', 'item_count']
    price_level_counts = price_level_counts.sort_values(by='price_levelID', key=lambda x: pd.to_numeric(x, errors='coerce'))
    price_level_counts.to_csv(os.path.join(output_dir, 'priceLevel_summary.csv'), index=False)

    # 10. Guardar sesiones estructuradas en CSV (opcional, para inspección)
    def guardar_structured_sessions_csv(seqs, price_seqs, cat_seqs, path):
        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['session_id', 'item_sequence', 'price_level_sequence', 'category_sequence'])
            for i, (item_s, price_s, cat_s) in enumerate(zip(seqs, price_seqs, cat_seqs)):
                writer.writerow([i, ','.join(item_s), ','.join(price_s), ','.join(cat_s)])

    print("  Guardando sesiones estructuradas en CSV (para inspección)...")
    guardar_structured_sessions_csv(train_item_seqs, train_price_seqs, train_cat_seqs, os.path.join(output_dir, 'structured_train_sessions.csv'))
    guardar_structured_sessions_csv(test_item_seqs, test_price_seqs, test_cat_seqs, os.path.join(output_dir, 'structured_test_sessions.csv'))

    print(f"Fase adicional completada. Archivos generados en: {output_dir}")

# --- FIN: Función para generación de archivos adicionales estructurados ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocesamiento Diginetica PASBR v2 con discretización de precios delegada a PILL/preprocess.py")
    parser.add_argument('--discretization_method', type=str, default='quantiles', help='Método de discretización de precios (quantiles, equal_width, logistic, custom_logistic, custom_logistic_per_category)')
    args = parser.parse_args()

    ensure_dirs()
    fase_0_verificar_archivos_fuente()
    fase_1_preparacion_oid_nid()
    fase_3_ejecutar_preprocess_py(args.discretization_method)
    fase_2_generar_category_maps()
    fase_4_generar_niid2ncid()
    fase_5_mover_archivos()
    # Llamada a la nueva fase adicional
    generar_archivos_adicionales_diginetica_pricelevel2(args.discretization_method)