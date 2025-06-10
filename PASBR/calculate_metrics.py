import json
import sys
import os
import math
import numpy as np
from collections import Counter

def main(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {json_path}.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: El archivo {json_path} no tiene un formato JSON válido.")
        sys.exit(1)

    if not isinstance(data, list):
        print("Error: El archivo JSON debe contener una lista de objetos.")
        sys.exit(1)

    total_targets = len(data)
    if total_targets == 0:
        print("Error: El archivo JSON no contiene datos.")
        sys.exit(1)

    aciertos = 0
    rr_sum = 0.0
    ndcg_sum = 0.0

    ap_sum = 0.0  # Para MAP@20
    all_recommended_items = []  # Para Gini Index

    for entry in data:
        target = entry.get("target")
        recommended = entry.get("recommended", [])
        if not isinstance(recommended, list):
            recommended = []

        all_recommended_items.extend(recommended)

        # Precisión@20 y MRR
        if target in recommended:
            aciertos += 1
            pos = recommended.index(target) + 1  # posiciones empiezan en 1
            rr_sum += 1.0 / pos
            ap_sum += 1.0 / pos  # AP para esta entrada
        else:
            rr_sum += 0.0
            ap_sum += 0.0  # AP para esta entrada

        # DCG@20
        dcg = 0.0
        for i in range(min(20, len(recommended))):
            relevancia = 1.0 if recommended[i] == target else 0.0
            dcg += relevancia / math.log2(i + 2)  # i+2 porque i empieza en 0 y log2(1+1)=1

        # IDCG@20
        if target in recommended:
            idcg = 1.0  # ya que log2(2) = 1
        else:
            idcg = 0.0

        # nDCG@20
        if idcg > 0:
            ndcg = dcg / idcg
        else:
            ndcg = 0.0
        ndcg_sum += ndcg

    precision_at_20 = aciertos / total_targets
    mrr = rr_sum / total_targets
    ndcg_at_20 = ndcg_sum / total_targets

    # Recall@20 (igual a precision_at_20)
    recall_at_20 = precision_at_20

    # MAP@20
    map_at_20 = ap_sum / total_targets

    # F1 Score@20 (igual a precision_at_20 en este contexto)
    f1_at_20 = precision_at_20

    # Gini Index
    if not all_recommended_items:
        gini_coefficient = 0.0
    else:
        counts = np.array(list(Counter(all_recommended_items).values()))
        sorted_counts = np.sort(counts)  # type: ignore
        N = len(sorted_counts)
        total_sum_of_counts = np.sum(sorted_counts)  # type: ignore
        if N <= 1 or total_sum_of_counts == 0:
            gini_coefficient = 0.0
        else:
            index_terms = 2 * np.arange(1, N + 1) - N - 1
            numerator = np.sum(index_terms * sorted_counts)
            denominator = N * total_sum_of_counts
            gini_coefficient = numerator / denominator

    # Diversidad Agregada (Ítems Únicos)
    if not all_recommended_items:
        diversidad_agregada = 0
    else:
        diversidad_agregada = len(set(all_recommended_items))
    print(f"Diversidad Agregada (Ítems Únicos): {diversidad_agregada}")

    print(f"Precisión@20: {precision_at_20:.4f}")
    print(f"Hit Rate@20: {precision_at_20:.4f}")
    print(f"Recall@20: {recall_at_20:.4f}")
    print(f"MAP@20: {map_at_20:.4f}")
    print(f"F1 Score@20: {f1_at_20:.4f}")
    print(f"Gini Index: {gini_coefficient:.4f}")
    print(f"MRR: {mrr:.4f}")
    print(f"nDCG@20: {ndcg_at_20:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python calculate_metrics.py <ruta_al_archivo_json>")
        sys.exit(1)
    json_path = sys.argv[1]
    main(json_path)