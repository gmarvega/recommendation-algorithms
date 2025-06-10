import torch as th
import pandas as pd
import numpy as np
from datetime import datetime
import json
import math
from pathlib import Path


class RecommendationSaver:
    """A class to handle saving top-k recommendations from PASBR model"""

    def __init__(self, save_dir="recommendations", dataset_dir=None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load price mapping based on dataset
        self.price_mapping = None
        if dataset_dir:
            if 'yoochoose' in dataset_dir:
                df_price = pd.read_csv("./PILL/utils/data/yoochoose_process/input_data/renew_yoo_niid_2_priceid_dispersed_50.txt", 
                                     delimiter=',', names=['iid', 'pid'])
            elif 'amazon' in dataset_dir:
                df_price = pd.read_csv("./datasets/amazon/helpprice.txt", 
                                     delimiter=',', names=['iid', 'pid'])
            elif 'yelp' in dataset_dir:
                df_price = pd.read_csv("./PILL/datasets/yelp/yelp_nid_to_price.csv", 
                                     delimiter=',', names=['iid', 'pid'])
            else:
                df_price = pd.read_csv("./PILL/utils/data/niid_2_priceid.txt", 
                                     delimiter=',', names=['iid', 'pid'])
            
            self.price_mapping = dict(zip(df_price.iid, df_price.pid))

    def get_and_save_recommendations(
            self,
            model,
            data_loader,
            device,
            cutoffs=[10, 20],
            save_format='csv',
            include_scores=True,
            include_metadata=True,
    ):
        """
        Genera y guarda recomendaciones para múltiples valores de cutoff.
        
        Args:
            model: Modelo PASBR a evaluar
            data_loader: DataLoader con datos de prueba
            device: Dispositivo (CPU/GPU)
            cutoffs: Lista de valores de cutoff para generar recomendaciones
            save_format: Formato para guardar ('csv' o 'json')
            include_scores: Si se incluyen los scores de las predicciones
            include_metadata: Si se incluyen metadatos adicionales
            
        Returns:
            Tuple con diccionario de DataFrames y diccionario de metadatos para cada cutoff
        """
        model.eval()
        # Obtener el cutoff máximo para calcular predicciones una sola vez
        max_cutoff = max(cutoffs)
        
        # Resultados por cutoff
        all_results = {}
        for cutoff in cutoffs:
            all_results[cutoff] = {
                'session_ids': [],
                'true_items': [],
                'recommended_items': [],
                'scores': [] if include_scores else None,
                'prices': []
            }
        
        # Contadores para métricas por cutoff
        metrics_counters = {}
        for cutoff in cutoffs:
            metrics_counters[cutoff] = {
                'mrr': 0,
                'hit': 0,
                'num_samples': 0
            }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        with th.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                inputs, labels = self._prepare_batch(batch, device)
                logits = model(*inputs)
                batch_size = logits.size(0)
                
                # Calcular top-K para el máximo cutoff
                scores_max, predictions_max = logits.topk(k=max_cutoff)
                labels = labels.unsqueeze(-1)
                
                # Sesiones IDs para este batch
                session_ids = [f"session_{batch_idx}_{i}" for i in range(len(labels))]
                
                # Procesar cada cutoff
                for cutoff in cutoffs:
                    # Usar las primeras 'cutoff' predicciones para este cutoff
                    predictions = predictions_max[:, :cutoff].clone()
                    scores = scores_max[:, :cutoff].clone()
                    
                    # Calcular métricas para este cutoff
                    hit_ranks = th.where(predictions == labels)[1] + 1
                    metrics_counters[cutoff]['hit'] += hit_ranks.numel()
                    metrics_counters[cutoff]['mrr'] += hit_ranks.float().reciprocal().sum().item()
                    metrics_counters[cutoff]['num_samples'] += batch_size
                    
                    # Convertir a numpy para procesamiento
                    np_predictions = predictions.cpu().numpy()
                    np_scores = scores.cpu().numpy()
                    np_labels = labels.cpu().numpy()
                    
                    # Obtener precios para los ítems recomendados
                    batch_prices = []
                    for pred_items in np_predictions:
                        item_prices = [self.price_mapping.get(str(item_idx), -1) for item_idx in pred_items]
                        batch_prices.append(item_prices)
                    
                    # Guardar resultados para este cutoff
                    all_results[cutoff]['session_ids'].extend(session_ids)
                    all_results[cutoff]['true_items'].extend(np_labels.flatten())
                    all_results[cutoff]['recommended_items'].extend(np_predictions.tolist())
                    all_results[cutoff]['prices'].extend(batch_prices)
                    if include_scores:
                        all_results[cutoff]['scores'].extend(np_scores.tolist())

        # Calcular métricas finales y guardar resultados para cada cutoff
        all_final_metrics = {}
        all_dataframes = {}
        
        for cutoff in cutoffs:
            # Calcular métricas finales
            metrics = self._calculate_metrics(all_results[cutoff], metrics_counters[cutoff], cutoff)
            all_final_metrics[cutoff] = metrics
            
            # Guardar resultados en archivo
            df, metadata = self._save_cutoff_results(
                all_results[cutoff],
                metrics,
                cutoff,
                timestamp,
                save_format,
                include_scores,
                include_metadata
            )
            
            all_dataframes[cutoff] = df
            
            # Imprimir métricas
            print(f"\nMétricas finales para K@{cutoff}:")
            print(f"  MRR@{cutoff}: {metrics['mrr']:.4f}, HR@{cutoff}: {metrics['hit_rate']:.4f}, P@{cutoff}: {metrics['precision']:.4f}")
            print(f"  NDCG@{cutoff}: {metrics['ndcg']:.4f}, MAP@{cutoff}: {metrics['map']:.4f}, F1@{cutoff}: {metrics['f1']:.4f}")
            print(f"  Recall@{cutoff}: {metrics['recall']:.4f}, Diversity@{cutoff}: {metrics['diversity']:.4f}")

        # Guardar adicionalmente en el nuevo formato solicitado
        for cutoff in cutoffs:
            self._save_target_format_json(
                all_results[cutoff]['true_items'],
                all_results[cutoff]['recommended_items'],
                cutoff,
                timestamp
            )

        return all_dataframes, all_final_metrics
        
    def _calculate_metrics(self, results, counters, cutoff):
        """Calcula métricas finales para un cutoff específico."""
        num_samples = counters['num_samples']
        
        if num_samples == 0:
            return {
                'mrr': 0, 'hit_rate': 0, 'precision': 0, 'recall': 0,
                'ndcg': 0, 'map': 0, 'diversity': 0, 'f1': 0
            }
        
        # Hit Ratio@K / Recall@K
        hit_rate = counters['hit'] / num_samples
        recall = hit_rate  # Son iguales para un solo ítem relevante
        
        # MRR@K (Mean Reciprocal Rank)
        mrr_score = counters['mrr'] / num_samples
        
        # Precision@K
        precision = counters['hit'] / (num_samples * cutoff)
        
        # NDCG@K (aproximación)
        ideal_dcg = 1.0  # 1/log2(1+1) = 1, ideal para un ítem relevante en posición 1
        approx_dcg = 1.0 / (1 + (1/mrr_score - 1) * (1 - 1/math.log2(cutoff+1))) if mrr_score > 0 else 0
        ndcg = approx_dcg / ideal_dcg if ideal_dcg > 0 else 0
        
        # MAP@K (igual a MRR@K para un solo ítem relevante)
        map_score = mrr_score
        
        # Diversity@K
        items = results['recommended_items']
        unique_items = len(set([item for sublist in items for item in sublist]))
        total_items = sum(len(sublist) for sublist in items)
        diversity = unique_items / total_items if total_items > 0 else 0
        
        # F1-Score@K
        f1_score = 2 * hit_rate / (1 + cutoff) if (1 + cutoff) > 0 else 0
        
        return {
            'mrr': mrr_score,
            'hit_rate': hit_rate,
            'precision': precision,
            'recall': recall,
            'ndcg': ndcg,
            'map': map_score,
            'diversity': diversity,
            'f1': f1_score
        }
    
    def _save_cutoff_results(self, results, metrics, cutoff, timestamp, save_format, include_scores, include_metadata):
        """Guarda los resultados para un cutoff específico."""
        # Expandir resultados
        expanded_results = {
            'session_id': [],
            'true_item': [],
            'recommended_item': [],
            'price': [],
            'score': [] if include_scores else None
        }
        
        for i in range(len(results['session_ids'])):
            session_id = results['session_ids'][i]
            true_item = results['true_items'][i]
            rec_items = results['recommended_items'][i]
            prices = results['prices'][i]
            scores = results['scores'][i] if include_scores else None
            
            for j in range(len(rec_items)):
                expanded_results['session_id'].append(session_id)
                expanded_results['true_item'].append(true_item)
                expanded_results['recommended_item'].append(rec_items[j])
                expanded_results['price'].append(prices[j])
                if include_scores:
                    expanded_results['score'].append(scores[j])
        
        df = pd.DataFrame(expanded_results)
        
        metadata = {
            'timestamp': timestamp,
            'num_samples': len(results['session_ids']),
            'cutoff_k': cutoff,
            **metrics,
            'model_name': 'PILL'
        }
        
        base_filename = f"recommendations_{timestamp}_k{cutoff}"
        
        if save_format == 'csv':
            df.to_csv(self.save_dir / f"{base_filename}.csv", index=False)
            if include_metadata:
                with open(self.save_dir / f"{base_filename}_metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
        
        elif save_format == 'json':
            output = {
                'recommendations': df.to_dict(orient='records'),
                'metadata': metadata if include_metadata else None
            }
            with open(self.save_dir / f"{base_filename}.json", 'w') as f:
                json.dump(output, f, indent=2)
        
        print(f"Guardadas {len(df)} recomendaciones para K@{cutoff} en {self.save_dir}")
        
        return df, metadata

    def _save_target_format_json(self, true_items, recommended_items, cutoff, timestamp):
        """
        Guarda las recomendaciones en el nuevo formato solicitado:
        [
            {
                "target": <item real>,
                "recommended": [<lista de top-K recomendados>]
            },
            ...
        ]
        """
        data = []
        for target, recommended in zip(true_items, recommended_items):
            data.append({
                "target": int(target),
                "recommended": [int(item) for item in recommended]
            })
        base_filename = f"recommendations_{timestamp}_k{cutoff}_target_format.json"
        output_path = self.save_dir / base_filename
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Guardadas {len(data)} recomendaciones en formato target para K@{cutoff} en {output_path}")

    def _prepare_batch(self, batch, device):
        inputs, labels = batch
        inputs_gpu = [x.to(device) for x in inputs]
        labels_gpu = labels.to(device)
        return inputs_gpu, labels_gpu

    def load_recommendations(self, filename):
        file_path = self.save_dir / filename
        if filename.endswith('.csv'):
            return pd.read_csv(file_path)
        elif filename.endswith('.json'):
            with open(file_path) as f:
                data = json.load(f)
            return pd.DataFrame(data['recommendations']), data.get('metadata')
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON files.")
