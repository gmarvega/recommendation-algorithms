import time
import datetime
import torch as th
from torch import nn, optim
import os
import logging
import numpy as np
from collections import Counter

from .recommendation_saver import RecommendationSaver
from .metric_saver import MetricSaver
from .experiment_logger import ExperimentLogger

# Configurar el logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TrainRunner')

def calculate_gini_coefficient(values):
    """
    Calcula el coeficiente de Gini para una lista o array de números.

    El coeficiente de Gini es una medida de la desigualdad de una distribución.
    Varía entre 0 (igualdad perfecta) y 1 (desigualdad perfecta).

    Args:
        values (list or np.ndarray): Una lista o array de numpy con números
                                     (por ejemplo, frecuencias de aparición).

    Returns:
        float: El coeficiente de Gini. Devuelve 0.0 para una lista vacía,
               una lista con un solo elemento, o si todos los valores son cero.
               Devuelve np.nan si algún valor es negativo.
    """
    array = np.array(values, dtype=np.float64)
    if array.size == 0:
        return 0.0
    if np.any(array < 0):
        return np.nan
    if np.all(array == 0):
        return 0.0
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    numerator = np.sum((2 * index - n - 1) * array)
    denominator = (n * np.sum(array))
    if denominator == 0:
        return 0.0
    return numerator / denominator


# ignore weight decay for parameters in bias, batch norm and activation
def fix_weight_decay(model):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(map(lambda x: x in name, ['bias', 'batch_norm', 'activation'])):
            no_decay.append(param)
        else:
            decay.append(param)
    params = [{'params': decay}, {'params': no_decay, 'weight_decay': 0}]
    return params


def prepare_batch(batch, device):
    inputs, labels = batch
    inputs_gpu = [x.to(device) for x in inputs]
    labels_gpu = labels.to(device)
    return inputs_gpu, labels_gpu


def evaluate(model, data_loader, device, cutoffs=[10, 20]):
    """
    Evalúa el modelo con múltiples valores de cutoff.
    
    Args:
        model: Modelo a evaluar
        data_loader: DataLoader con datos de prueba
        device: Dispositivo (CPU/GPU)
        cutoffs: Lista de valores de cutoff para la evaluación (por defecto [10, 20])
        
    Returns:
        Dictionary con métricas para cada cutoff
    """
    model.eval()
    # Diccionario para almacenar métricas por cutoff
    all_metrics = {}
    
    # Obtener el cutoff máximo para calcular las predicciones una sola vez
    max_cutoff = max(cutoffs)
    
    # Inicializar contadores para cada cutoff
    num_samples = 0
    hit_per_cutoff = {k: 0 for k in cutoffs}
    mrr_per_cutoff = {k: 0 for k in cutoffs}
    dcg_sum_per_cutoff = {k: 0 for k in cutoffs}
    all_recommended_items_per_cutoff = {k: [] for k in cutoffs}
    
    # Procesar lotes de datos
    with th.no_grad():
        for batch in data_loader:
            inputs, labels = prepare_batch(batch, device)
            logits = model(*inputs)
            batch_size = logits.size(0)
            num_samples += batch_size
            
            # Obtener top-K predicciones con sus scores para el cutoff máximo
            scores, topk_max = logits.topk(k=max_cutoff)
            labels = labels.unsqueeze(-1)
            
            # Calcular métricas para cada cutoff
            for cutoff in cutoffs:
                # Usar solo las primeras 'cutoff' predicciones
                topk = topk_max[:, :cutoff]
                
                # Encontrar posiciones donde los ítems relevantes están en el top-K
                hit_ranks = th.where(topk == labels)[1] + 1
                hit_per_cutoff[cutoff] += hit_ranks.numel()
                
                # Calcular MRR (Mean Reciprocal Rank)
                mrr_per_cutoff[cutoff] += hit_ranks.float().reciprocal().sum().item()
                
                # Calcular NDCG
                # Crear tensor de relevancia (1 donde hay match, 0 en el resto)
                relevance = (topk == labels).float()
                # Calcular posición+1 para cada ítem en top-K
                position = th.arange(1, cutoff + 1, device=device).float().unsqueeze(0).expand_as(topk)
                # Calcular DCG: relevancia / log2(posición+1)
                dcg = (relevance / th.log2(position + 1)).sum(dim=1)
                dcg_sum_per_cutoff[cutoff] += dcg.sum().item()
                
                # Recolectar ítems para diversidad
                all_recommended_items_per_cutoff[cutoff].extend(topk.cpu().numpy().flatten().tolist())
    
    # Calcular métricas finales para cada cutoff
    for cutoff in cutoffs:
        # Hit Ratio@K / Recall@K (son iguales cuando hay un solo ítem relevante por sesión)
        hit_rate = hit_per_cutoff[cutoff] / num_samples
        
        # MRR@K (Mean Reciprocal Rank)
        mrr_score = mrr_per_cutoff[cutoff] / num_samples
        
        # Precision@K
        precision = hit_per_cutoff[cutoff] / (num_samples * cutoff)
        
        # NDCG@K (Normalized Discounted Cumulative Gain)
        # En este caso, IDCG es siempre 1/log2(1+1) = 1 para un solo ítem relevante
        ndcg = dcg_sum_per_cutoff[cutoff] / num_samples
        
        # MAP@K (Mean Average Precision)
        # Para un solo ítem relevante, MAP@K es igual a MRR@K
        map_score = mrr_score
        
        # Diversity@K (proporción de ítems únicos en las recomendaciones)
        items = all_recommended_items_per_cutoff[cutoff]
        unique_items = len(set(items))
        total_recommendations = len(items)
        diversity = unique_items / total_recommendations if total_recommendations > 0 else 0
        
        # Calcular Coeficiente de Gini
        items_for_gini = all_recommended_items_per_cutoff[cutoff]
        item_counts = Counter(items_for_gini)
        frequencies = list(item_counts.values())
        gini_coefficient = calculate_gini_coefficient(frequencies)
        
        # F1-Score@K (media armónica entre precision y recall)
        # Para un solo ítem relevante: F1@K = 2 * HR@K / (1 + K)
        f1_score = 2 * hit_rate / (1 + cutoff) if (1 + cutoff) > 0 else 0
        
        # Guardar métricas para este cutoff
        all_metrics[cutoff] = {
            'mrr': mrr_score,
            'hr': hit_rate,
            'precision': precision,
            'recall': hit_rate,  # Recall@K = HR@K con un solo ítem relevante
            'ndcg': ndcg,
            'map': map_score,
            'diversity': diversity,
            'gini': gini_coefficient,
            'f1': f1_score
        }
    
    return all_metrics


class TrainRunner:
    def __init__(
            self,
            model,
            train_loader,
            test_loader,
            device,
            lr=1e-3,
            weight_decay=0,
            patience=5,
            save_recommendations=True,
            recommendation_dir="recommendations",
            dataset_dir=None,
            save_metrics=True,
            results_dir="results",
            use_experiment_logger=True,
            config_name=None,
            args=None
    ):
        self.model = model
        if weight_decay > 0:
            params = fix_weight_decay(model)
        else:
            params = model.parameters()
        self.optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epoch = 0
        self.batch = 0
        self.patience = patience
        self.save_recommendations = save_recommendations
        self.recommendation_dir = recommendation_dir
        self.dataset_dir = dataset_dir
        
        # Inicializar el guardado de recomendaciones si está habilitado
        if save_recommendations:
            self.recommendation_saver = RecommendationSaver(save_dir=recommendation_dir, dataset_dir=dataset_dir)
        
        # Inicializar el sistema de registro y métricas
        self.save_metrics = save_metrics
        self.use_experiment_logger = use_experiment_logger
        
        # Usar el nuevo ExperimentLogger si está habilitado
        if use_experiment_logger and save_metrics:
            try:
                self.experiment_logger = ExperimentLogger(
                    dataset_dir=dataset_dir,
                    config_name=config_name,
                    args=args,
                    base_dir=results_dir
                )
                logger.info(f"ExperimentLogger inicializado. Experimento en {self.experiment_logger.get_experiment_path()}")
            except Exception as e:
                logger.error(f"Error al inicializar ExperimentLogger: {e}")
                self.use_experiment_logger = False
                # Intentar usar el MetricSaver como respaldo
                try:
                    self.metric_saver = MetricSaver(results_dir=results_dir, recommendation_dir=recommendation_dir)
                    logger.info(f"MetricSaver inicializado como respaldo. Guardando métricas en {results_dir}")
                except Exception as e:
                    logger.error(f"Error al inicializar MetricSaver: {e}")
                    self.save_metrics = False
        # Usar el sistema de métricas anterior si no se usa el nuevo logger
        elif save_metrics and not use_experiment_logger:
            try:
                self.metric_saver = MetricSaver(results_dir=results_dir, recommendation_dir=recommendation_dir)
                logger.info(f"MetricSaver inicializado. Guardando métricas en {results_dir}")
            except Exception as e:
                logger.error(f"Error al inicializar MetricSaver: {e}")
                self.save_metrics = False

    def train(self, epochs, log_interval=100, save_best=True, cutoffs=[10, 20]):
        """
        Entrena el modelo durante un número específico de épocas.
        
        Args:
            epochs: Número de épocas a entrenar.
            log_interval: Cada cuántos batches imprimir información.
            save_best: Si es True, guarda recomendaciones solo del mejor modelo.
            cutoffs: Lista de valores de cutoff para la evaluación.
            
        Returns:
            Dictionary con métricas máximas para cada cutoff.
        """
        # Inicializar valores máximos para cada métrica y cutoff
        max_metrics = {}
        for cutoff in cutoffs:
            max_metrics[cutoff] = {
                'mrr': 0, 'hr': 0, 'precision': 0, 'recall': 0,
                'ndcg': 0, 'map': 0, 'diversity': 0, 'gini': 0, 'f1': 0
            }
        
        # Usaremos K@20 como métrica principal para early stopping
        main_cutoff = max(cutoffs)
        
        bad_counter = 0
        t = time.time()
        mean_loss = 0
        best_epoch = 0

        for epoch in range(epochs):
            self.model.train()
            for batch in self.train_loader:
                inputs, labels = prepare_batch(batch, self.device)
                self.optimizer.zero_grad()
                logits = self.model(*inputs)
                # DEBUG: Imprimir formas y rangos antes de la pérdida
                #print(f"DEBUG TRAIN: logits.shape: {logits.shape}")
                #print(f"DEBUG TRAIN: labels.shape: {labels.shape}, labels.min: {labels.min()}, labels.max: {labels.max()}, labels.dtype: {labels.dtype}")
                loss = nn.functional.cross_entropy(logits, labels)
                loss.backward()
                self.optimizer.step()
                mean_loss += loss.item() / log_interval
                if self.batch > 0 and self.batch % log_interval == 0:
                    print(
                        f'Batch {self.batch}: Loss = {mean_loss:.4f}, Time Elapsed = {time.time() - t:.2f}s'
                    )
                    t = time.time()
                    mean_loss = 0
                self.batch += 1

            # Evaluar el modelo y obtener todas las métricas para cada cutoff
            all_metrics = evaluate(self.model, self.test_loader, self.device, cutoffs=cutoffs)
            
            # Imprimir métricas para cada cutoff
            print(f'Epoch {self.epoch}:')
            for cutoff in cutoffs:
                metrics = all_metrics[cutoff]
                print(f'  Métricas para K@{cutoff}:')
                print(f'    MRR@{cutoff} = {metrics["mrr"] * 100:.3f}%, HR@{cutoff} = {metrics["hr"] * 100:.3f}%, P@{cutoff} = {metrics["precision"] * 100:.3f}%')
                print(f'    NDCG@{cutoff} = {metrics["ndcg"] * 100:.3f}%, MAP@{cutoff} = {metrics["map"] * 100:.3f}%, F1@{cutoff} = {metrics["f1"] * 100:.3f}%')
                print(f'    Recall@{cutoff} = {metrics["recall"] * 100:.3f}%, Diversity@{cutoff} = {metrics["diversity"] * 100:.3f}%, Gini@{cutoff} = {metrics.get("gini", float("nan")) * 100:.3f}%')
            
            # Guardar métricas para esta epoch
            if self.save_metrics:
                # Usar el nuevo ExperimentLogger si está habilitado
                if self.use_experiment_logger:
                    try:
                        for cutoff, metrics in all_metrics.items():
                            self.experiment_logger.log_metrics(self.epoch, metrics, cutoff=cutoff)
                    except Exception as e:
                        logger.error(f"Error al guardar métricas con ExperimentLogger para epoch {self.epoch}: {e}")
                # Si no, usar el sistema de métricas anterior
                else:
                    try:
                        for cutoff, metrics in all_metrics.items():
                            # Añadir época y cutoff a las métricas
                            metrics_to_save = {**metrics, 'epoch': self.epoch, 'cutoff': cutoff}
                            self.metric_saver.save_metrics(**metrics_to_save)
                    except Exception as e:
                        logger.error(f"Error al guardar métricas para epoch {self.epoch}: {e}")

            # Comprobar early stopping basado en métricas principales (usando el cutoff máximo)
            metrics_main = all_metrics[main_cutoff]
            if metrics_main["mrr"] < max_metrics[main_cutoff]["mrr"] and metrics_main["hr"] < max_metrics[main_cutoff]["hr"] and metrics_main["precision"] < max_metrics[main_cutoff]["precision"]:
                bad_counter += 1
                if bad_counter == self.patience:
                    break
            else:
                if save_best and self.save_recommendations:
                    print("New best model found! Saving recommendations...")
                    self.recommendation_saver.get_and_save_recommendations(
                        self.model,
                        self.test_loader,
                        self.device,
                        cutoffs=cutoffs,
                        save_format='json',
                        include_scores=True,
                        include_metadata=True
                    )
                # Guardar solo el state_dict para compatibilidad con PyTorch 2.x
                th.save(self.model.state_dict(), 'params.pkl')
                # Si se requiere guardar la arquitectura, guardar el script o el código fuente.
                # Para cargar el modelo:
                #   model.load_state_dict(torch.load('params.pkl'))
                #   model.eval()
                bad_counter = 0
            
            # Actualizar valores máximos de todas las métricas para cada cutoff
            for cutoff in cutoffs:
                for key in max_metrics[cutoff]:
                    if key in all_metrics[cutoff]:
                        max_metrics[cutoff][key] = max(max_metrics[cutoff][key], all_metrics[cutoff][key])
            
            self.epoch += 1

        # Save final recommendations if requested
        if self.save_recommendations and not save_best:
            print("Saving final recommendations...")
            self.recommendation_saver.get_and_save_recommendations(
                self.model,
                self.test_loader,
                self.device,
                cutoffs=cutoffs,
                save_format='json',
                include_scores=True,
                include_metadata=True
            )
        
        # Guardar métricas finales usando ExperimentLogger
        if self.save_metrics and self.use_experiment_logger:
            try:
                self.experiment_logger.log_final_metrics(max_metrics)
                logger.info("Métricas finales guardadas en el directorio del experimento")
            except Exception as e:
                logger.error(f"Error al guardar métricas finales: {e}")

        return max_metrics
