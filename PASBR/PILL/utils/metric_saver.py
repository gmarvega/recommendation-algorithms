import os
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime
import re
import logging

# Configurar el logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MetricSaver')

class MetricSaver:
    """
    Clase para guardar las métricas de evaluación (HR, MRR, Precision) 
    después de cada epoch de entrenamiento.
    """
    
    def __init__(self, results_dir="results", recommendation_dir=None):
        """
        Inicializa el MetricSaver.
        
        Args:
            results_dir (str): Directorio donde se guardarán los resultados de métricas.
            recommendation_dir (str): Directorio de recomendaciones para extraer timestamp y runID.
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directorio de resultados: {self.results_dir}")
        
        # Extraer timestamp y runID del directorio de recomendaciones
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = "default"
        self.dataset = "unknown"
        
        if recommendation_dir:
            try:
                # Patrón para extraer información del directorio de recomendaciones
                # Formato esperado: "recommendations/{dataset}_{config}_{timestamp}"
                pattern = r"(?:recommendations/)?([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_(\d{8}_\d{6})"
                match = re.search(pattern, recommendation_dir)
                
                if match:
                    self.dataset, self.run_id, self.timestamp = match.groups()
                    logger.info(f"Información extraída: dataset={self.dataset}, runID={self.run_id}, timestamp={self.timestamp}")
                else:
                    # Si no coincide con el patrón, asignar valores por defecto
                    logger.warning(f"No se pudo extraer información del directorio: {recommendation_dir}")
                    base_dir = os.path.basename(recommendation_dir)
                    self.run_id = base_dir if base_dir else "default"
            except Exception as e:
                logger.error(f"Error al extraer información del directorio: {e}")
        
        # Nombre del archivo de salida
        self.output_file = self.results_dir / f"{self.timestamp}_{self.run_id}_{self.dataset}.csv"
        logger.info(f"Archivo de métricas: {self.output_file}")
        
        # Crear archivo con encabezados si no existe
        if not self.output_file.exists():
            try:
                with open(self.output_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'timestamp', 'run_id', 'dataset', 'epoch',
                        'hr', 'mrr', 'precision', 'recall', 'ndcg', 'map', 'diversity', 'gini', 'f1',
                        'cutoff', 'time'
                    ])
                logger.info(f"Archivo de métricas creado: {self.output_file}")
            except Exception as e:
                logger.error(f"Error al crear archivo de métricas: {e}")
    
    def save_metrics(self, epoch, cutoff=20, timestamp=None, **metrics):
        """
        Guarda las métricas de evaluación en el archivo CSV.
        
        Args:
            epoch (int): Número de época actual.
            cutoff (int): Valor de cutoff utilizado en la evaluación.
            timestamp (str, optional): Timestamp de la evaluación. Si es None, se utiliza la hora actual.
            **metrics: Diccionario con todas las métricas a guardar. Debe incluir al menos:
                - hr: Hit Ratio
                - mrr: Mean Reciprocal Rank
                - precision: Precisión
                - recall: Recall
                - ndcg: Normalized Discounted Cumulative Gain
                - map: Mean Average Precision
                - diversity: Diversidad
                - gini: Coeficiente de Gini
                - f1: F1-Score
       
       Returns:
           bool: True si se guardaron las métricas correctamente, False en caso contrario.
       """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Preparar todos los valores a guardar (columnas)
            # Asegurarse de que el orden coincida con el encabezado del CSV
            metric_names_in_order = ['hr', 'mrr', 'precision', 'recall', 'ndcg', 'map', 'diversity', 'gini', 'f1']
            metric_values = []
            
            # Verificar que todas las métricas requeridas están presentes y obtener sus valores
            for metric_name in metric_names_in_order:
                metric_value = metrics.get(metric_name, float('nan')) # Usar NaN si falta alguna métrica
                if pd.isna(metric_value) and metric_name != 'gini': # Gini puede ser NaN por diseño
                    logger.warning(f"Métrica '{metric_name}' no proporcionada o es NaN. Usando float('nan') como valor predeterminado.")
                metric_values.append(metric_value)

            with open(self.output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, self.run_id, self.dataset, epoch,
                    *metric_values, cutoff, datetime.now().strftime("%H:%M:%S")
                ])
            
            # Resumen de las métricas más importantes
            # Formateo seguro para evitar errores con NaN/None
            def safe_fmt(val):
                try:
                    if pd.isna(val):
                        return "nan"
                    return f"{float(val):.4f}"
                except Exception:
                    return "nan"

            log_msg = (
                f"Métricas guardadas para época {epoch}: "
                f"HR={safe_fmt(metrics.get('hr', float('nan')))}, "
                f"MRR={safe_fmt(metrics.get('mrr', float('nan')))}, "
                f"P={safe_fmt(metrics.get('precision', float('nan')))}, "
                f"NDCG={safe_fmt(metrics.get('ndcg', float('nan')))}, "
                f"Gini={safe_fmt(metrics.get('gini', float('nan')))}, "
                f"F1={safe_fmt(metrics.get('f1', float('nan')))}"
            )
            logger.info(log_msg)
            
            return True
        except Exception as e:
            logger.error(f"Error al guardar métricas para época {epoch}: {e}")
            return False
    
    def get_metrics_history(self):
        """
        Obtiene el historial de métricas guardadas.
        
        Returns:
            pandas.DataFrame: DataFrame con el historial de métricas.
        """
        try:
            if self.output_file.exists():
                return pd.read_csv(self.output_file)
            else:
                logger.warning(f"El archivo de métricas no existe: {self.output_file}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error al leer archivo de métricas: {e}")
            return pd.DataFrame()