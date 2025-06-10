import os
import csv
import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

class ExperimentLogger:
    """
    Clase para gestionar el registro de experimentos, métricas y configuración 
    en una estructura organizada similar a fpar-CoHHN.
    
    Estructura de directorios:
    results/
      dataset_config_timestamp/
        logs/
          experiment.log
        metrics/
          metrics_per_epoch.csv
          final_metrics.csv
        config/
          params.json
    """
    
    def __init__(self, dataset_dir, config_name=None, args=None, base_dir="results"):
        """
        Inicializa el ExperimentLogger.
        
        Args:
            dataset_dir (str): Directorio del dataset, usado para extraer el nombre
            config_name (str, optional): Nombre de la configuración. Si es None, se usa "default"
            args (argparse.Namespace, optional): Argumentos de configuración a guardar
            base_dir (str): Directorio base para resultados
        """
        # Extraer nombre del dataset del directorio
        self.dataset = os.path.basename(dataset_dir) if dataset_dir else "unknown"
        self.config_name = config_name if config_name else "default"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Crear estructura de directorios
        self.base_dir = Path(base_dir)
        self.experiment_dir = self.base_dir / f"{self.dataset}_{self.config_name}_{self.timestamp}"
        self.logs_dir = self.experiment_dir / "logs"
        self.metrics_dir = self.experiment_dir / "metrics"
        self.config_dir = self.experiment_dir / "config"
        
        # Crear directorios
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.metrics_dir.mkdir(exist_ok=True)
        self.config_dir.mkdir(exist_ok=True)
        
        # Configurar logger
        self.logger = self._setup_logger()
        self.logger.info(f"Experimento inicializado: {self.experiment_dir}")
        
        # Guardar configuración si está disponible
        if args:
            self._save_config(args)
            self.logger.info("Configuración guardada.")
    
    def _setup_logger(self):
        """Configura el logger para escribir a archivo y consola."""
        logger = logging.getLogger(f"Experiment_{self.timestamp}")
        logger.setLevel(logging.INFO)
        
        # Evitar duplicación de handlers
        if not logger.handlers:
            # Handler para archivo
            # Convertir Path a str para compatibilidad con Python 3.10
            file_handler = logging.FileHandler(str(self.logs_dir / "experiment.log"))
            file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
            
            # Handler para consola
            console_handler = logging.StreamHandler()
            console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_format)
            logger.addHandler(console_handler)
        
        return logger
    
    def _save_config(self, args):
        """Guarda la configuración en formato JSON."""
        config_path = self.config_dir / "params.json"
        
        # Convertir argumentos a diccionario
        if hasattr(args, '__dict__'):
            config_dict = args.__dict__
        else:
            config_dict = args
        
        # Asegurar que todos los valores son serializables
        # Copiar para no modificar el original
        config_serializable = {}
        for key, value in config_dict.items():
            try:
                json.dumps(value)
                config_serializable[key] = value
            except (TypeError, OverflowError):
                config_serializable[key] = str(value)
        
        with open(config_path, 'w') as f:
            json.dump(config_serializable, f, indent=2)
    
    def log_metrics(self, epoch, metrics, cutoff=None):
        """
        Guarda las métricas de evaluación para una época específica.
        
        Args:
            epoch (int): Número de época
            metrics (dict): Diccionario con métricas
            cutoff (int, optional): Valor de cutoff. Si las métricas ya incluyen
                                   este valor, puede omitirse.
        
        Returns:
            bool: True si se guardaron las métricas correctamente
        """
        try:
            # Preparar el path del archivo
            metrics_path = self.metrics_dir / "metrics_per_epoch.csv"
            
            # Comprobar si necesitamos crear el archivo con encabezados
            create_header = not metrics_path.exists()
            
            # Preparar datos para guardar
            data = {'epoch': epoch}
            
            # Si tenemos un diccionario anidado por cutoff
            if isinstance(metrics, dict) and cutoff is not None and cutoff in metrics:
                metrics_dict = metrics[cutoff]
                data.update(metrics_dict)
                data['cutoff'] = cutoff
            # Si tenemos un diccionario plano de métricas
            elif isinstance(metrics, dict):
                data.update(metrics)
                if cutoff is not None:
                    data['cutoff'] = cutoff
            else:
                self.logger.error(f"Formato de métricas no reconocido: {type(metrics)}")
                return False
            
            # Añadir timestamp
            data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Escribir al archivo CSV
            with open(metrics_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(data.keys()))
                if create_header:
                    writer.writeheader()
                writer.writerow(data)
            
            # Log breve de las métricas principales
            log_msg = f"Métricas guardadas para época {epoch}"
            if 'mrr' in data:
                log_msg += f", MRR: {data['mrr']:.4f}"
            if 'hr' in data:
                log_msg += f", HR: {data['hr']:.4f}"
            if 'precision' in data:
                log_msg += f", P: {data['precision']:.4f}"
            if 'ndcg' in data:
                log_msg += f", NDCG: {data['ndcg']:.4f}"
            
            self.logger.info(log_msg)
            return True
            
        except Exception as e:
            self.logger.error(f"Error al guardar métricas: {e}")
            return False
    
    def log_final_metrics(self, best_metrics):
        """
        Guarda las métricas finales del experimento.
        
        Args:
            best_metrics (dict): Diccionario con las mejores métricas 
                                (puede estar anidado por cutoff)
        
        Returns:
            bool: True si se guardaron las métricas correctamente
        """
        try:
            # Preparar el path del archivo
            metrics_path = self.metrics_dir / "final_metrics.csv"
            
            # Si tenemos un diccionario anidado por cutoff
            if any(isinstance(value, dict) for value in best_metrics.values()):
                # Aplanar el diccionario anidado
                rows = []
                for cutoff, metrics in best_metrics.items():
                    row = {'cutoff': cutoff}
                    row.update(metrics)
                    rows.append(row)
                
                # Crear DataFrame y guardar
                df = pd.DataFrame(rows)
                df['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                df.to_csv(metrics_path, index=False)
                
                # Log de las métricas
                self.logger.info(f"Métricas finales guardadas para {len(rows)} valores de cutoff")
            else:
                # Es un diccionario plano
                data = best_metrics.copy()
                data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                with open(metrics_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(data.keys()))
                    writer.writeheader()
                    writer.writerow(data)
                
                self.logger.info("Métricas finales guardadas")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error al guardar métricas finales: {e}")
            return False
    
    def get_experiment_path(self):
        """Devuelve la ruta al directorio del experimento."""
        return self.experiment_dir
    
    def get_metrics_history(self):
        """
        Obtiene el historial de métricas guardadas.
        
        Returns:
            pandas.DataFrame: DataFrame con el historial de métricas o None si hay error
        """
        try:
            metrics_path = self.metrics_dir / "metrics_per_epoch.csv"
            if metrics_path.exists():
                return pd.read_csv(metrics_path)
            else:
                self.logger.warning(f"No hay métricas guardadas en {metrics_path}")
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error al leer métricas: {e}")
            return None