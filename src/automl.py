"""
AutoML Module
Automated hyperparameter tuning and model selection
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from itertools import product
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoML:
    """Automated Machine Learning for stock prediction"""
    
    def __init__(self, models_to_test: List[str] = None):
        """
        Initialize AutoML
        
        Args:
            models_to_test: List of model types to test 
                          ['lstm', 'attention_lstm', 'tcn', 'transformer', 'nbeats']
        """
        if models_to_test is None:
            self.models_to_test = ['lstm', 'attention_lstm', 'tcn', 'transformer']
        else:
            self.models_to_test = models_to_test
        
        self.results = []
        self.best_config = None
    
    def get_hyperparameter_grid(self, model_type: str) -> Dict:
        """Get hyperparameter search space for model type"""
        
        grids = {
            'lstm': {
                'lstm_units': [[32, 32], [64, 64], [128, 64], [64, 32]],
                'dropout_rate': [0.1, 0.2, 0.3],
                'learning_rate': [0.001, 0.0005, 0.0001],
                'batch_size': [32, 64]
            },
            'attention_lstm': {
                'lstm_units': [[64, 64], [128, 64], [64, 32]],
                'dropout_rate': [0.1, 0.2, 0.3],
                'learning_rate': [0.001, 0.0005],
                'batch_size': [32, 64]
            },
            'tcn': {
                'num_channels': [[32, 32, 32], [64, 64, 64], [32, 64, 128]],
                'kernel_size': [3, 5, 7],
                'dropout': [0.1, 0.2],
                'learning_rate': [0.001, 0.0005],
                'batch_size': [32, 64]
            },
            'transformer': {
                'd_model': [64, 128],
                'nhead': [4, 8],
                'num_layers': [2, 3, 4],
                'dropout': [0.1, 0.2],
                'learning_rate': [0.001, 0.0005],
                'batch_size': [32, 64]
            },
            'nbeats': {
                'stack_types': [
                    ('generic', 'generic'),
                    ('trend', 'seasonality'),
                ],
                'nb_blocks_per_stack': [2, 3],
                'hidden_layer_units': [128, 256],
                'learning_rate': [0.001, 0.0005],
                'batch_size': [32, 64]
            }
        }
        
        return grids.get(model_type, {})
    
    def random_search(self, model_type: str, n_iterations: int = 10) -> List[Dict]:
        """
        Random search over hyperparameter space
        
        Args:
            model_type: Type of model
            n_iterations: Number of random configurations to try
            
        Returns:
            List of sampled configurations
        """
        grid = self.get_hyperparameter_grid(model_type)
        configs = []
        
        for _ in range(n_iterations):
            config = {}
            for param, values in grid.items():
                config[param] = np.random.choice(values) if isinstance(values[0], (int, float)) else np.random.choice(len(values))
                if not isinstance(values[0], (int, float)):
                    config[param] = values[config[param]]
            configs.append(config)
        
        return configs
    
    def grid_search(self, model_type: str, max_combinations: int = 50) -> List[Dict]:
        """
        Grid search over hyperparameter space
        
        Args:
            model_type: Type of model
            max_combinations: Maximum number of combinations to try
            
        Returns:
            List of configurations to test
        """
        grid = self.get_hyperparameter_grid(model_type)
        
        # Generate all combinations
        keys = list(grid.keys())
        values = list(grid.values())
        
        all_combinations = list(product(*values))
        
        # Limit combinations if too many
        if len(all_combinations) > max_combinations:
            indices = np.random.choice(len(all_combinations), max_combinations, replace=False)
            all_combinations = [all_combinations[i] for i in indices]
        
        # Convert to list of dicts
        configs = []
        for combo in all_combinations:
            config = dict(zip(keys, combo))
            configs.append(config)
        
        return configs
    
    def evaluate_config(self, model, trainer, X_train, y_train, X_val, y_val, 
                       config: Dict, epochs: int = 50) -> Dict:
        """
        Evaluate a single configuration
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Update model with config
            for key, value in config.items():
                if hasattr(model, key):
                    setattr(model, key, value)
            
            # Train model
            history = trainer.train(
                X_train, y_train, X_val, y_val,
                epochs=epochs,
                batch_size=config.get('batch_size', 32),
                early_stopping=True,
                patience=10
            )
            
            # Evaluate
            val_loss = min(history['val_loss'])
            train_loss = history['train_loss'][history['val_loss'].index(val_loss)]
            
            # Additional metrics
            model.model.eval()
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_val)
                if torch.cuda.is_available():
                    X_val_tensor = X_val_tensor.cuda()
                
                predictions = model.model(X_val_tensor)
                if isinstance(predictions, tuple):
                    predictions = predictions[-1]
                
                predictions = predictions.cpu().numpy()
                
                # Calculate RMSE, MAE, MAPE
                from sklearn.metrics import mean_squared_error, mean_absolute_error
                
                rmse = np.sqrt(mean_squared_error(y_val, predictions))
                mae = mean_absolute_error(y_val, predictions)
                mape = np.mean(np.abs((y_val - predictions) / y_val)) * 100
            
            return {
                'config': config,
                'val_loss': val_loss,
                'train_loss': train_loss,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'overfitting': abs(val_loss - train_loss) / train_loss,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error evaluating config: {e}")
            return {
                'config': config,
                'val_loss': float('inf'),
                'success': False,
                'error': str(e)
            }
    
    def find_best_hyperparameters(self, model_class, trainer_class, 
                                 X_train, y_train, X_val, y_val,
                                 model_type: str,
                                 search_type: str = 'random',
                                 n_iterations: int = 20,
                                 epochs_per_trial: int = 30) -> Dict:
        """
        Find best hyperparameters for a model
        
        Args:
            model_class: Model class to instantiate
            trainer_class: Trainer class
            X_train, y_train: Training data
            X_val, y_val: Validation data
            model_type: Type of model
            search_type: 'random' or 'grid'
            n_iterations: Number of configurations to try
            epochs_per_trial: Epochs to train each configuration
            
        Returns:
            Best configuration and results
        """
        logger.info(f"Starting {search_type} search for {model_type}")
        
        # Get configurations to test
        if search_type == 'random':
            configs = self.random_search(model_type, n_iterations)
        else:
            configs = self.grid_search(model_type, n_iterations)
        
        results = []
        
        for i, config in enumerate(configs):
            logger.info(f"Testing configuration {i+1}/{len(configs)}: {config}")
            
            # Create model instance with config
            try:
                seq_length = X_train.shape[1]
                n_features = X_train.shape[2]
                
                # Instantiate model with configuration
                model_kwargs = {'seq_length': seq_length, 'n_features': n_features}
                model_kwargs.update(config)
                
                # Remove trainer-specific params
                model_kwargs.pop('batch_size', None)
                
                model = model_class(**model_kwargs)
                trainer = trainer_class(model)
                
                # Evaluate
                result = self.evaluate_config(
                    model, trainer, X_train, y_train, X_val, y_val,
                    config, epochs_per_trial
                )
                
                results.append(result)
                
                logger.info(f"Val Loss: {result['val_loss']:.6f}, "
                          f"RMSE: {result.get('rmse', 0):.6f}")
                
            except Exception as e:
                logger.error(f"Error with config {config}: {e}")
                continue
        
        # Find best configuration
        successful_results = [r for r in results if r.get('success', False)]
        
        if successful_results:
            best_result = min(successful_results, key=lambda x: x['val_loss'])
            
            self.results = results
            self.best_config = best_result
            
            logger.info(f"Best configuration found: {best_result['config']}")
            logger.info(f"Best validation loss: {best_result['val_loss']:.6f}")
            
            return best_result
        else:
            logger.warning("No successful configurations found")
            return None
    
    def compare_models(self, models_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare results across different model types
        
        Args:
            models_results: Dict mapping model_type to best result
            
        Returns:
            Comparison DataFrame
        """
        comparison = []
        
        for model_type, result in models_results.items():
            if result and result.get('success'):
                comparison.append({
                    'Model': model_type.upper(),
                    'Val Loss': f"{result['val_loss']:.6f}",
                    'RMSE': f"{result.get('rmse', 0):.6f}",
                    'MAE': f"{result.get('mae', 0):.6f}",
                    'MAPE': f"{result.get('mape', 0):.2f}%",
                    'Overfitting': f"{result.get('overfitting', 0):.2%}",
                    'Best Config': str(result['config'])
                })
        
        return pd.DataFrame(comparison)
    
    def save_results(self, filepath: str = 'models/automl_results.json'):
        """Save AutoML results to file"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Convert numpy types to native Python types
            def convert_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(item) for item in obj]
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj
            
            results_to_save = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'best_config': convert_types(self.best_config),
                'all_results': convert_types(self.results)
            }
            
            with open(filepath, 'w') as f:
                json.dump(results_to_save, f, indent=2)
                
            logger.info(f"Results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
