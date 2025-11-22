import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt

from config import *
from config import MODEL_DIR
from config import PLOTS_DIR
from model import FraudDetectionSystem
from sklearn.metrics import accuracy_score, classification_report

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetaLearner(nn.Module):
    """Meta-learner that combines predictions from base models"""
    
    def __init__(self, n_base_models: int, hidden_size: int = 64, 
                 output_size: int = 2, dropout_rate: float = 0.2):
        """
        Args:
            n_base_models: Number of base models (input features will be n_base_models * 2 for class probabilities)
            hidden_size: Size of hidden layers
            output_size: Number of output classes
            dropout_rate: Dropout rate for regularization
        """
        super(MetaLearner, self).__init__()
        
        # Input size is n_base_models * output_size (each model outputs class probabilities)
        input_size = n_base_models * output_size
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through meta-learner"""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)  
        return x


class EnsembleSystem:
    """
    Ensemble learning system with stacking
    
    Supports multiple stacking methods:
    - average: Simple averaging of predictions
    - weighted: Weighted averaging based on model performance
    - meta: Meta-learning with neural network
    """
    
    def __init__(self, stacking_method: str = "meta", 
                 base_model_names: Optional[List[str]] = None,
                 meta_config: Optional[Dict[str, Any]] = None):
        """
        Initialize ensemble system
        
        Args:
            stacking_method: Method for combining predictions ("average", "weighted", "meta")
            base_model_names: Names of base models
            meta_config: Configuration for meta-learner
        """
        self.stacking_method = stacking_method
        
        # Base model names
        if base_model_names is None:
            self.base_model_names = ["traffic", "mobility", "social", "all_naive"]
        else:
            self.base_model_names = base_model_names
        
        # Initialize base models
        self.base_models: Dict[str, FraudDetectionSystem] = {
            name: FraudDetectionSystem() for name in self.base_model_names
        }
        
        # Meta-learner configuration
        if meta_config is None:
            self.meta_config = {
                "hidden_size": 64,
                "output_size": 2,
                "dropout_rate": 0.2,
                "learning_rate": 0.001,
                "epochs": NUM_EPOCHS
            }
        else:
            self.meta_config = meta_config
        
        # Meta-learner model
        self.meta_learner: Optional[MetaLearner] = None
        self.meta_optimizer: Optional[optim.Optimizer] = None
        self.meta_criterion = nn.CrossEntropyLoss()
        
        # Model weights for weighted averaging
        self.model_weights: Dict[str, float] = {name: 1.0 for name in self.base_model_names}
        
        # Training history
        self.ensemble_history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_accuracy': []
        }
    
    def fit_base_models(self, train_data_paths: Dict[str, str], 
                       num_epochs: int = NUM_EPOCHS, 
                       timer: int = PRINT_FREQUENCY,
                       force_retrain: bool = False):
        """
        Train or load all base models
        
        Args:
            train_data_paths: Dictionary mapping model names to their training data paths
            num_epochs: Number of training epochs
            timer: Frequency for logging
            force_retrain: If True, retrain even if saved model exists
        """
        logger.info(f"Training/Loading {len(self.base_models)} base models...")
        
        for model_name in self.base_model_names:
            model_path = Path(MODEL_DIR) / f"{model_name}.pth"
            plot_dir = Path(PLOTS_DIR) / model_name
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if we should load existing model
            if model_path.exists() and not force_retrain:
                try:
                    logger.info(f"Loading existing model: {model_name}")
                    self.base_models[model_name].load_model(str(model_path))
                    logger.info(f"✓ Model {model_name} loaded successfully")
                    continue
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")
                    logger.info("→ Training new model...")
            
            # Train new model
            logger.info(f"Training base model: {model_name}")
            
            # Get training data path
            if model_name not in train_data_paths:
                logger.warning(f"No training data path for {model_name}, skipping...")
                continue
            
            train_path = train_data_paths[model_name]
            
            try:
                # Load and preprocess data
                X, y, feature_names = self.base_models[model_name].load_and_preprocess_data(train_path)
                logger.info(f"Loaded {len(X)} samples with {len(feature_names)} features")
                
                # Prepare training data
                data_tensors = self.base_models[model_name].prepare_training_data(X, y)
                logger.info(f"Training data shape: {data_tensors['X_train'].shape}")
                
                # Train model
                self.base_models[model_name].train(data_tensors, num_epochs, timer=timer)
                
                # Save model
                self.base_models[model_name].save_model(str(model_path))
                logger.info(f"✓ Model {model_name} saved to {model_path}")
                
                # Visualize training history
                # if self.base_models[model_name].training_history:
                #     self._visualize_training_history(
                #         self.base_models[model_name].training_history,
                #         plot_dir / "training_history.png",
                #         model_name,
                #         sample_epoch=timer
                #     )
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        logger.info("All base models trained/loaded successfully")
    
    def load_and_preprocess_data(self, data_path: str):
        self.base_models_val_data = {}
        if type(data_path) is dict:
            for b in self.base_model_names:
                X_val, y_val, _ = self.base_models[b].load_and_preprocess_data(data_path[b])
                self.base_models_val_data[b] = (X_val, y_val)
            logger.info(f"✓ Loaded validation data from {data_path[b]}")
        else:
            for b in self.base_model_names:
                if b in data_path:
                    X_val, y_val, _ = self.base_models[b].load_and_preprocess_data(data_path)
                    logger.info(f"✓ Loaded validation data from {data_path}")
                    self.base_models_val_data[b] = (X_val, y_val)
                    return X_val, y_val
    
    def load_validation_data(self, val_path):
        return self.load_and_preprocess_data(val_path)

    def load_test_data(self, test_path):
        return self.load_and_preprocess_data(test_path)
    

    def load_base_models(self, model_dir: str = MODEL_DIR):
        """Load all base models from saved files"""
        logger.info(f"Loading base models from {model_dir}...")
        
        for model_name in self.base_model_names:
            model_path = Path(model_dir) / f"{model_name}.pth"
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.base_models[model_name].load_model(str(model_path))
            logger.info(f"✓ Loaded {model_name}")
        
        logger.info("All base models loaded successfully")
    

    def generate_meta_features(self, X: np.ndarray, 
                               data_path: Optional[str] = None) -> torch.Tensor:
        """
        Generate meta-features from base model predictions
        
        Args:
            X: Input features (already preprocessed)
            data_path: Optional path to raw data file (for preprocessing)
            
        Returns:
            Meta-features tensor combining all base model predictions
        """
        # Check all models are loaded
        for name, model in self.base_models.items():
            if model.model is None:
                raise ValueError(f"Base model '{name}' not trained/loaded")
        
        meta_features_list = []
        
        # Get predictions from each base model
        for name, model in self.base_models.items():
            # Convert to tensor if needed
            if isinstance(X, np.ndarray):
                X_tensor = torch.tensor(X, dtype=torch.float32)
            else:
                X_tensor = X
            
            # Get prediction probabilities
            model.model.eval()
            with torch.no_grad():
                outputs = model.model(X_tensor)
                probs = torch.softmax(outputs, dim=1)  # Shape: (n_samples, n_classes)
                meta_features_list.append(probs)
        
        # Concatenate all predictions
        # Shape: (n_samples, n_models * n_classes)
        meta_features = torch.cat(meta_features_list, dim=1)
        
        return meta_features
    
    def train_meta_learner(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: Optional[np.ndarray] = None, 
                          y_val: Optional[np.ndarray] = None,
                          val_split: float = 0.2):
        """
        Train the meta-learner on base model predictions
        
        Args:
            X_train: Training features (preprocessed)
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            val_split: Validation split ratio if X_val not provided
        """
        if self.stacking_method != "meta":
            raise ValueError("train_meta_learner only works with stacking_method='meta'")
        
        logger.info("Training meta-learner...")
        
        # Split validation set if not provided
        if X_val is None:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_split, random_state=RANDOM_SEED
            )
        
        # Generate meta-features
        logger.info("Generating meta-features from base models...")
        meta_features_train = self.generate_meta_features(X_train)
        meta_features_val = self.generate_meta_features(X_val)
        
        # Convert labels to tensors
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        
        # Initialize meta-learner
        n_base_models = len(self.base_models)
        self.meta_learner = MetaLearner(
            n_base_models=n_base_models,
            hidden_size=self.meta_config['hidden_size'],
            output_size=self.meta_config['output_size'],
            dropout_rate=self.meta_config['dropout_rate']
        )
        
        self.meta_optimizer = optim.Adam(
            self.meta_learner.parameters(), 
            lr=self.meta_config['learning_rate']
        )
        
        # Training loop
        num_epochs = self.meta_config['epochs']
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            # Training phase
            self.meta_learner.train()
            self.meta_optimizer.zero_grad()
            
            outputs = self.meta_learner(meta_features_train)
            loss = self.meta_criterion(outputs, y_train_tensor)
            loss.backward()
            self.meta_optimizer.step()
            
            self.ensemble_history['train_loss'].append(loss.item())
            
            # Validation phase
            if (epoch + 1) % PRINT_FREQUENCY == 0:
                self.meta_learner.eval()
                with torch.no_grad():
                    val_outputs = self.meta_learner(meta_features_val)
                    val_preds = torch.argmax(val_outputs, dim=1)
                    val_acc = accuracy_score(y_val, val_preds.numpy())
                    self.ensemble_history['val_accuracy'].append(val_acc)
                    
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                    
                    logger.info(
                        f'Epoch [{epoch+1}/{num_epochs}], '
                        f'Loss: {loss.item():.4f}, '
                        f'Val Acc: {val_acc:.4f} '
                        f'(Best: {best_val_acc:.4f})'
                    )
        
        logger.info(f"Meta-learner training complete. Best validation accuracy: {best_val_acc:.4f}")
    
    def compute_model_weights(self, X_val: np.ndarray, y_val: np.ndarray):
        """
        Compute optimal weights for each base model based on validation performance
        
        Args:
            X_val: Validation features
            y_val: Validation labels
        """
        logger.info("Computing model weights based on validation performance...")
        
        accuracies = {}
        
        for name, model in self.base_models.items():
            if model.model is None:
                logger.warning(f"Model {name} not loaded, skipping weight computation")
                continue
            
            # Get predictions
            X_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_tensor = torch.tensor(y_val, dtype=torch.long)
            
            results = model.evaluate(X_tensor, y_tensor)
            accuracies[name] = results['accuracy']
            logger.info(f"  {name}: {results['accuracy']:.4f}")
        
        # Normalize weights
        total_acc = sum(accuracies.values())
        self.model_weights = {name: acc / total_acc for name, acc in accuracies.items()}
        
        logger.info("Model weights computed:")
        for name, weight in self.model_weights.items():
            logger.info(f"  {name}: {weight:.4f}")
    


    def fit(self, X: np.ndarray, y: np.ndarray, val_split: float = 0.2):
        """
        Fit the ensemble system
        
        Args:
            X: Training features (already preprocessed by base models)
            y: Training labels
            val_split: Validation split ratio
        """
        if self.stacking_method == "meta":
            self.train_meta_learner(X, y, val_split=val_split)
        elif self.stacking_method == "weighted":
            from sklearn.model_selection import train_test_split
            _, X_val, _, y_val = train_test_split(
                X, y, test_size=val_split, random_state=RANDOM_SEED
            )
            self.compute_model_weights(X_val, y_val)
        elif self.stacking_method == "average":
            logger.info("Using simple averaging, no training needed")
        else:
            raise ValueError(f"Unknown stacking method: {self.stacking_method}")
    


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the ensemble
        
        Args:
            X: Input features (preprocessed)
            
        Returns:
            Predictions array
        """
        if self.stacking_method == "meta":
            if self.meta_learner is None:
                raise ValueError("Meta-learner not trained. Call fit() first.")
            
            meta_features = self.generate_meta_features(X)
            
            self.meta_learner.eval()
            with torch.no_grad():
                outputs = self.meta_learner(meta_features)
                predictions = torch.argmax(outputs, dim=1)
            
            return predictions.numpy()
        
        elif self.stacking_method in ["average", "weighted"]:
            # Get predictions from all base models
            all_probs = []
            
            for name, model in self.base_models.items():
                X_tensor = torch.tensor(X, dtype=torch.float32)
                
                model.model.eval()
                with torch.no_grad():
                    outputs = model.model(X_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    
                    # Apply weights if using weighted averaging
                    if self.stacking_method == "weighted":
                        weight = self.model_weights.get(name, 1.0)
                        probs = probs * weight
                    
                    all_probs.append(probs)
            
            # Average predictions
            avg_probs = torch.stack(all_probs).mean(dim=0)
            predictions = torch.argmax(avg_probs, dim=1)
            
            return predictions.numpy()
        
        else:
            raise ValueError(f"Unknown stacking method: {self.stacking_method}")
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate ensemble performance
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(X)
        
        accuracy = accuracy_score(y, predictions)
        report = classification_report(y, predictions)
        
        results = {
            'predictions': predictions,
            'accuracy': accuracy,
            'classification_report': report
        }
        
        logger.info(f'Ensemble Accuracy: {accuracy:.4f}')
        logger.info(f'\n{report}')
        
        return results
    
    def save_ensemble(self, save_dir: str = MODEL_DIR):
        """Save the entire ensemble system"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save meta-learner if using meta stacking
        if self.stacking_method == "meta" and self.meta_learner is not None:
            meta_path = save_path / "meta_learner.pth"
            torch.save({
                'model_state_dict': self.meta_learner.state_dict(),
                'meta_config': self.meta_config,
                'ensemble_history': self.ensemble_history
            }, meta_path)
            logger.info(f"Meta-learner saved to {meta_path}")
        
        # Save model weights if using weighted averaging
        if self.stacking_method == "weighted":
            import json
            weights_path = save_path / "model_weights.json"
            with open(weights_path, 'w') as f:
                json.dump(self.model_weights, f, indent=2)
            logger.info(f"Model weights saved to {weights_path}")
        
        # Save ensemble configuration
        config_path = save_path / "ensemble_config.json"
        import json
        with open(config_path, 'w') as f:
            json.dump({
                'stacking_method': self.stacking_method,
                'base_model_names': self.base_model_names,
                'meta_config': self.meta_config
            }, f, indent=2)
        logger.info(f"Ensemble configuration saved to {config_path}")
    
    def load_ensemble(self, save_dir: str = MODEL_DIR):
        """Load the entire ensemble system"""
        save_path = Path(save_dir)
        
        # Load ensemble configuration
        config_path = save_path / "ensemble_config.json"
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.stacking_method = config['stacking_method']
        self.base_model_names = config['base_model_names']
        self.meta_config = config['meta_config']
        
        # Load base models
        self.load_base_models(save_dir)
        
        # Load meta-learner if using meta stacking
        if self.stacking_method == "meta":
            meta_path = save_path / "meta_learner.pth"
            checkpoint = torch.load(meta_path, map_location='cpu')
            
            n_base_models = len(self.base_models)
            self.meta_learner = MetaLearner(
                n_base_models=n_base_models,
                hidden_size=self.meta_config['hidden_size'],
                output_size=self.meta_config['output_size'],
                dropout_rate=self.meta_config['dropout_rate']
            )
            self.meta_learner.load_state_dict(checkpoint['model_state_dict'])
            self.ensemble_history = checkpoint['ensemble_history']
            logger.info(f"Meta-learner loaded from {meta_path}")
        
        # Load model weights if using weighted averaging
        if self.stacking_method == "weighted":
            weights_path = save_path / "model_weights.json"
            with open(weights_path, 'r') as f:
                self.model_weights = json.load(f)
            logger.info(f"Model weights loaded from {weights_path}")
        
        logger.info("Ensemble system loaded successfully")
    
    def _visualize_training_history(self, history: Dict[str, List[float]], 
                                   save_path: Path, model_name: str,
                                   sample_epoch: int = 100):
        """Visualize and save training history for a base model"""
        plt.figure(figsize=(14, 5))
        
        # Training loss
        plt.subplot(1, 2, 1)
        train_loss = history.get('train_loss', [])
        if train_loss:
            sampled_losses = train_loss[::sample_epoch]
            epochs = range(0, len(train_loss), sample_epoch)
            plt.plot(epochs, sampled_losses, 'b-', linewidth=2)
            plt.title(f'{model_name} - Training Loss', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.grid(True, alpha=0.3)
        
        # Test accuracy
        plt.subplot(1, 2, 2)
        test_acc = history.get('test_accuracy', [])
        if test_acc:
            plt.plot(test_acc, 'r-', linewidth=2, marker='o', markersize=4)
            plt.title(f'{model_name} - Validation Accuracy', fontsize=14, fontweight='bold')
            plt.xlabel('Validation Step', fontsize=12)
            plt.ylabel('Accuracy', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            best_acc = max(test_acc)
            plt.axhline(y=best_acc, color='g', linestyle='--', alpha=0.5,
                       label=f'Best: {best_acc:.4f}')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Training history saved to {save_path}")
    
    def visualize_ensemble_performance(self, X_test: np.ndarray, y_test: np.ndarray):
        """
        Visualize ensemble and individual model performances
        """
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Ensemble Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Individual model accuracies
        ax = axes[0, 0]
        model_accs = {}
        for name, model in self.base_models.items():
            X_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_tensor = torch.tensor(y_test, dtype=torch.long)
            results = model.evaluate(X_tensor, y_tensor)
            model_accs[name] = results['accuracy']
        
        # Add ensemble accuracy
        ensemble_pred = self.predict(X_test)
        ensemble_acc = accuracy_score(y_test, ensemble_pred)
        model_accs['Ensemble'] = ensemble_acc
        
        colors = ['skyblue'] * len(self.base_models) + ['coral']
        ax.bar(model_accs.keys(), model_accs.values(), color=colors)
        ax.set_title('Model Accuracies Comparison')
        ax.set_ylabel('Accuracy')
        ax.set_ylim([0, 1])
        ax.tick_params(axis='x', rotation=45)
        for i, (name, acc) in enumerate(model_accs.items()):
            ax.text(i, acc + 0.02, f'{acc:.3f}', ha='center')
        
        # 2. Ensemble confusion matrix
        ax = axes[0, 1]
        cm = confusion_matrix(y_test, ensemble_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Ensemble Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        # 3. Model weights (if applicable)
        ax = axes[0, 2]
        if self.stacking_method == "weighted":
            ax.bar(self.model_weights.keys(), self.model_weights.values())
            ax.set_title('Model Weights')
            ax.set_ylabel('Weight')
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, f'Stacking: {self.stacking_method}', 
                   ha='center', va='center', fontsize=14)
            ax.set_title('Stacking Method')
            ax.axis('off')
        
        # 4. Training history (if meta-learner)
        ax = axes[1, 0]
        if self.stacking_method == "meta" and self.ensemble_history['train_loss']:
            ax.plot(self.ensemble_history['train_loss'][::10], 'b-')
            ax.set_title('Meta-Learner Training Loss')
            ax.set_xlabel('Epoch (x10)')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
        else:
            ax.axis('off')
        
        # 5. Validation accuracy (if meta-learner)
        ax = axes[1, 1]
        if self.stacking_method == "meta" and self.ensemble_history['val_accuracy']:
            ax.plot(self.ensemble_history['val_accuracy'], 'r-', marker='o')
            ax.set_title('Meta-Learner Validation Accuracy')
            ax.set_xlabel('Validation Step')
            ax.set_ylabel('Accuracy')
            ax.grid(True, alpha=0.3)
        else:
            ax.axis('off')
        
        # 6. Performance improvement
        ax = axes[1, 2]
        base_accs = [acc for name, acc in model_accs.items() if name != 'Ensemble']
        mean_base = np.mean(base_accs)
        best_base = max(base_accs)
        
        improvement_data = {
            'Mean Base': mean_base,
            'Best Base': best_base,
            'Ensemble': ensemble_acc
        }
        colors = ['lightblue', 'skyblue', 'coral']
        bars = ax.bar(improvement_data.keys(), improvement_data.values(), color=colors)
        ax.set_title('Performance Comparison')
        ax.set_ylabel('Accuracy')
        ax.set_ylim([0, 1])
        for i, (name, acc) in enumerate(improvement_data.items()):
            ax.text(i, acc + 0.02, f'{acc:.3f}', ha='center')
        
        # Add improvement percentage
        improvement = ((ensemble_acc - best_base) / best_base) * 100
        ax.text(0.5, 0.1, f'Improvement: {improvement:+.2f}%', 
               transform=ax.transAxes, ha='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
        
        logger.info(f"Ensemble Accuracy: {ensemble_acc:.4f}")
        logger.info(f"Best Base Model: {best_base:.4f}")
        logger.info(f"Improvement: {improvement:+.2f}%")


