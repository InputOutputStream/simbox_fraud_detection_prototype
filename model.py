import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, roc_curve, auc

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

import seaborn as sns

import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from preprocessor import DataPreprocessor

from config import *
QuickConfig.quick_test()

"""
1. Data Processing Flow ✅
Raw CSV → Discretization → Feature Selection → Scaling → (Optional PCA) → Model Input

2. Training Pipeline ✅
Data Loading → Preprocessing → Train/Test Split → Sampling → Model Training → Evaluation

3. Model Persistence ✅
Training → Save (Model + Preprocessor State) → Load → Predict on New Data

"""

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLPFraudDetector(nn.Module):
    """Multi-Layer Perceptron for fraud detection"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 2, dropout_rate: float = 0.2):
        super(MLPFraudDetector, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class ImbalancedDataHandler:
    """Handles imbalanced dataset using various sampling techniques"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.smote = SMOTE(random_state=random_state)
        self.undersampler = RandomUnderSampler(random_state=random_state)
    
    def apply_combined_sampling(self, X: np.ndarray, y: np.ndarray, 
                               iterations: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Apply iterative SMOTE + Random Under Sampling"""
        X_resampled, y_resampled = X.copy(), y.copy()
        
        for i in range(iterations):
            logger.info(f"Sampling iteration {i+1}/{iterations}")
            # Apply SMOTE
            X_resampled, y_resampled = self.smote.fit_resample(X_resampled, y_resampled)
            # Apply Random Under Sampling
            X_resampled, y_resampled = self.undersampler.fit_resample(X_resampled, y_resampled)
            
            logger.info(f"After iteration {i+1}: {len(X_resampled)} samples")
        
        return X_resampled, y_resampled


class FraudDetectionSystem:
    """Main fraud detection system orchestrating all components"""
    
    def __init__(self, model_params: Dict[str, Any] = None, random_state: int = 42):
        self.random_state = random_state
        self.preprocessor = DataPreprocessor()
        self.imbalance_handler = ImbalancedDataHandler(random_state)
        self.model = None
        self.model_params = model_params or {'hidden_size': 64, 'dropout_rate': 0.2}
        self.training_history = []
        self.input_size = None  # Store input size for model saving/loading
        
        # Training parameters
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        
    
    def load_and_preprocess_data(self, file_path: str, target_column: str = TARGET_COLUMN,
                                use_pca: bool = USE_PCA) -> tuple[np.ndarray, np.ndarray, List[str]]:
        """        
        Returns:
            X_processed: Processed features (original or principal components)
            y: Target values
            feature_names: Names of features/components
        """
        print(f"Loading data from {file_path}")
        data = pd.read_csv(file_path)
        
        # Discretize categorical features
        data = self.preprocessor.discretize_categorical_features(data)
        
        # Extract target and features
        y = data[target_column].values
        X = data[self.preprocessor.feature_columns].copy()
        
        if use_pca:
            print("Applying PCA transformation...")
            X_processed, explained_var, feature_names = self.preprocessor.apply_pca(X, fit=True)
            print(f"Using {len(feature_names)} principal components")
            
            # Visualize PCA analysis
            if GENERATE_PLOTS["pca_components"]:
                self.preprocessor.visualize_pca_components(data, y)
                
        else:
            print("Using original features (no PCA)")
            X_processed = self.preprocessor.scaler.fit_transform(X)
            feature_names = self.preprocessor.feature_columns
        
        print(f"Final data shape: {X_processed.shape}")
        print(f"Class distribution: {np.bincount(y)}")
        
        return X_processed, y, feature_names
    
    def get_feature_analysis(self) -> Dict[str, Any]:
        """
        Get analysis of features/components used in the model
        """
        analysis = {
            "using_pca": self.preprocessor.pca is not None,
            "n_features": len(self.preprocessor.feature_columns)
        }
        
        if self.preprocessor.pca is not None:
            analysis.update({
                "n_components": len(self.preprocessor.component_names),
                "explained_variance": sum(self.preprocessor.pca.explained_variance_ratio_),
                "component_names": self.preprocessor.component_names,
                "component_explanations": self.preprocessor.explain_components()
            })
        else:
            analysis.update({
                "feature_names": self.preprocessor.feature_columns
            })
        
        return analysis

    def prepare_training_data(self, X: np.ndarray, y: np.ndarray, 
                             test_size: float = TEST_SIZE, apply_sampling: bool = APPLY_SAMPLING) -> Dict[str, torch.Tensor]:
        """Prepare training and testing data with optional sampling"""
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Apply sampling to training data if requested
        if apply_sampling:
            X_train, y_train = self.imbalance_handler.apply_combined_sampling(X_train, y_train)
        
        # Convert to PyTorch tensors
        return {
            'X_train': torch.tensor(X_train, dtype=torch.float32),
            'X_test': torch.tensor(X_test, dtype=torch.float32),
            'y_train': torch.tensor(y_train, dtype=torch.long),
            'y_test': torch.tensor(y_test, dtype=torch.long)
        }        
    
    def initialize_model(self, input_size: int, lr: float = 0.001, output_size: int = OUTPUT_SIZE):
        """Initialize the MLP model"""
        if self.model is None:
            self.input_size = input_size  # Store the integer, not the shape
            self.model = MLPFraudDetector(
                input_size=input_size,  # Pass the integer directly
                hidden_size=self.model_params.get('hidden_size', 64),
                output_size=output_size,  # Should be 2 for binary classification
                dropout_rate=self.model_params.get('dropout_rate', 0.2),
            )
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
            logger.info(f"Model initialized with input size: {input_size}")
    
    def train(self, data_tensors: Dict[str, torch.Tensor], num_epochs: int = NUM_EPOCHS, 
              lr: float = LEARNING_RATE, output_size: int = OUTPUT_SIZE,
              timer : int = PRINT_FREQUENCY) -> Dict[str, List[float]]:

        if self.model is None:
            input_size = data_tensors['X_train'].shape[1] 
            self.initialize_model(input_size, output_size=output_size, lr=lr)
            
        X_train, y_train = data_tensors['X_train'], data_tensors['y_train']
        X_test, y_test = data_tensors['X_test'], data_tensors['y_test']
        
        train_losses, test_accuracies = [], []
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            self.optimizer.zero_grad()
            
            outputs = self.model(X_train)
            loss = self.criterion(outputs, y_train)
            loss.backward()
            self.optimizer.step()
            
            train_losses.append(loss.item())
            
            # Validation phase (every 'timer' epochs)
            if (epoch + 1) % timer == 0:
                self.model.eval()
                with torch.no_grad():
                    test_outputs = self.model(X_test)
                    test_pred = torch.argmax(test_outputs, axis=1)
                    test_acc = accuracy_score(y_test.numpy(), test_pred.numpy())
                    test_accuracies.append(test_acc)
                
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Acc: {test_acc:.4f}')
        
        self.training_history = {'train_loss': train_losses, 'test_accuracy': test_accuracies}
        return self.training_history
    
    def evaluate(self, X: torch.Tensor, y: torch.Tensor = None) -> Dict[str, Any]:
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            predictions = torch.argmax(outputs, axis=1)
        
        results = {'predictions': predictions.numpy()}
        
        if y is not None:
            accuracy = accuracy_score(y.numpy(), predictions.numpy())
            report = classification_report(y.numpy(), predictions.numpy())
            
            results.update({
                'accuracy': accuracy,
                'classification_report': report
            })
            
            logger.info(f'Accuracy: {accuracy:.4f}')
            logger.info(f'\n{report}')
        
        return results
    
    def test_robustness(self, X: np.ndarray, y: np.ndarray, noise_levels : List = NOISE_LEVELS) -> Dict[float, float]:
        """Test model robustness with different noise levels"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        robustness_results = {}
        
        for noise_level in noise_levels:
            logger.info(f"Testing robustness with noise level: {noise_level}")
            
            # Create noisy version of data
            X_df = pd.DataFrame(X, columns=self.preprocessor.feature_columns)
            X_noisy_df = self.preprocessor.add_gaussian_noise(
                X_df, self.preprocessor.feature_columns, std_dev=noise_level
            )
            
            # Scale and convert to tensor
            X_noisy_scaled = self.preprocessor.scaler.transform(X_noisy_df)
            X_noisy_tensor = torch.tensor(X_noisy_scaled, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.long)
            
            # Evaluate
            results = self.evaluate(X_noisy_tensor, y_tensor)
            robustness_results[noise_level] = results['accuracy']
        
        return robustness_results
    
    def save_model(self, filepath: str, save_separate: bool = False):
        """Save the trained model and preprocessor"""
        if self.model is None:
            raise ValueError("No model to save")

        if save_separate:
            # Modern approach: separate model weights and preprocessor
            model_path = filepath.replace('.pth', '_weights.pth')
            preprocessor_path = filepath.replace('.pth', '_preprocessor.pkl')
            
            # Save only model weights (safe with weights_only=True)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_params': self.model_params,
                'input_size': self.input_size,
                'training_history': self.training_history
            }, model_path)
            
            # Save preprocessor separately with pickle
            import pickle
            with open(preprocessor_path, 'wb') as f:
                pickle.dump(self.preprocessor, f)
            
            logger.info(f"Model weights saved to {model_path}")
            logger.info(f"Preprocessor saved to {preprocessor_path}")
        else:
            # Original approach with weights_only=False compatibility
            save_dict = {
                'model_state_dict': self.model.state_dict(),
                'model_params': self.model_params,
                'input_size': self.input_size,
                'preprocessor': self.preprocessor,
                'training_history': self.training_history
            }
            
            torch.save(save_dict, filepath)
            logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str, load_separate: bool = False):
        """Load a trained model and preprocessor"""
        
        if load_separate:
            # Load from separate files
            model_path = filepath.replace('.pth', '_weights.pth')
            preprocessor_path = filepath.replace('.pth', '_preprocessor.pkl')
            
            # Load model weights (safe)
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
            
            # Load preprocessor with pickle
            import pickle
            with open(preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
            
            logger.info(f"Model weights loaded from {model_path}")
            logger.info(f"Preprocessor loaded from {preprocessor_path}")
        else:
            # Handle existing models with fallback
            try:
                # Try with safe globals first
                torch.serialization.add_safe_globals([DataPreprocessor, ImbalancedDataHandler])
                checkpoint = torch.load(filepath, map_location='cpu', weights_only=True)
            except:
                # Fallback to old method
                logger.warning("Using weights_only=False for backward compatibility")
                checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
                
            self.preprocessor = checkpoint['preprocessor']
        
        # Common loading logic
        self.model_params = checkpoint['model_params']
        self.input_size = checkpoint['input_size']
        self.training_history = checkpoint['training_history']
        
        # Initialize model with correct parameters
        self.model = MLPFraudDetector(
            input_size=self.input_size,
            hidden_size=self.model_params['hidden_size'],
            dropout_rate=self.model_params['dropout_rate']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        logger.info(f"Model loaded successfully")

    def predict_new_data(self, file_path: str, use_same_preprocessing: bool = True, 
                        sample_epoch : int = NUM_EPOCHS, target_column: str = TARGET_COLUMN,
                        save_plots: bool = True, plot_dir: str = "plots") -> Dict[str, Any]:
        """Predict on new unseen data using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() or load_model() first.")
        
        logger.info(f"Loading new data from {file_path}")
        data = pd.read_csv(file_path)
        
        # Apply same preprocessing as training data
        if use_same_preprocessing:
            data = self.preprocessor.discretize_categorical_features(data)
            
            # Check if target column exists (for evaluation)
            has_target = target_column in data.columns
            if has_target:
                y_true = data[target_column].values
                X = data[self.preprocessor.feature_columns].copy()
            else:
                y_true = None
                X = data[self.preprocessor.feature_columns].copy()
            
            # Apply same scaling (but not fit)
            if self.preprocessor.pca is not None:
                X_processed, _, names = self.preprocessor.apply_pca(X, fit=False)
                print(names)
            else:
                X_processed = self.preprocessor.scaler.transform(X)
        else:
            raise NotImplementedError("Different preprocessing not implemented yet")
        
        # Convert to tensor and predict
        X_tensor = torch.tensor(X_processed, dtype=torch.float32)
        
        if has_target:
            y_tensor = torch.tensor(y_true, dtype=torch.long)
            results = self.evaluate(X_tensor, y_tensor)
        else:
            results = self.evaluate(X_tensor)
            return results
        
        # Visualize and save training results
        if save_plots and self.training_history:
            # Create plots directory if it doesn't exist
            Path(plot_dir).mkdir(parents=True, exist_ok=True)
            
            # Generate plot filename from file_path
            base_name = Path(file_path).stem
            plot_path = Path(plot_dir) / f"{base_name}_training_history.png"
            
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(self.training_history['train_loss'][::sample_epoch])
            plt.title('Training Loss')
            plt.xlabel(f'Epoch (x{sample_epoch})')
            plt.ylabel('Loss')
            
            plt.subplot(1, 2, 2)
            if self.training_history['test_accuracy']:
                plt.plot(self.training_history['test_accuracy'])
                plt.title('Test Accuracy')
                plt.xlabel('Validation Step')
                plt.ylabel('Accuracy')
            
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Training history plot saved to {plot_path}")
        
        return results
        
    def visualize_data(self, data: pd.DataFrame = None, file_path: str = None, 
                    target_column: str = TARGET_COLUMN, 
                    show_distributions: bool = True, 
                    show_correlations: bool = True,
                    show_class_balance: bool = True,
                    show_pca_analysis: bool = True,
                    show_feature_importance: bool = True,
                    show_training_history: bool = True,
                    figsize: tuple = (15, 10)):
        """
        Comprehensive data visualization method for fraud detection analysis
        
        Args:
            data: DataFrame to visualize (optional if file_path provided)
            file_path: Path to CSV file (optional if data provided)
            target_column: Name of target column
            show_distributions: Show feature distributions by class
            show_correlations: Show correlation heatmap
            show_class_balance: Show class distribution
            show_pca_analysis: Show PCA variance analysis
            show_feature_importance: Show feature importance from PCA
            show_training_history: Show training metrics if available
            figsize: Figure size for plots
        """
        
        # Load data if not provided
        if data is None and file_path is not None:
            logger.info(f"Loading data from {file_path}")
            data = pd.read_csv(file_path)
            data = self.preprocessor.discretize_categorical_features(data)
        elif data is None:
            raise ValueError("Either data or file_path must be provided")
        
        # Create a copy to avoid modifying original data
        viz_data = data.copy()
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        plot_count = 0
        total_plots = sum([show_class_balance, show_correlations, show_distributions, 
                        show_pca_analysis, show_feature_importance, show_training_history])
        
        if total_plots == 0:
            logger.warning("No visualization options selected")
            return
        
        # Calculate subplot layout
        if total_plots <= 2:
            rows, cols = 1, total_plots
        elif total_plots <= 4:
            rows, cols = 2, 2
        elif total_plots <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3
        
        fig = plt.figure(figsize=figsize)
        
        # 1. Class Distribution
        if show_class_balance and target_column in viz_data.columns:
            plot_count += 1
            plt.subplot(rows, cols, plot_count)
            
            class_counts = viz_data[target_column].value_counts()
            colors = ['lightcoral', 'lightblue'] if len(class_counts) == 2 else None
            
            sns.countplot(data=viz_data, x=target_column, palette=colors)
            plt.title('Class Distribution (Fraudulent vs Legitimate)')
            plt.xlabel('Fraudulent User')
            plt.ylabel('Count')
            
            # Add percentage labels
            total = len(viz_data)
            for i, (label, count) in enumerate(class_counts.items()):
                plt.text(i, count + total*0.01, f'{count}\n({count/total*100:.1f}%)', 
                        ha='center', va='bottom')
        
        # 2. Feature Correlations
        if show_correlations and target_column in viz_data.columns:
            plot_count += 1
            plt.subplot(rows, cols, plot_count)
            
            # Select numerical columns for correlation
            numeric_cols = viz_data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 15:  # Limit to most important features
                if hasattr(self.preprocessor, 'feature_columns'):
                    numeric_cols = [col for col in self.preprocessor.feature_columns 
                                if col in numeric_cols][:15]
            
            corr_matrix = viz_data[numeric_cols].corr()
            
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                    center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8})
            plt.title('Feature Correlation Matrix')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
        
        # 3. Feature Distributions by Class
        if show_distributions and target_column in viz_data.columns:
            # Create separate figure for distributions
            fig_dist = plt.figure(figsize=(16, 12))
            
            # Select key features for distribution plots
            key_features = self.preprocessor.feature_columns[:8]  # Top 8 features
            
            for i, feature in enumerate(key_features):
                if feature in viz_data.columns:
                    plt.subplot(3, 3, i + 1)
                    
                    # Create distribution plot
                    for class_val in viz_data[target_column].unique():
                        subset = viz_data[viz_data[target_column] == class_val][feature]
                        label = 'Fraudulent' if class_val == 1 else 'Legitimate'
                        plt.hist(subset, alpha=0.7, label=label, bins=20, density=True)
                    
                    plt.title(f'Distribution of {feature}')
                    plt.xlabel(feature)
                    plt.ylabel('Density')
                    plt.legend()
            
            plt.tight_layout()
            plt.show()
        
        # 4. PCA Analysis
        if show_pca_analysis:
            plot_count += 1
            plt.subplot(rows, cols, plot_count)
            
            # Perform PCA if not already done
            if hasattr(self.preprocessor, 'pca') and self.preprocessor.pca is not None:
                explained_var = self.preprocessor.pca.explained_variance_ratio_
            else:
                # Temporary PCA for visualization
                features_for_pca = viz_data[self.preprocessor.feature_columns].copy()
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features_for_pca)
                
                pca_temp = PCA()
                pca_temp.fit(features_scaled)
                explained_var = pca_temp.explained_variance_ratio_
            
            cumulative_var = np.cumsum(explained_var)
            
            plt.plot(range(1, len(explained_var) + 1), cumulative_var, 'bo-')
            plt.axhline(y=0.95, color='red', linestyle='--', label='95% threshold')
            plt.axhline(y=0.99, color='orange', linestyle='--', label='99% threshold')
            plt.xlabel('Principal Components')
            plt.ylabel('Cumulative Explained Variance')
            plt.title('PCA Cumulative Explained Variance')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 5. Feature Importance from PCA
        if show_feature_importance and hasattr(self.preprocessor, 'pca') and self.preprocessor.pca is not None:
            plot_count += 1
            plt.subplot(rows, cols, plot_count)
            
            # Get feature importance from first 3 components
            components = np.abs(self.preprocessor.pca.components_[:3])
            feature_importance = np.mean(components, axis=0)
            
            feature_names = self.preprocessor.feature_columns
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=True)
            
            plt.barh(range(len(importance_df)), importance_df['importance'])
            plt.yticks(range(len(importance_df)), importance_df['feature'])
            plt.xlabel('Average Absolute Loading')
            plt.title('Feature Importance (from PCA)')
        
        # 6. Training History
        if show_training_history and self.training_history:
            plot_count += 1
            
            # Create separate figure for training history
            fig_train = plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            train_losses = self.training_history.get('train_loss', [])
            if train_losses:
                # Sample every 100 epochs for cleaner plot
                sample_rate = max(1, len(train_losses) // 100)
                sampled_losses = train_losses[::sample_rate]
                epochs = range(0, len(train_losses), sample_rate)
                
                plt.plot(epochs, sampled_losses, 'b-', linewidth=2)
                plt.title('Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            test_accuracies = self.training_history.get('test_accuracy', [])
            if test_accuracies:
                plt.plot(test_accuracies, 'r-', linewidth=2)
                plt.title('Validation Accuracy')
                plt.xlabel('Validation Step')
                plt.ylabel('Accuracy')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        # Show the main figure
        plt.tight_layout()
        plt.show()
        
        # Additional box plots for key features
        if show_distributions and target_column in viz_data.columns:
            self._plot_feature_boxplots(viz_data, target_column)

    def _plot_feature_boxplots(self, data: pd.DataFrame, target_column: str):
        """Helper method to create box plots for key features"""
        
        key_features = ['duration', 'size', 'hour', 'minute']  # Adjust based on your features
        available_features = [f for f in key_features if f in data.columns]
        
        if not available_features:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.ravel()
        
        for i, feature in enumerate(available_features[:4]):
            sns.boxplot(data=data, x=target_column, y=feature, ax=axes[i])
            axes[i].set_title(f'{feature} by Fraudulent Status')
            axes[i].set_xlabel('Fraudulent User')
            
        plt.tight_layout()
        plt.show()

    def plot_model_performance(self, X_test: torch.Tensor, y_test: torch.Tensor):
        """
        Plot detailed model performance metrics
        
        Args:
            X_test: Test features
            y_test: Test labels
        """
        if self.model is None:
            logger.warning("No trained model available")
            return
                
        # Get predictions
        results = self.evaluate(X_test, y_test)
        y_pred = results['predictions']
        y_true = y_test.numpy()
        
        # Get prediction probabilities
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test)
            y_prob = torch.softmax(outputs, dim=1)[:, 1].numpy()  # Probability of fraud class
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend(loc="lower right")
        axes[1].grid(True, alpha=0.3)
        
        # 3. Prediction Distribution
        axes[2].hist(y_prob[y_true == 0], alpha=0.7, label='Legitimate', bins=30)
        axes[2].hist(y_prob[y_true == 1], alpha=0.7, label='Fraudulent', bins=30)
        axes[2].set_xlabel('Fraud Probability')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Prediction Probability Distribution')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        logger.info(f"Model Performance - Accuracy: {results['accuracy']:.4f}, AUC: {roc_auc:.4f}")

    def visualize_robustness_test(self, robustness_results: Dict[float, float]):
        """
        Visualize robustness test results
        
        Args:
            robustness_results: Dictionary mapping noise levels to accuracies
        """
        if not robustness_results:
            logger.warning("No robustness results to visualize")
            return
        
        noise_levels = list(robustness_results.keys())
        accuracies = list(robustness_results.values())
        
        plt.figure(figsize=(10, 6))
        plt.plot(noise_levels, accuracies, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Noise Level (Standard Deviation)')
        plt.ylabel('Model Accuracy')
        plt.title('Model Robustness to Gaussian Noise')
        plt.grid(True, alpha=0.3)
        
        # Add accuracy values as text
        for i, (noise, acc) in enumerate(zip(noise_levels, accuracies)):
            plt.text(noise, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        logger.info("Robustness analysis completed")