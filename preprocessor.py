import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Optional
from config import *
from config import PCA_VARIANCE_THRESHOLD
from config import PCA_WHITEN

class DataPreprocessor:
    """Enhanced DataPreprocessor that works with Principal Components directly"""
    
    def __init__(self, feature_columns: List[str] = None, categorical_columns: List[str] = None):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.pca = None
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self.categorical_columns = categorical_columns or CATEGORICAL_COLUMNS
        self.original_feature_names = None
        self.component_names = None
        
    def discretize_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Discretize categorical features using label encoding approach"""
        data_copy = data.copy()
        
        for column in self.categorical_columns:
            if column in data_copy.columns:
                data_copy = self._discretize_column(data_copy, column)
        
        return data_copy
    
    def _discretize_column(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """Discretize a single column"""
        if column not in data.columns:
            print(f"Warning: Column {column} not found in dataset")
            return data
        
        unique_values = data[column].unique()
        mapping = {value: i+1 for i, value in enumerate(sorted(unique_values))}
        data[column] = data[column].map(mapping)
        
        return data
    
    def apply_pca(self, features: pd.DataFrame, n_components: Optional[int] = None, 
                  variance_threshold: float = PCA_VARIANCE_THRESHOLD, 
                  fit: bool = True) -> tuple[np.ndarray, float, List[str]]:
        """
        Apply PCA and return principal components directly
        
        Returns:
            principal_components: Transformed data in PC space
            explained_variance: Total explained variance
            component_names: Names for the principal components
        """
        if fit:
            # Store original feature names
            self.original_feature_names = list(features.columns)
            
            # Scale the features first
            features_scaled = self.scaler.fit_transform(features)
            
            # Determine optimal number of components
            if n_components is None:
                pca_temp = PCA()
                pca_temp.fit(features_scaled)
                cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
                n_components = np.argmax(cumsum >= variance_threshold) + 1
                print(f"Auto-selected {n_components} components for {variance_threshold:.1%} variance")
            
            # Apply PCA
            self.pca = PCA(n_components=n_components, whiten=PCA_WHITEN)
            principal_components = self.pca.fit_transform(features_scaled)
            explained_variance = sum(self.pca.explained_variance_ratio_)
            
            # Create meaningful component names
            self.component_names = [f"PC{i+1}" for i in range(n_components)]
            
            print(f"PCA: {features.shape[1]} features → {n_components} components")
            print(f"Explained variance: {explained_variance:.4f}")
            
        else:
            if self.pca is None:
                raise ValueError("PCA not fitted. Call with fit=True first.")
            
            features_scaled = self.scaler.transform(features)
            principal_components = self.pca.transform(features_scaled)
            explained_variance = sum(self.pca.explained_variance_ratio_)
        
        return principal_components, explained_variance, self.component_names

    def add_gaussian_noise(self, data: pd.DataFrame, feature_columns: List[str], std_dev: float = 0.1) -> pd.DataFrame:
        """Add Gaussian noise to specified feature columns"""
        data_noisy = data.copy()
        for column in feature_columns:
            if column in data_noisy.columns:
                noise = np.random.normal(0, std_dev, size=data_noisy.shape[0])
                data_noisy[column] += noise
        return data_noisy
    
    def get_pca_loadings(self) -> pd.DataFrame:
        """
        Get PCA loadings matrix showing how original features contribute to each PC
        
        Returns:
            DataFrame with original features as rows, PCs as columns
        """
        if self.pca is None:
            raise ValueError("PCA not fitted yet")
        
        loadings = pd.DataFrame(
            self.pca.components_.T,
            columns=self.component_names,
            index=self.original_feature_names
        )
        
        return loadings
    
    def explain_components(self, n_features: int = 3) -> Dict[str, List[str]]:
        """
        Explain what each principal component represents based on highest loadings
        
        Args:
            n_features: Number of top contributing features to show per component
            
        Returns:
            Dictionary mapping component names to their top contributing features
        """
        if self.pca is None:
            raise ValueError("PCA not fitted yet")
        
        loadings = self.get_pca_loadings()
        component_explanations = {}
        
        for pc in self.component_names:
            # Get absolute loadings for this component
            abs_loadings = loadings[pc].abs().sort_values(ascending=False)
            top_features = abs_loadings.head(n_features).index.tolist()
            
            # Add signs to show direction
            feature_contributions = []
            for feature in top_features:
                sign = "+" if loadings.loc[feature, pc] > 0 else "-"
                feature_contributions.append(f"{sign}{feature}")
            
            component_explanations[pc] = feature_contributions
        
        return component_explanations
    
    def visualize_pca_components(self, data: pd.DataFrame = None, target: np.ndarray = None):
        """
        Visualize PCA components and their relationships
        """
        if self.pca is None:
            raise ValueError("PCA not fitted yet")
        
        n_components = len(self.component_names)
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Explained Variance Plot
        plt.subplot(2, 3, 1)
        explained_var_ratio = self.pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var_ratio)
        
        plt.bar(range(1, n_components + 1), explained_var_ratio, alpha=0.7, label='Individual')
        plt.plot(range(1, n_components + 1), cumulative_var, 'ro-', label='Cumulative')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA Explained Variance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Component Loadings Heatmap
        plt.subplot(2, 3, 2)
        loadings = self.get_pca_loadings()
        sns.heatmap(loadings.T, annot=True, cmap='RdBu_r', center=0, 
                   fmt='.2f', cbar_kws={'label': 'Loading'})
        plt.title('PCA Loadings Matrix')
        plt.xlabel('Original Features')
        plt.ylabel('Principal Components')
        
        # 3. First Two Components Scatter (if target provided)
        if data is not None and target is not None and n_components >= 2:
            plt.subplot(2, 3, 3)
            
            # Get PC scores
            pc_data, _, _ = self.apply_pca(data[self.feature_columns], fit=False)
            
            # Create scatter plot
            scatter = plt.scatter(pc_data[:, 0], pc_data[:, 1], c=target, 
                                cmap='RdYlBu', alpha=0.6)
            plt.colorbar(scatter, label='Fraudulent')
            plt.xlabel(f'{self.component_names[0]} ({explained_var_ratio[0]:.1%} var)')
            plt.ylabel(f'{self.component_names[1]} ({explained_var_ratio[1]:.1%} var)')
            plt.title('First Two Principal Components')
            plt.grid(True, alpha=0.3)
        
        # 4. Top Contributing Features per Component
        plt.subplot(2, 3, 4)
        explanations = self.explain_components(n_features=5)
        
        y_pos = 0
        colors = plt.cm.Set3(np.linspace(0, 1, len(explanations)))
        
        for i, (pc, features) in enumerate(explanations.items()):
            for j, feature in enumerate(features):
                plt.barh(y_pos, 1, color=colors[i], alpha=0.7)
                plt.text(0.5, y_pos, f'{pc}: {feature}', ha='center', va='center')
                y_pos += 1
        
        plt.xlim(0, 1)
        plt.ylim(-0.5, y_pos - 0.5)
        plt.title('Top Contributing Features per PC')
        plt.axis('off')
        
        # 5. Component Correlation Matrix (if enough components)
        if n_components > 2 and data is not None:
            plt.subplot(2, 3, 5)
            pc_data, _, _ = self.apply_pca(data[self.feature_columns], fit=False)
            pc_df = pd.DataFrame(pc_data, columns=self.component_names)
            
            corr_matrix = pc_df.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       fmt='.2f', square=True)
            plt.title('Principal Component Correlations')
        
        # 6. Individual Component Distributions (if target provided)
        if data is not None and target is not None:
            plt.subplot(2, 3, 6)
            pc_data, _, _ = self.apply_pca(data[self.feature_columns], fit=False)
            
            # Plot distribution of first component by class
            for class_val in np.unique(target):
                class_data = pc_data[target == class_val, 0]
                label = 'Fraudulent' if class_val == 1 else 'Legitimate'
                plt.hist(class_data, alpha=0.7, label=label, bins=30, density=True)
            
            plt.xlabel(f'{self.component_names[0]}')
            plt.ylabel('Density')
            plt.title(f'Distribution of {self.component_names[0]} by Class')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print component explanations
        print("\n" + "="*60)
        print("PRINCIPAL COMPONENT INTERPRETATIONS")
        print("="*60)
        
        explanations = self.explain_components(n_features=5)
        for pc, features in explanations.items():
            var_explained = self.pca.explained_variance_ratio_[int(pc[2:]) - 1]
            print(f"\n{pc} (explains {var_explained:.1%} of variance):")
            print(f"  Mainly represents: {', '.join(features[:3])}")
            print(f"  All contributors: {', '.join(features)}")
