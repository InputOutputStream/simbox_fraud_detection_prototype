from typing import List, Dict, Any, Optional
import os

# =============================================================================
# BASIC SETTINGS
# =============================================================================

# Reproducibility
RANDOM_SEED: int = 42

# Logging
LOG_LEVEL: str = "INFO"

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# File paths

"""
for example, 
train_data_paths = {
    "traffic": "TestCDR/advanced_traffic/traffic_12%_50sims/Op_1_CDRTrace_flagged.csv",
    "mobility": "TestCDR/advanced_mobility/mobility_12%_50sims/Op_1_CDRTrace_flagged.csv",
    "social": "TestCDR/advanced_social/social_12%_50sims/Op_1_CDRTrace_flagged.csv",
    "all_naive": "TestCDR/all_naive/all_naive_12%_50sims/Op_1_CDRTrace_flagged.csv"
}
"""

TRAIN_DATA_PATHS = {
    "traffic": "TestCDR/advanced_traffic/",
    "mobility": "TestCDR/advanced_mobility/",
    "social": "TestCDR/advanced_social/",
    "all_naive": "TestCDR/all_naive/"
}

TRAIN_DATA_PATH: str = "TestCDR/"
TEST_DATA_PATH: str = "TestCDR/"
VALIDATION_DATA_PATH: str = "TestCDR/"

# Column definitions
TARGET_COLUMN: str = 'fraudulent_user'

# Feature columns for model input
FEATURE_COLUMNS: List[str] = [
    'size', 'src_number', 'dst_number', 'duration', 'hour', 
    'CDR_type', 'dst_imei', 'src_imei', 'minute', 'second', 
    'current_src_cellId'
]

# Categorical columns that need encoding
CATEGORICAL_COLUMNS: List[str] = [
    'src_number', 'CDR_type', 'src_imei', 'initial_src_cellId', 
    'current_src_cellId', 'dst_number', 'dst_imei', 
    'initial_dst_cellId', 'current_dst_cellId', 'year'
]

# Numerical columns for robustness testing
NUMERICAL_COLUMNS: List[str] = [
    'size', 'duration', 'hour', 'minute', 'second'
]

# Data split
TEST_SIZE: float = 0.2
VALIDATION_SIZE: float = 0.1

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

# Network structure
INPUT_SIZE: int = len(FEATURE_COLUMNS)
HIDDEN_SIZE: int = 64
OUTPUT_SIZE: int = 2
DROPOUT_RATE: float = 0.2

# Alternative architectures for experiments
ARCHITECTURES: Dict[str, Dict[str, Any]] = {
    "small": {"hidden_size": 32, "dropout_rate": 0.1},
    "medium": {"hidden_size": 64, "dropout_rate": 0.2},
    "large": {"hidden_size": 128, "dropout_rate": 0.3},
    "deep": {"hidden_size": 64, "dropout_rate": 0.25}  # Can be extended to multiple layers
}

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

# Training settings
NUM_EPOCHS: int = 20
LEARNING_RATE: float = 0.001
BATCH_SIZE: int = 32  # For future batch implementation

# Optimizer
OPTIMIZER: str = "Adam"
WEIGHT_DECAY: float = 1e-5

# Monitoring
PRINT_FREQUENCY: int = 10
VALIDATION_FREQUENCY: int = 10

# Early stopping (optional)
USE_EARLY_STOPPING: bool = False
EARLY_STOPPING_PATIENCE: int = 20

# =============================================================================
# DATA PREPROCESSING
# =============================================================================

# Scaling method
SCALING_METHOD: str = "StandardScaler"  # StandardScaler, MinMaxScaler

# PCA settings
USE_PCA: bool = True #False
PCA_VARIANCE_THRESHOLD: float = 0.95
PCA_N_COMPONENTS: Optional[int] = None  # If None, determined by variance threshold
PCA_WHITEN: bool = True #false  # Whiten components to remove correlations

# Feature engineering
CREATE_POLYNOMIAL_FEATURES: bool = False
POLYNOMIAL_DEGREE: int = 2

# =============================================================================
# IMBALANCED DATA HANDLING
# =============================================================================

# Sampling strategy
APPLY_SAMPLING: bool = True
SAMPLING_METHOD: str = "combined"  # "smote", "undersampling", "combined"

# SMOTE parameters
SMOTE_K_NEIGHBORS: int = 5

# Combined sampling iterations
SAMPLING_ITERATIONS: int = 3

# =============================================================================
# MODEL EVALUATION
# =============================================================================

# Metrics to track
TRACK_METRICS: List[str] = ["accuracy", "precision", "recall", "f1_score", "auc"]

# Cross-validation for research
USE_CROSS_VALIDATION: bool = False
CV_FOLDS: int = 5

# Robustness testing
ROBUSTNESS_TEST: bool = True
NOISE_LEVELS: List[float] = [0.1, 0.2, 0.5]

# Threshold optimization
OPTIMIZE_THRESHOLD: bool = False

# =============================================================================
# VISUALIZATION
# =============================================================================

# Plot settings
FIGURE_SIZE: tuple = (15, 10)
PLOT_STYLE: str = "seaborn"

# Colors
BINARY_COLORS: List[str] = ['lightcoral', 'lightblue']
PLOT_COLORS: str = "viridis"

# Which plots to generate
GENERATE_PLOTS: Dict[str, bool] = {
    "data_distribution": True,
    "feature_correlation": True,
    "training_history": True,
    "confusion_matrix": True,
    "roc_curve": True,
    "pca_components": True,  # Plot principal components instead of feature importance
    "pca_variance": True     # Show explained variance by components
}

# Save plots
SAVE_PLOTS: bool = True
PLOT_FORMAT: str = "png"

# =============================================================================
# MODEL PERSISTENCE
# =============================================================================

# Directories
MODEL_DIR: str = "models"
RESULTS_DIR: str = "results"
PLOTS_DIR: str = "plots"
DATA_DIR: str = "TestCDR"

# Model naming
MODEL_NAME: str = "fraud_detector"
SAVE_BEST_MODEL: bool = True
SAVE_TRAINING_HISTORY: bool = True

# =============================================================================
# EXPERIMENT SETTINGS
# =============================================================================

# For research experiments
EXPERIMENT_NAME: str = "fraud_detection_research"

# Hyperparameter search (for research)
PARAM_SEARCH: Dict[str, List[Any]] = {
    "learning_rate": [0.001, 0.01, 0.0001],
    "hidden_size": [32, 64, 128],
    "dropout_rate": [0.1, 0.2, 0.3]
}

# Multiple runs for statistical significance
NUM_EXPERIMENT_RUNS: int = 1  # Set to 5-10 for research papers
STATISTICAL_TESTS: bool = False

# =============================================================================
# RESEARCH-SPECIFIC SETTINGS
# =============================================================================

# Ablation studies
ABLATION_STUDIES: Dict[str, bool] = {
    "without_pca": False,
    "without_sampling": False,
    "without_dropout": False,
    "different_architectures": False
}

# Comparison methods (for research papers)
BASELINE_METHODS: List[str] = ["LogisticRegression", "RandomForest", "SVM"]
COMPARE_WITH_BASELINES: bool = False

# Error analysis
DETAILED_ERROR_ANALYSIS: bool = False
ANALYZE_MISCLASSIFICATIONS: bool = False

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_directories():
    """Create necessary directories"""
    dirs = [MODEL_DIR, RESULTS_DIR, PLOTS_DIR, "data"]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

def get_model_filename(architecture: str = "medium", timestamp: bool = True) -> str:
    """Generate model filename"""
    filename = f"{MODEL_NAME}_{architecture}"
    
    if timestamp:
        from datetime import datetime
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename += f"_{timestamp_str}"
    
    return os.path.join(MODEL_DIR, f"{filename}.pth")

def get_config_summary() -> Dict[str, Any]:
    """Get a summary of current configuration for logging"""
    return {
        "model": {
            "architecture": f"{INPUT_SIZE}-{HIDDEN_SIZE}-{OUTPUT_SIZE}",
            "dropout": DROPOUT_RATE
        },
        "training": {
            "epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "optimizer": OPTIMIZER
        },
        "data": {
            "features": len(FEATURE_COLUMNS),
            "test_size": TEST_SIZE,
            "use_pca": USE_PCA,
            "apply_sampling": APPLY_SAMPLING
        },
        "evaluation": {
            "cv_folds": CV_FOLDS if USE_CROSS_VALIDATION else "None",
            "robustness_test": ROBUSTNESS_TEST
        }
    }

# =============================================================================
# QUICK CONFIGURATION PRESETS
# =============================================================================

class QuickConfig:
    """Quick configuration presets for common scenarios"""
    
    @staticmethod
    def quick_test():
        """Fast configuration for testing"""
        global NUM_EPOCHS, PRINT_FREQUENCY, ROBUSTNESS_TEST
        NUM_EPOCHS = 20
        PRINT_FREQUENCY = 10
        ROBUSTNESS_TEST = False
        GENERATE_PLOTS["training_history"] = True
        # Turn off other plots for speed
        for key in GENERATE_PLOTS:
            if key != "training_history":
                GENERATE_PLOTS[key] = False
    
    @staticmethod
    def research_paper():
        """Configuration for research paper results"""
        global NUM_EXPERIMENT_RUNS, USE_CROSS_VALIDATION, STATISTICAL_TESTS
        NUM_EXPERIMENT_RUNS = 5
        USE_CROSS_VALIDATION = True
        STATISTICAL_TESTS = True
        COMPARE_WITH_BASELINES = True
        # Enable all plots
        for key in GENERATE_PLOTS:
            GENERATE_PLOTS[key] = True
    
    @staticmethod
    def ablation_study():
        """Configuration for ablation studies"""
        global NUM_EXPERIMENT_RUNS
        NUM_EXPERIMENT_RUNS = 3
        # Enable ablation studies
        for key in ABLATION_STUDIES:
            ABLATION_STUDIES[key] = True

# =============================================================================
# INITIALIZATION
# =============================================================================

# Auto-create directories
create_directories()

# Configuration validation
def validate_basic_config():
    """Basic validation of essential parameters"""
    assert len(FEATURE_COLUMNS) > 0, "FEATURE_COLUMNS cannot be empty"
    assert NUM_EPOCHS > 0, "NUM_EPOCHS must be positive"
    assert 0 < TEST_SIZE < 1, "TEST_SIZE must be between 0 and 1"
    assert LEARNING_RATE > 0, "LEARNING_RATE must be positive"

# Validate on import
validate_basic_config()

# Print configuration summary when imported
if __name__ == "__main__":
    import json
    print("Configuration Summary:")
    print(json.dumps(get_config_summary(), indent=2))
    print(f"\nModel will be saved as: {get_model_filename()}")
    print("Configuration validated successfully!")

# Easy access to all settings
__all__ = [
    # Core settings
    'RANDOM_SEED', 'TARGET_COLUMN', 'FEATURE_COLUMNS', 'CATEGORICAL_COLUMNS',
    
    # Model
    'INPUT_SIZE', 'HIDDEN_SIZE', 'OUTPUT_SIZE', 'DROPOUT_RATE', 'ARCHITECTURES',
    
    # Training
    'NUM_EPOCHS', 'LEARNING_RATE', 'PRINT_FREQUENCY', 'USE_EARLY_STOPPING',
    
    # Data
    'TEST_SIZE', 'USE_PCA', 'APPLY_SAMPLING', 'SAMPLING_ITERATIONS',
    
    # Evaluation
    'ROBUSTNESS_TEST', 'NOISE_LEVELS', 'USE_CROSS_VALIDATION',
    
    # Visualization
    'GENERATE_PLOTS', 'SAVE_PLOTS', 'BINARY_COLORS',
    
    # Utilities
    'QuickConfig', 'create_directories', 'get_model_filename', 'get_config_summary'
]