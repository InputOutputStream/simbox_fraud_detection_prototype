from ensemble import *
import pandas as pd
from config import *

# Initialize ensemble
ensemble = EnsembleSystem(stacking_method="meta")

# Define training data paths for each base model
train_data_paths = {
    "traffic": "TestCDR/advanced_traffic/traffic_12%_50sims/Op_1_CDRTrace_flagged.csv",
    "mobility": "TestCDR/advanced_mobility/mobility_12%_50sims/Op_1_CDRTrace_flagged.csv",
    "social": "TestCDR/advanced_social/social_12%_50sims/Op_1_CDRTrace_flagged.csv",
    "all_naive": "TestCDR/all_naive/all_naive_12%_50sims/Op_1_CDRTrace_flagged.csv"
}

# Fit base models
ensemble.fit_base_models(train_data_paths, num_epochs=NUM_EPOCHS, timer=PRINT_FREQUENCY)

# Load validation data for meta-learner training
X_val, y_val = ensemble.load_validation_data()

ensemble.fit(X_val, y_val)

# Make predictions
X_test, y_test = load_test_data()
results = ensemble.evaluate(X_test, y_test)

# Visualize
ensemble.visualize_ensemble_performance(X_test, y_test)

# Save ensemble
ensemble.save_ensemble()