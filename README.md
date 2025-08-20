Convertir les fichiers .txt en .csv avec txt2csv.py.
Catégoriser les données avec categorize_data.py.
Entraîner le modèle 
Visualiser les résultats 


1. txt2csv.py (Convert .txt to .csv)
2. categorize_data.py (Mark fraudulent users)
3. train_test_split.py (Split into train/test)
4. model.py (Train and evaluate)


### pipe line

from model import FraudDetectionSystem

# Initialize the fraud detection system
fraud_detector = FraudDetectionSystem(
    model_params={'hidden_size': 64, 'dropout_rate': 0.2},
    random_state=42
)

# Load and preprocess data
X, y = fraud_detector.load_and_preprocess_data()

# Prepare training data
data_tensors = fraud_detector.prepare_training_data(X, y, apply_sampling=True)

# Train the model
history = fraud_detector.train(data_tensors, num_epochs=1000, lr=0.001, timer=100)

# Evaluate performance
results = fraud_detector.evaluate(data_tensors['X_test'], data_tensors['y_test'])

# Test robustness
robustness = fraud_detector.test_robustness(X, y)
print("Robustness results:", robustness)

# Save the model
fraud_detector.save_model('fraud_detection_model.pth')

# Visualize results
fraud_detector.visualize_results(data_tensors)
