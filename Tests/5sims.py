from model import FraudDetectionSystem 
from config import *

def test_complete_pipeline(train_data, x, num_epochs=NUM_EPOCHS, timer=PRINT_FREQUENCY, model_name = 'test_model'):

    # 1. Load training data
    model_name = f"{model_name}.pth"
    fraud_detector = FraudDetectionSystem()

    try:
        fraud_detector.load_model(model_name)
    except Exception as e:
        print("Previous model not found, proceeding to train a new model.")
        
    X, y, _ = fraud_detector.load_and_preprocess_data(train_data)
    
    # 2. Train model
    data_tensors = fraud_detector.prepare_training_data(X, y)
    print(data_tensors['X_train'].shape)
    fraud_detector.train(data_tensors, num_epochs, timer)  # Shorter for testing
    
    # 3. Save model
    fraud_detector.save_model(model_name)
    
    # 4. Load model in new instance
    new_detector = FraudDetectionSystem()
    new_detector.load_model(model_name)
    
    # 5. Test prediction on new data
    results = new_detector.predict_new_data(x)
    
    # k = y.reshape((y.shape[0], 1))
    # Z = X+k
    # new_detector.visualize_data(Z)
    
    print("Pipeline test completed successfully!")
    print("Results:", results)


test_complete_pipeline('TestCDR/all_naive/all_naive_12%_5sims/Op_1_CDRTrace_flagged.csv', 
                       'TestCDR/all_naive/all_naive_12%_5sims/Op_1_CDRTrace_test_split.csv',NUM_EPOCHS, PRINT_FREQUENCY, model_name="all_naive_12%")

print("***************************************************************************************************************************")
test_complete_pipeline('TestCDR/advanced_mobility/mobility_12%_5sims/Op_1_CDRTrace_flagged.csv', 
                       'TestCDR/advanced_mobility/mobility_12%_5sims/Op_1_CDRTrace_test_split.csv', NUM_EPOCHS, PRINT_FREQUENCY, "mobility_12%")

