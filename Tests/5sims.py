from model import FraudDetectionSystem 

# Test script to verify everything works
def test_complete_pipeline(train_data, x):
    # 1. Load training data
    fraud_detector = FraudDetectionSystem()
    X, y = fraud_detector.load_and_preprocess_data(train_data)
    
    # 2. Train model
    data_tensors = fraud_detector.prepare_training_data(X, y)
    fraud_detector.train(data_tensors, num_epochs=100, timer=10)  # Shorter for testing
    
    # 3. Save model
    fraud_detector.save_model('test_model.pth')
    
    # 4. Load model in new instance
    new_detector = FraudDetectionSystem()
    new_detector.load_model('test_model.pth')
    
    # 5. Test prediction on new data
    results = new_detector.predict_new_data(x)
    
    print("Pipeline test completed successfully!")
    print("Results:", results)


test_complete_pipeline('TestCDR/all_naive/all_naive_3%_5sims/Op_1_CDRTrace_flagged.csv', 
                       'TestCDR/all_naive/all_naive_3%_5sims/Op_1_CDRTrace_test_split.csv')

