import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour sauvegarder les images

from model import FraudDetectionSystem 
from config import *
import matplotlib.pyplot as plt
from pathlib import Path

def test_complete_pipeline(train_data, x, num_epochs=NUM_EPOCHS, timer=PRINT_FREQUENCY, model_name='test_model'):
    """Test complet du pipeline avec visualisations sauvegardées"""
    
    # 1. Configuration
    model_name_pth = f"{model_name}.pth"
    plot_dir = Path("plots") / model_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    fraud_detector = FraudDetectionSystem()

    # 2. Charger ou créer le modèle
    try:
        fraud_detector.load_model(model_name_pth)
        print(f"✓ Model {model_name_pth} loaded successfully")
    except Exception as e:
        print(f"✗ Previous model not found: {e}")
        print("→ Training new model...")
        
        # Charger et prétraiter les données
        X, y, _ = fraud_detector.load_and_preprocess_data(train_data)
        
        # Préparer les données d'entraînement
        data_tensors = fraud_detector.prepare_training_data(X, y)
        print(f"Training data shape: {data_tensors['X_train'].shape}")
        
        # Entraîner le modèle
        fraud_detector.train(data_tensors, num_epochs, timer=timer)
        
        # Sauvegarder le modèle
        fraud_detector.save_model(model_name_pth)
        print(f"✓ Model saved to {model_name_pth}")
        
        # Visualiser l'historique d'entraînement
        if fraud_detector.training_history:
            visualize_training_history(fraud_detector.training_history, 
                                      plot_dir / "training_history.png",
                                      sample_epoch=timer)
    
    # 3. Charger le modèle dans une nouvelle instance
    new_detector = FraudDetectionSystem()
    new_detector.load_model(model_name_pth)
    
    # 4. Prédiction sur nouvelles données
    print(f"\n→ Testing on: {x}")
    results = new_detector.predict_new_data(x, save_plots=False)
    
    # 5. Visualiser les performances
    visualize_results(results, plot_dir / "performance.png", model_name)
    
    # 6. Visualiser la matrice de confusion si disponible
    if 'classification_report' in results:
        print("\n" + "="*70)
        print(f"Classification Report for {model_name}:")
        print("="*70)
        print(results['classification_report'])
    
    print(f"\n✓ Pipeline test completed successfully for {model_name}!")
    print(f"✓ Visualizations saved to: {plot_dir}")
    
    return results


def visualize_training_history(history, save_path, sample_epoch=100):
    """Visualiser et sauvegarder l'historique d'entraînement"""
    plt.figure(figsize=(14, 5))
    
    # Training loss
    plt.subplot(1, 2, 1)
    train_loss = history.get('train_loss', [])
    if train_loss:
        sampled_losses = train_loss[::sample_epoch]
        epochs = range(0, len(train_loss), sample_epoch)
        plt.plot(epochs, sampled_losses, 'b-', linewidth=2)
        plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
        plt.xlabel(f'Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
    
    # Test accuracy
    plt.subplot(1, 2, 2)
    test_acc = history.get('test_accuracy', [])
    if test_acc:
        plt.plot(test_acc, 'r-', linewidth=2, marker='o', markersize=4)
        plt.title('Validation Accuracy Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Validation Step', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Ajouter la meilleure accuracy
        best_acc = max(test_acc)
        plt.axhline(y=best_acc, color='g', linestyle='--', alpha=0.5, 
                   label=f'Best: {best_acc:.4f}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Training history saved to {save_path}")


def visualize_results(results, save_path, model_name):
    """Visualiser les résultats de prédiction"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Afficher l'accuracy si disponible
    if 'accuracy' in results:
        accuracy = results['accuracy']
        
        # Créer un graphique de gauge simple
        categories = ['Accuracy']
        values = [accuracy * 100]
        
        bars = ax.barh(categories, values, color='skyblue', edgecolor='navy', linewidth=2)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Percentage (%)', fontsize=12)
        ax.set_title(f'Model Performance - {model_name}', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Ajouter le texte de l'accuracy
        for bar, value in zip(bars, values):
            ax.text(value + 1, bar.get_y() + bar.get_height()/2, 
                   f'{value:.2f}%', 
                   va='center', fontsize=12, fontweight='bold')
        
        # Ajouter une ligne de référence à 90%
        ax.axvline(x=90, color='green', linestyle='--', alpha=0.5, label='90% threshold')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No accuracy data available', 
               ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Performance visualization saved to {save_path}")


# ============================================================================
# Tests
# ============================================================================

results = {}

print("="*80)
print("STARTING FRAUD DETECTION TESTS")
print("="*80)

# Test 1: all_naive_12%
print("\n" + "="*80)
print("TEST 1: All Naive 12%")
print("="*80)
results["all_naive"] = test_complete_pipeline(
    'TestCDR/all_naive/all_naive_12%_5sims/Op_1_CDRTrace_flagged.csv', 
    'TestCDR/all_naive/all_naive_12%_5sims/Op_1_CDRTrace_test_split.csv',
    NUM_EPOCHS, 
    PRINT_FREQUENCY, 
    model_name="all_naive_12%"
)

# Test 2: mobility_12%
print("\n" + "="*80)
print("TEST 2: Advanced Mobility 12%")
print("="*80)
results["mobility"] = test_complete_pipeline(
    'TestCDR/advanced_mobility/mobility_12%_5sims/Op_1_CDRTrace_flagged.csv', 
    'TestCDR/advanced_mobility/mobility_12%_5sims/Op_1_CDRTrace_test_split.csv', 
    NUM_EPOCHS, 
    PRINT_FREQUENCY, 
    "mobility_12%"
)


# Test 2: traffic_12%
print("\n" + "="*80)
print("TEST 2: Advanced Traffic 12%")
print("="*80)
results["traffic"] = test_complete_pipeline(
    'TestCDR/advanced_traffic/traffic_12%_5sims/Op_1_CDRTrace_flagged.csv', 
    'TestCDR/advanced_traffic/traffic_12%_5sims/Op_1_CDRTrace_test_split.csv', 
    NUM_EPOCHS, 
    PRINT_FREQUENCY, 
    "traffic_12%"
)

# Test 2: social_12%
print("\n" + "="*80)
print("TEST 2: Advanced social 12%")
print("="*80)
results["social"] = test_complete_pipeline(
    'TestCDR/advanced_social/social_12%_5sims/Op_1_CDRTrace_flagged.csv', 
    'TestCDR/advanced_social/social_12%_5sims/Op_1_CDRTrace_test_split.csv', 
    NUM_EPOCHS, 
    PRINT_FREQUENCY, 
    "social_12%"
)

# Résumé final
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

for k, v in results.items():
    print(f"{k} 12% Accuracy: {v.get('accuracy', 'N/A'):.4f}" if 'accuracy' in v else "{k} 12%: No accuracy data")

print("="*80)
print("✓ All tests completed! Check the 'plots/' directory for visualizations.")
print("="*80)