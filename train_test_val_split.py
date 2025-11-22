import csv
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import List, Optional
import numpy as np


def split_flagged_file(
    directory: str,
    file_names: List[str],
    val_size: float = 0.1,
    test_size: float = 0.2,
    random_state: Optional[int] = 42,
    label_column: str = 'fraudulent_user'
):
    """
    Split flagged files into train, validation, and test datasets with stratification.
    
    Args:
        directory: Root directory to search for files
        file_names: List of base file names (will search for *_flagged.csv)
        val_size: Validation set size (default: 0.1 = 10%)
        test_size: Test set size (default: 0.2 = 20%)
        random_state: Random seed for reproducibility
        label_column: Column name for stratification
    
    Creates:
        - *_flagged.csv (overwritten with train data)
        - *_val_split.csv (validation data)
        - *_test_split.csv (test data)
    """
    
    if val_size + test_size >= 1.0:
        raise ValueError(f"val_size ({val_size}) + test_size ({test_size}) must be < 1.0")
    
    train_size = 1.0 - val_size - test_size
    
    print("="*70)
    print(f"Split: Train={train_size:.1%}, Val={val_size:.1%}, Test={test_size:.1%}")
    print(f"Seed: {random_state}, Stratify by: {label_column}")
    print("="*70)
    
    processed = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            for name in file_names:
                flagged_name = Path(name).stem + "_flagged.csv"
                
                if file == flagged_name:
                    flagged_path = Path(root) / file
                    print(f"\nProcessing: {flagged_path}")
                    
                    try:
                        # Load data
                        with open(flagged_path, 'r') as f:
                            reader = csv.reader(f)
                            headers = next(reader)
                            data = list(reader)
                        
                        if not data:
                            print(f"  ✗ Empty file, skipping")
                            continue
                        
                        print(f"  Total rows: {len(data)}")
                        
                        # Get labels for stratification
                        try:
                            label_idx = headers.index(label_column)
                            labels = [row[label_idx] for row in data]
                            
                            # Check class balance
                            unique, counts = np.unique(labels, return_counts=True)
                            print(f"  Classes: {dict(zip(unique, counts))}")
                            
                            # First split: separate test set
                            train_val_data, test_data = train_test_split(
                                data, 
                                test_size=test_size,
                                random_state=random_state,
                                stratify=labels
                            )
                            
                            # Second split: separate validation from train
                            train_val_labels = [row[label_idx] for row in train_val_data]
                            val_size_adjusted = val_size / (1 - test_size)
                            
                            train_data, val_data = train_test_split(
                                train_val_data,
                                test_size=val_size_adjusted,
                                random_state=random_state,
                                stratify=train_val_labels
                            )
                            
                        except ValueError:
                            print(f"  ⚠ Column '{label_column}' not found, no stratification")
                            
                            # Split without stratification
                            train_val_data, test_data = train_test_split(
                                data, test_size=test_size, random_state=random_state
                            )
                            val_size_adjusted = val_size / (1 - test_size)
                            train_data, val_data = train_test_split(
                                train_val_data, test_size=val_size_adjusted, random_state=random_state
                            )
                        
                        print(f"  Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
                        
                        # Define output paths
                        base_path = flagged_path.parent / flagged_path.stem.replace("_flagged", "")
                        val_file = Path(str(base_path) + "_val_split.csv")
                        test_file = Path(str(base_path) + "_test_split.csv")
                        
                        # Write train (overwrite flagged file)
                        with open(flagged_path, 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(headers)
                            writer.writerows(train_data)
                        
                        # Write validation
                        with open(val_file, 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(headers)
                            writer.writerows(val_data)
                        
                        # Write test
                        with open(test_file, 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(headers)
                            writer.writerows(test_data)
                        
                        print(f"  ✓ Train: {flagged_path}")
                        print(f"  ✓ Val:   {val_file}")
                        print(f"  ✓ Test:  {test_file}")
                        
                        processed += 1
                        
                    except Exception as e:
                        print(f"  ✗ Error: {e}")
                        continue
    
    print(f"\n{'='*70}")
    print(f"Processed {processed} file(s)")
    print("="*70)


# Usage
cdr_directory = 'TestCDR'
file_names = ["Op_1_CDRTrace.csv", "International_CDRTrace.csv"]

split_flagged_file(
    directory=cdr_directory,
    file_names=file_names,
    val_size=0.1,    # 10% validation
    test_size=0.2,   # 20% test
    random_state=42  # 70% train
)