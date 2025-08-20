import csv
import os
from sklearn.model_selection import train_test_split

def split_flagged_file(directory, file_names, test_size=0.2, random_state=None):
    """Split flagged files into train and test datasets."""
    for root, _, files in os.walk(directory):
        for file in files:
            for name in file_names:
                flagged_name = os.path.splitext(name)[0] + "_flagged.csv"
                if file == flagged_name:
                    flagged_path = os.path.join(root, file)
                    
                    # Load the flagged file
                    with open(flagged_path, 'r') as f_flagged:
                        reader = csv.reader(f_flagged)
                        headers = next(reader)  # Get headers
                        data = list(reader)
                    
                    if not data:
                        print(f"No data found in file: {flagged_path}")
                        continue
                    
                    # Perform train-test split
                    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
                    
                    # Overwrite the flagged file with training data
                    with open(flagged_path, 'w', newline='') as f_flagged:
                        writer = csv.writer(f_flagged)
                        writer.writerow(headers)
                        writer.writerows(train_data)
                    
                    # Create the test split file
                    test_file = os.path.splitext(flagged_path)[0].replace("_flagged", "_test_split") + ".csv"
                    with open(test_file, 'w', newline='') as f_test:
                        writer = csv.writer(f_test)
                        writer.writerow(headers)
                        writer.writerows(test_data)
                    
                    print(f"Train split written to: {flagged_path}")
                    print(f"Test split created: {test_file}")

# Usage
cdr_directory = 'TestCDR'  
file_names = ["Op_1_CDRTrace.csv", "International_CDRTrace.csv"]  # Target file names
test_size = 0.2  # 20% of the data will be used for testing
random_state = 42  # Set seed for reproducibility

split_flagged_file(cdr_directory, file_names, test_size=test_size, random_state=random_state)
