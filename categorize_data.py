import csv
import os

def mark_fraudulent_users(cdr_file, fraudulent_file, output_file):
    """Mark fraudulent users in a CDR file."""
    # Load fraudulent numbers from fraudulentUsers.csv into a set for faster lookups
    with open(fraudulent_file, 'r') as f_fraud:
        fraudulent_reader = csv.reader(f_fraud)
        fraudulent_numbers = {row[0] for row in fraudulent_reader}  # Using set for quick lookup

    # Open the CDR file for processing
    with open(cdr_file, 'r') as f_cdr:
        cdr_reader = csv.reader(f_cdr)
        headers = next(cdr_reader)  # Read headers
        
        # Add a new column "fraudulent_user"
        headers.append('fraudulent_user')

        # Open the output file for writing
        with open(output_file, 'w', newline='') as f_output:
            writer = csv.writer(f_output)
            writer.writerow(headers)  # Write updated headers
            
            # Process each row in the CDR file
            for row in cdr_reader:
                # Skip malformed rows with fewer columns than expected
                if len(row) < len(headers) - 1:
                    #print(f"Skipping malformed row: {row}")
                    row.append(1)

                    #continue

                # Extract src_number and dst_number
                src_number = row[7] if len(row) > 7 else ""
                dst_number = row[11] if len(row) > 11 else ""

                # Check if either number is fraudulent
                is_fraudulent = int(src_number in fraudulent_numbers or dst_number in fraudulent_numbers)
               
                # Append the result to the row
               
                row.append(is_fraudulent)
                if is_fraudulent:
                    print(row)
                writer.writerow(row)  # Write updated row to output file

    #print(f"New file created: {output_file}")


def categorise_resursively(directory):
    """Process specific CDR files in a directory and mark fraudulent users."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file in ["Op_1_CDRTrace.csv", "International_CDRTrace.csv"]:
                input_path = os.path.join(root, file)
                
                # Dynamically locate the fraudulentUsers.csv file in the same directory
                fraudulent_file = os.path.join(root, "fraudulentUsers.csv")
                if not os.path.exists(fraudulent_file):
                    print(f"Error: Fraudulent users file not found in {root}")
                    continue
                
                output_path = os.path.splitext(input_path)[0] + "_flagged.csv"
                mark_fraudulent_users(input_path, fraudulent_file, output_path)
                print(f"Flagged {input_path}")


# Usage
cdr_directory = 'TestCDR'  
categorise_resursively(cdr_directory)
