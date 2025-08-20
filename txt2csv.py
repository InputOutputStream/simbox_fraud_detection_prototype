import os
import csv

def process_timestamp(timestamp):
    """Split a timestamp into day, month, year, hour, minute, second."""
    date, time = timestamp.split()
    day, month, year = map(int, date.split('-'))
    hour, minute, second = map(int, time.split(':'))
    return day, month, year, hour, minute, second

def txt_to_csv(input_file, output_file):
    """Convert a single text file to a CSV file."""
    with open(input_file, 'r') as txt_file:
        lines = txt_file.readlines()

    with open(output_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Extract headers from the first line of the text file
        headers = lines[0].strip().split()
        # Handle timestamp splitting in headers
        headerC = headers.copy()
        
        if "timestamp" in headers:
            index = headers.index("timestamp")
            headers = headers[:index] + ["day", "month", "year", "hour", "minute", "second"] + headers[index + 1:]

        csv_writer.writerow(headers)
        
        # Process remaining lines
        for line in lines[1:]:
            fields = line.strip().split()
            
            # Handle timestamp splitting in data
            if "timestamp" in headerC:
                index = headers.index("day")  # First timestamp-related field
                timestamp = line.split()
                timestamp = timestamp[0]+" "+timestamp[1]
                day, month, year, hour, minute, second = process_timestamp(timestamp)
                fields = [day, month, year, hour, minute, second] + fields[2:]
            csv_writer.writerow(fields)

def convert_directory_to_csv(directory):
    """Convert all text files in a directory and its subdirectories to CSV."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                input_path = os.path.join(root, file)
                output_path = os.path.splitext(input_path)[0] + ".csv"
                txt_to_csv(input_path, output_path)
                print(f"Converted {input_path} to {output_path}")

"""
    specifier le dossier source: par exemple chez moi 
    c'est FraudzenCDRs le nom du dossier source qui contient les datasets, 
    a l'interieur on retrouve advancedTraffic, advancedMobility etc
"""
directory_path = "TestCDR" 
convert_directory_to_csv(directory_path)
