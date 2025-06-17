import csv
import os
from datetime import datetime

def write_log(filename, prediction, confidence):
    log_file = 'logs/logs.csv'
    file_exists = os.path.isfile(log_file)

    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp', 'Filename', 'Prediction', 'Confidence'])
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            filename,
            prediction,
            f"{confidence:.2f}%"
        ])
