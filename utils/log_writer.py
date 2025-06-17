
import csv
from datetime import datetime
import os

LOG_FILE = 'logs.csv'

def write_log(filename, prediction, confidence):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp', 'Filename', 'Prediction', 'Confidence'])
        writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), filename, prediction, round(confidence, 2)])
