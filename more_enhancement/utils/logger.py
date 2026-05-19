# utils/logger.py

import csv
import os

LOG_FILE = "prediction_logs.csv"

def log_prediction(text, prediction, confidence, timestamp):
    header = ["text", "prediction", "confidence", "timestamp"]
    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, "a", newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([text, prediction, confidence, timestamp])
