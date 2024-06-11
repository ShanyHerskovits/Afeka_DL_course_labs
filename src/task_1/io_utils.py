import os
import json


def save_report_to_file(report, filename):
    # Ensure the folder exists
    folder = "model_results"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Construct the full file path
    filepath = os.path.join(folder, f"{filename}.json")

    with open(filepath, "w") as f:
        json.dump(report, f)


def get_model_results(filename: str) -> dict:
    with open(filename, "r") as f:
        report = json.load(f)

    return report
