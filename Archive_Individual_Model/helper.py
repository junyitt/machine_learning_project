import pandas as pd

def get_report_df(report, label = "train"):
    precision = report["1"]["precision"]
    recall = report["1"]["recall"]
    f1 = report["1"]["f1-score"]
    accuracy = report["accuracy"]

    report_df = pd.DataFrame([precision, recall, f1, accuracy]).T
    report_df.columns = ['precision', 'recall', 'f1', 'accuracy']
    report_df["sample"] = label
    return report_df
