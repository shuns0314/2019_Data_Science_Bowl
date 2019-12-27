import numpy as np
import pandas as pd


RAW_PATH = "/data/raw"


def main():
    """main."""
    train_df = pd.read_csv(f"{RAW_PATH}/train.csv")
    train_labels_df = pd.read_csv(f"{RAW_PATH}/train_labels.csv")
    specs_df = pd.read_csv(f"{RAW_PATH}/specs.csv")
    