import logging
import pandas as pd
from src.data_cleaning import DataCleaning, DataProcessStrategy

def get_data_for_test():
    try:
        df = pd.read_csv("./data/olist_customers_dataset.csv")
        df = df.sample(n=100)
        strategy = DataProcessStrategy()
        cleaned = DataCleaning(df, strategy)
        df = cleaned.handle_data()
        df.drop(["review_score"], axis=1, inplace=True)
        return df.to_json(orient="split")
    except Exception as e:
        logging.error(f"Error in get_data_for_test: {e}")
        raise e
