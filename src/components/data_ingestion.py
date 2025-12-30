import pandas as pd

class DataIngestion:
    def load_data(self):
      
        df = pd.read_csv("notebook/anonymized_costs.csv")
        return df
