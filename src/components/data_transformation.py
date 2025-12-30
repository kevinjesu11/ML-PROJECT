import numpy as np
import pandas as pd

class DataTransformation:
    def transform(self, df):
      
        cost_col = "CostInBillingCurrency"

        # Sort by date
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        # Feature engineering
        df['rolling_mean_7'] = df[cost_col].rolling(7).mean()
        df['rolling_std_7'] = df[cost_col].rolling(7).std()
        df['pct_change'] = df[cost_col].pct_change()

        df['zscore'] = (
            (df[cost_col] - df[cost_col].mean()) /
            df[cost_col].std()
        )

        df.fillna(0, inplace=True)

        # Feature columns
        features = [
            cost_col,
            'rolling_mean_7',
            'rolling_std_7',
            'pct_change',
            'zscore'
        ]

        X = df[features].copy()

        # Numerical safety
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)
        X = X.clip(lower=-1e6, upper=1e6)

        return df, X
