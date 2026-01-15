from src.components.data_transformation import DataTransformation
import numpy as np

class PredictPipeline:
    def predict(self, df, iso_model, rf_model, scaler):
    
        df, X = DataTransformation().transform(df)

        X_scaled = scaler.transform(X)
        df['if_anomaly'] = iso_model.predict(X_scaled)

        df['z_anomaly'] = np.where(abs(df['zscore']) > 3, -1, 1)

        df['ultimate_anomaly'] = np.where(
            (df['if_anomaly'] == -1) & (df['z_anomaly'] == -1),
            -1,
            1
        )

        return df
