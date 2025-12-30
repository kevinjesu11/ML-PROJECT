from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import numpy as np

class TrainPipeline:
    def run(self):
        print("Starting training pipeline...")

        # 1. Load data
        df = DataIngestion().load_data()

        # 2. Transform data
        df, X = DataTransformation().transform(df)

        # 3. Train Isolation Forest
        trainer = ModelTrainer()
        iso_model, scaler = trainer.train_isolation_forest(X)

        # 4. Isolation Forest predictions
        X_scaled = scaler.transform(X)
        df['if_anomaly'] = iso_model.predict(X_scaled)

        # 5. Z-score anomaly
        df['z_anomaly'] = np.where(abs(df['zscore']) > 3, -1, 1)

        # 6. Ensemble label
        df['final_anomaly'] = np.where(
            (df['if_anomaly'] == -1) & (df['z_anomaly'] == -1),
            -1,
            1
        )

        # 7. Prepare RF labels
        df['rf_label'] = df['final_anomaly'].map({1: 0, -1: 1})

        # 8. Train Random Forest
        rf_model = trainer.train_random_forest(X, df['rf_label'])

        print("Training pipeline completed successfully ✔")

        return df, iso_model, rf_model


if __name__ == "__main__":
    TrainPipeline().run()
