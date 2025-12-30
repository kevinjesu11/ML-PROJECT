from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class ModelTrainer:

    def train_isolation_forest(self, X):
   
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        iso_model = IsolationForest(
            n_estimators=300,
            contamination=0.03,
            random_state=42
        )

        iso_model.fit(X_scaled)
        return iso_model, scaler

    def train_random_forest(self, X, y):
      
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        )

        rf_model.fit(X, y)
        return rf_model
