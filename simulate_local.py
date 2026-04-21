import flwr as fl
import pandas as pd
from baseline_model.model import get_model, get_model_parameters, set_model_parameters
from baseline_model.custom_strategy import SaveModelStrategy
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    log_loss,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from server import aggregate_metrics


# Mock Client mirroring the production hospital clients
class MockClient(fl.client.NumPyClient):
    def __init__(self, hospital_id):
        self.hospital_id = hospital_id

        # Load local dummy data
        df = pd.read_parquet(
            f"local_simulation/dummy_data/hospital_{hospital_id}.parquet"
        )
        df.fillna(df.median(numeric_only=True), inplace=True)
        df.fillna(0, inplace=True)

        X = df.drop(columns=["SepsisLabel"]).values
        y = df["SepsisLabel"].values

        # Split just like the real hospitals
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = get_model()

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        set_model_parameters(self.model, parameters)
        self.model.fit(self.X_train, self.y_train)
        return self.get_parameters(config), len(self.X_train), {}

    def evaluate(self, parameters, config):
        set_model_parameters(self.model, parameters)

        y_pred = self.model.predict(self.X_val)
        y_proba = self.model.predict_proba(self.X_val)[:, 1]

        tn, fp, fn, tp = confusion_matrix(self.y_val, y_pred, labels=[0, 1]).ravel()
        cost_score = int(fp) + int(fn) * 5

        return (
            float(cost_score),
            len(self.X_val),
            {
                "cost_score": float(cost_score),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "true_negatives": int(tn),
                "auroc": float(roc_auc_score(self.y_val, y_proba)),
                "log_loss": float(log_loss(self.y_val, y_proba)),
                "accuracy": float(accuracy_score(self.y_val, y_pred)),
                "f1_score": float(f1_score(self.y_val, y_pred, zero_division=0)),
                "precision": float(
                    precision_score(self.y_val, y_pred, zero_division=0)
                ),
                "recall": float(recall_score(self.y_val, y_pred, zero_division=0)),
            },
        )


def client_fn(cid: str) -> fl.client.Client:
    return MockClient(hospital_id=int(cid) + 1).to_client()


if __name__ == "__main__":
    print("Starting Local Simulation...")
    strategy = SaveModelStrategy(
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
        evaluate_metrics_aggregation_fn=aggregate_metrics,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=5,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
