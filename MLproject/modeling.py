import os
import mlflow
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def main(data_path, n_estimators):
    print("Script mulai dieksekusi...")

    mlflow.set_tracking_uri("http://127.0.0.1:5000/")  
    mlflow.set_experiment("Energy Consumption Predictions")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File tidak ditemukan: {data_path}")

    data = pd.read_csv(data_path)

    target_column = 'EnergyConsumption'
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)

        mlflow.sklearn.log_model(model, artifact_path="MLproject")
        mlflow.log_artifact(data_path)

        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print("Semua data, model, dan artifact berhasil dilog ke MLflow!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--n_estimators", type=int, default=100)
    args = parser.parse_args()

    main(args.data_path, args.n_estimators)
