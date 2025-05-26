import argparse
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

parser = argparse.ArgumentParser()
parser.add_argument('--csv_url', type=str, required=True)
parser.add_argument('--target_var', type=str, required=True)
args = parser.parse_args()

mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Ubah jika pakai remote
mlflow.set_experiment("Energy Consumption Predictions")
mflow.sklearn.autolog()

with mlflow.start_run() as run:
    print(f"Running with run_id: {run.info.run_id}")
    data = pd.read_csv(args.csv_url)

    X = data.drop(columns=[args.target_var])
    y = data[args.target_var]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    mlflow.sklearn.log_model(model, artifact_path="model")
    mlflow.log_artifact(args.csv_url)

    print("Training and logging done.")
