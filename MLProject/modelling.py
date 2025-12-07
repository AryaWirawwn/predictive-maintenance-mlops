import pandas as pd
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import mlflow

dataPath = 'dataset_mesin_membangun_sistem_machine_learning_preprocessing.csv'
experimentName = 'Predictive Maintenance - Model'

mlflow.set_tracking_uri("file:./mlruns")

def run_model(args):
    mlflow.set_experiment(experimentName)
    mlflow.sklearn.autolog(log_input_examples=True)
    
    print('Training dimulai...')

    try:
        df = pd.read_csv(dataPath)
        print('Data berhasil diload')
    except FileNotFoundError:
        print(f"Error: Dataset tidak ditemukan di {dataPath}.")
        return
    
    x = df.drop(['Target', 'Failure Type'], axis=1)
    y = df['Target']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print('Data berhasil di split')
    
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    with mlflow.start_run() as run:
        print('\nMemulai pelatihan model...')
        mlflow.log_param("data_path", dataPath)

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            max_depth=args.max_depth,
            bootstrap=args.bootstrap,
            random_state=42
        )
        
        model.fit(x_train_scaled, y_train)
        score = model.score(x_test_scaled, y_test)
        
        print(f"Akurasi: {score}")
        print(f"Pelatihan selesai. Run ID: {run.info.run_id}")
        print(f"Artifacts tersimpan di: {run.info.artifact_uri}")

        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)
        print("Run ID berhasil disimpan ke run_id.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--min_samples_split", type=int, default=5)
    parser.add_argument("--min_samples_leaf", type=int, default=4)
    parser.add_argument("--max_features", type=float, default=0.5)
    parser.add_argument("--max_depth", type=int, default=15)
    parser.add_argument("--bootstrap", type=bool, default=True)

    args = parser.parse_args()
    
    run_model(args)