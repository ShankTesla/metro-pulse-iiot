import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.xgboost

# Config
DATA_PATH = "./data/MetroPT3.csv" 
TARGET_COLUMN = "y"

# Loading Data
print("Loading data...")
df = pd.read_csv(DATA_PATH)

# Converting timestamp to datetime objects (Essential for labeling)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')

# Label Engineering
def get_labels(df):
    """
    Maps maintenance logs to the dataset.
    Source: UCI MetroPT-3 Dataset Description
    """
    # Initializing everything as 0 (0 means Normal)
    df['y'] = 0
    
    # Defined Failure Periods (According to dataset)
    # Failure 1: Air Leak
    df.loc[(df['timestamp'] >= '2020-04-18 00:00:00') & 
           (df['timestamp'] <= '2020-04-18 23:59:00'), 'y'] = 1
           
    # Failure 2: Air Leak (High Stress)
    df.loc[(df['timestamp'] >= '2020-05-29 23:30:00') & 
           (df['timestamp'] <= '2020-05-30 06:00:00'), 'y'] = 1
           
    # Failure 3: Air Leak
    df.loc[(df['timestamp'] >= '2020-06-05 10:00:00') & 
           (df['timestamp'] <= '2020-06-07 14:30:00'), 'y'] = 1
           
    # Failure 4: Oil Leak
    df.loc[(df['timestamp'] >= '2020-07-15 14:30:00') & 
           (df['timestamp'] <= '2020-07-15 19:00:00'), 'y'] = 1
           
    return df

print("Labeling data based on maintenance logs...")
df = get_labels(df)

# Checking if we actually found failures
print(f"Normal samples: {len(df[df['y']==0])}")
print(f"Failure samples: {len(df[df['y']==1])}")

# --------------------------------------------

# Preparing for Training
# Drop non-feature columns
X = df.drop(columns=['timestamp', TARGET_COLUMN, 'Unnamed: 0'], errors='ignore')
y = df[TARGET_COLUMN]

# Time-Based Splitting (Train on past, Test on future)
# We shall use 70% for train to ensure we capture the first 3 failures in training
# and leave Failure 4 for the test set.
split_index = int(len(df) * 0.7)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print(f"Training on {len(X_train)} rows, Testing on {len(X_test)} rows.")

# Defining Hyperparameters
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "n_jobs": -1 # -1 means use all CPU cores 
}

# Starting the MLflow Run
mlflow.set_experiment("Metro_APU_Failure_Prediction")

with mlflow.start_run():
    print("Training Model...")
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    print("Predicting...")
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Logging everything
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    # Logging the model for later use
    mlflow.xgboost.log_model(model, "model")
    
    print("Run is complete. View results with 'mlflow ui'") # message to note to self in future