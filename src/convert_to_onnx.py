import mlflow
import xgboost as xgb
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import onnxruntime as rt
import numpy as np
import time
import os
import sys

# Connecting to the correct experiment
experiment_name = "Metro_APU_Failure_Prediction"
try:
    mlflow.set_experiment(experiment_name)
except:
    print(f"Could not find experiment '{experiment_name}'. Make sure you ran train.py in this same folder.")
    sys.exit(1)

# Finding the latest run
print("Searching for latest MLflow run...")
runs = mlflow.search_runs(order_by=["start_time desc"])

if runs.empty:
    print("Error: No runs found in this experiment!")
    print("Tip: Check if the 'mlruns' folder exists in this directory.")
    sys.exit(1)

# Getting the most recent run
last_run = runs.iloc[0]
run_id = last_run.run_id
print(f"✅ Found latest Run ID: {run_id}")

# Loading the model using MLflow's native loader
logged_model = f'runs:/{run_id}/model'
print(f"Loading model from: {logged_model}")

try:
    xgboost_model = mlflow.xgboost.load_model(logged_model)
    print("Model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model. Error: {e}")
    sys.exit(1)

# Converting to ONNX
initial_type = [('float_input', FloatTensorType([None, 15]))] # 15 here represents the number of features

print("Converting to ONNX...")
onnx_model = onnxmltools.convert_xgboost(xgboost_model, initial_types = initial_type)

# Saving the Onnx Model 
onnx_path = "metro_model.onnx"
onnxmltools.utils.save_model(onnx_model, onnx_path)
print(f"Model saved to {onnx_path}")

# Benchmark XGBoost(Python ver) v/s ONNX Runtime
# lets make 1 row of dummy data for input
dummy_input = np.random.rand(1,15).astype(np.float32)

# Benchmarks

# XGBoost
print("Benchmarking XGBoost...")
start = time.time()
for _ in range(1000):
    # We pass the numpy array directly. 
    # The model might warn about missing feature names, but it won't crash.
    xgboost_model.predict(dummy_input)
end = time.time()
xgb_time = (end - start) / 1000
print(f"XGBoost Average Inference Time: {xgb_time * 1000:.4f} ms")

#ONNX Runtime
# To simulate edge device we create inference session
sess = rt.InferenceSession(onnx_path)
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

start = time.time()
for _ in range(1000):
    sess.run([label_name], {input_name: dummy_input})
end = time.time()
onnx_time = (end - start) / 1000
print(f"Onnx Average Inference Time: {onnx_time * 1000:.4f} ms")

# Concluded results
speedup = xgb_time / onnx_time
print(f"\n>>> ONNX is {speedup:.2f}x faster than standard XGBoost! <<<")
