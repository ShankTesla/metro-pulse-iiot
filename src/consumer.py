import json
import time
import numpy as np
import onnxruntime as rt 
from kafka import KafkaConsumer 

# Configuration [Set kafka topic, the localhost of client, onnx model]
KAFKA_TOPIC = "metro-sensor-data" 
BOOTSTRAP_SERVERS = ['localhost:19092']
MODEL_PATH = '../models/metro_model.onnx'

# Loading ONNX Model (The "Edge" Brain)
print("Loading ONNX model...")
sess = rt.InferenceSession(MODEL_PATH)
input_name = sess.get_inputs()[0].name 
label_name = sess.get_outputs()[0].name 

# Initialize Consumer
print("Connecting to Redpanda...")
consumer = KafkaConsumer(
    KAFKA_TOPIC, 
    bootstrap_servers = BOOTSTRAP_SERVERS,
    auto_offset_reset = 'latest', # start reading from Now (present), not past
    value_deserializer = lambda x: json.loads(x.decode('utf-8')) 
)

print("Listening for train sensor data....")

# Inference Loop
for message in consumer:
    data = message.value
    timestamp = data['timestamp']

    # preprocessing the stramed data in exact order the model expects
    # and so we have to drop 'timestamp', 'y', and 'Unnamed: 0' just like in training loop
    features = [
        data['TP2'], data['TP3'], data['H1'], data['DV_pressure'],
        data['Reservoirs'], data['Oil_temperature'], data['Motor_current'],
        data['COMP'], data['DV_eletric'], data['Towers'], data['MPG'],
        data['LPS'], data['Pressure_switch'], data['Oil_level'], 
        data['Caudal_impulses']
    ]
    
    #converting data to float32 numpy format (batch size of 1)
    input_data = np.array([features], dtype=np.float32)

    #Running the actual inference
    # Since sess.run returns a list, [0] is usualy the label (which is 0 or 1)
    pred_onnx = sess.run([label_name], {input_name: input_data})[0]
    predictions = pred_onnx[0]

    # Sending us alert logic
    status = "🔴 WARNING! FAILURE DETECTED"if predictions == 1 else "🟢 Normal functioning"
    print(f"[{timestamp}] Satus: {status} | Sensors: {features[:3]}....")



