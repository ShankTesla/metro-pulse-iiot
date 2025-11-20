# Pseudocode - You write the real code
import pandas as pd
import time
import json
from kafka import KafkaProducer

KAFKA_TOPIC = "metro-sensor-data"
BOOTSTRAP_SERVERS = ['localhost:19092'] #show only external since clients will be using


producer = KafkaProducer(
    bootstrap_servers = BOOTSTRAP_SERVERS,
    value_serializer = lambda x: json.dumps(x).encode('utf-8') #built in serializer that serializes json into bytes
)

print("Producer started. Press Ctrl+C to stop.")
csv_file_path = "./data/MetroPT3.csv"

for chunk in pd.read_csv(csv_file_path, chunksize=1000):
    for index,row in chunk.iterrows():
        data = row.to_dict()

        # send data to producer
        producer.send(KAFKA_TOPIC, value=data)

        # print timestamp alone so console is clean and tidy
        print(f"Sent: {data['timestamp']}")

        # Simulate real-time interval (Adjust this if you want it faster/slower)
        time.sleep(0.1)

    producer.flush()

