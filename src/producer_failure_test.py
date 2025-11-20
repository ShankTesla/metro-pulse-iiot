# ... imports and kafka setup remain the same ...
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

# LOAD DATA
print("Loading dataset to find failure events...")
df = pd.read_csv("./data/MetroPT3.csv")

# TIME TRAVEL: Jump to the failure date (April 18, 2020)
# We filter for data specifically on the day the Air Leak started
failure_df = df[df['timestamp'].str.startswith("2020-04-18")]

print(f"Found {len(failure_df)} failure rows. Streaming 'CHAOS MODE' now...")

# Iterate only through the bad data
for index, row in failure_df.iterrows():
    data = row.to_dict()
    
    # Send to Redpanda
    producer.send(KAFKA_TOPIC, value=data)
    
    print(f"Sent (Failure Event): {data['timestamp']}")
    
    # Let's speed it up slightly to see results faster
    time.sleep(0.05) 

producer.flush()