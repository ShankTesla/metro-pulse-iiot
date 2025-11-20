# MetroPulse: Edge-Optimized Predictive Maintenance Pipeline

![Status](https://img.shields.io/badge/Status-Active-success) ![Tech](https://img.shields.io/badge/Stack-Redpanda%20|%20Docker%20|%20ONNX-blue)

### 📋 Project Overview
**MetroPulse** is an end-to-end Industrial IoT (IIoT) system designed to predict Air Production Unit (APU) failures on urban transit trains in real-time. 

Unlike traditional batch-processing models, this project simulates a **streaming "Edge AI" environment**. It ingests high-frequency sensor data (MetroPT-3 dataset), detects anomalies using a quantized XGBoost model (ONNX), and triggers alerts with sub-millisecond latency.

**Business Value:** Proactive detection of compressor failures to prevent service disruptions and reduce maintenance costs (Predictive Maintenance).

---

### 🏗️ Architecture
The system follows a microservices-based Event-Driven Architecture:

`[Sensors / Digital Twin]`  **→** `[Redpanda (Message Bus)]`  **→** `[Inference Engine]`

1.  **Digital Twin (Producer):** Python service that replays historical sensor logs (pressure, temperature, motor current) to simulate live train telemetry.
2.  **Message Broker:** **Redpanda** (Kafka-compatible) running in Docker to handle high-throughput data streams.
3.  **Model Registry:** **MLflow** for tracking experiments and versioning model artifacts.
4.  **Inference Service (Consumer):** A lightweight Python service using **ONNX Runtime**. It consumes the stream and performs inference 3x faster than standard Scikit-learn pipelines.

---

### 🛠️ Tech Stack
* **Language:** Python 3.9+
* **Streaming:** Redpanda (Dockerized)
* **ML Frameworks:** XGBoost, Scikit-Learn
* **ML Ops:** MLflow, ONNX / ONNX Runtime
* **Data Source:** UCI MetroPT-3 Dataset

---

### 🚀 Key Results & Benchmarks
| Metric | Standard XGBoost | ONNX Runtime (Quantized) | Improvement |
| :--- | :--- | :--- | :--- |
| **Inference Latency** | ~0.15ms | ~0.04ms | **~3.7x Faster** |
| **Model Size** | 450 KB | 120 KB | **73% Smaller** |
| **Accuracy (F1)** | 0.99 | 0.99 | **Lossless** |

*Note: Benchmarks run on local CPU environment. ONNX optimization allows this model to be deployed on low-power edge devices (e.g., Raspberry Pi / Train ECU).*

---

### ⚙️ How to Run Locally

**1. Prerequisites**
* Docker Desktop
* Python 3.8+

**2. Clone & Setup**
```bash
git clone [https://github.com/yourusername/metro-pulse.git](https://github.com/yourusername/metro-pulse.git)
cd metro-pulse
pip install -r requirements.txt