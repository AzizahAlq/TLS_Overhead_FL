# TLS_Overhead_FL
Empirical evaluation of TLS-secured communication overhead in synchronous federated learning using gRPC-based FedAvg.

# TLS Overhead Evaluation in Federated Learning

This repository contains the experimental implementation used in the paper:

**"Empirical Evaluation of Transport-Layer Security Overhead in Federated Learning Systems."**

The project evaluates the performance impact of TLS-secured communication in synchronous federated learning environments implemented using gRPC and the Federated Averaging (FedAvg) algorithm.

## Key Features

- gRPC-based federated learning framework
- Plain and TLS-secured communication configurations
- Synchronous FedAvg aggregation
- Communication overhead measurement
- Resource monitoring (CPU, memory, RTT)
- Reproducible experiments with MNIST

## Repository Contents

- Federated learning client and server implementations
- Plain and TLS communication experiments
- Logging and performance measurement scripts
- Dataset preparation and experiment configuration

## Goal

The goal of this project is to quantify the real performance impact of TLS encryption on federated learning communication and distinguish encryption overhead from synchronization delays.

## Environment

The experiments were conducted in a controlled local environment with the following configuration:

- Hardware: Apple MacBook Pro (14-inch, 2021)
- Chip: Apple M1 Pro
- Memory: 16 GB RAM
- Operating System: macOS Sequoia 15.0.1
- Python Version: Python 3.10+

The federated learning system was executed using a **multi-process local deployment**, where the aggregation server and all six clients run as independent concurrent processes on the same host using localhost communication.

---

## Python Dependencies

The following Python libraries are required to run the experiments:

- tensorflow
- grpcio
- grpcio-tools
- numpy
- psutil
- protobuf

Install the dependencies using:

```bash
pip install tensorflow grpcio grpcio-tools numpy psutil protobuf

TLS_Overhead_FL
│
├── FEDAVG_plain/        # Federated learning implementation using plain gRPC
├── FEDAVG_TLS/          # Federated learning implementation using TLS-secured gRPC
├── certs/               # TLS certificates used for secure communication
├── logs/                # Communication and system monitoring logs
├── mnist.npz            # Local MNIST dataset file
└── README.md

