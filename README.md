# TLS Overhead Evaluation in Federated Learning

Empirical evaluation of TLS-secured communication overhead in synchronous federated learning using gRPC-based FedAvg.

## Publication

This repository contains the experimental implementation associated with our paper:

**"Empirical Evaluation of TLS Communication Overhead in Federated Learning Systems"**

**Authors:** Azizah AlQahtani and Tarek Helmy

**Venue:** The 6th International Workshop on Software Security Engineering (SSE-26), EASE 2026  
**Location:** Glasgow, United Kingdom  
**Date:** 12 June 2026

Paper information:  
https://conf.researchr.org/details/ease-2026/secure-software-2026-papers/4/Empirical-Evaluation-of-TLS-Communication-Overhead-in-Federated-Learning-Systems

EASE 2026:  
https://conf.researchr.org/home/ease-2026

---

## Overview

This project evaluates the performance impact of TLS-secured communication in synchronous federated learning environments implemented using gRPC and the Federated Averaging (FedAvg) algorithm.

Two identical federated learning configurations are evaluated:

- Plain gRPC communication
- TLS-secured gRPC communication

The experiments measure the additional communication and computational overhead introduced by TLS while keeping the federated learning configuration unchanged.

## Key Features

- gRPC-based federated learning framework
- Plain and TLS-secured communication configurations
- Synchronous FedAvg aggregation
- Communication overhead measurement
- Round-trip time (RTT) measurement
- CPU and memory monitoring
- Local training and serialization measurements
- Reproducible experiments using MNIST

## Main Findings

The experimental results show that the overhead introduced by TLS is relatively small:

- Approximately 4% increase in total round time
- Approximately 3% increase in round-trip time (RTT)
- No observed effect on model accuracy or convergence

These results demonstrate that TLS can secure federated learning communication with limited performance overhead under the evaluated experimental conditions.

## Repository Structure

```text
TLS_Overhead_FL/
│
├── FEDAVG_plain/        # Federated learning using plain gRPC
├── FEDAVG_TLS/          # Federated learning using TLS-secured gRPC
├── certs/               # TLS certificates for secure communication
├── logs/                # Communication and system monitoring logs
├── mnist.npz            # Local MNIST dataset
└── README.md
```

## Experimental Environment

The experiments were conducted in a controlled local environment with the following configuration:

- Hardware: Apple MacBook Pro (14-inch, 2021)
- Chip: Apple M1 Pro
- Memory: 16 GB RAM
- Operating System: macOS Sequoia 15.0.1
- Python Version: Python 3.10+

The federated learning system uses a multi-process local deployment. The aggregation server and six federated clients run as independent concurrent processes on the same host using localhost communication.

## Python Dependencies

The following Python libraries are required:

- tensorflow
- grpcio
- grpcio-tools
- numpy
- psutil
- protobuf

Install the dependencies using:

```bash
pip install tensorflow grpcio grpcio-tools numpy psutil protobuf
```

## Research Goal

The goal of this project is to quantify the performance impact of TLS encryption on federated learning communication.

The evaluation focuses on distinguishing the overhead introduced by secure transport from other costs associated with synchronous federated learning, including local training, serialization, and communication delays.

## Citation

If you use this repository or build upon this work, please cite:

```bibtex
@inproceedings{alqahtani2026tls,
  title={Empirical Evaluation of TLS Communication Overhead in Federated Learning Systems},
  author={AlQahtani, Azizah and Helmy, Tarek},
  booktitle={The 6th International Workshop on Software Security Engineering (SSE-26), EASE 2026},
  year={2026},
  address={Glasgow, United Kingdom}
}
```

## Authors

- Azizah AlQahtani
- Tarek Helmy
