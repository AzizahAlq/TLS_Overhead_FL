#!/usr/bin/env python3
"""
fedavg_mnist_exp1_aggregator_plain_sync.py

- INSECURE gRPC server (NO TLS)
- TRUE FedAvg FULL MODEL Aggregator
- SAME model, SAME weight structure, SAME pickle protocol as TLS
- SAME barrier logic
"""

import os, random, time, csv, pickle, threading
from concurrent import futures
from datetime import datetime

import numpy as np
import psutil
import grpc
import tensorflow as tf
from tensorflow.keras import models, layers

import myproto_pb2
import myproto_pb2_grpc


# ----------------- Reproducibility -----------------
SEED = 123
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ----------------- Config -----------------
NUM_CLIENTS = 6
PORT = 50051

INPUT_SHAPE = (28, 28, 1)
SHARED_DIM = 8

COMM_LOG = "fedavg_plain_aggregator_comm_log.csv"
RES_LOG  = "fedavg_plain_aggregator_resources.csv"
INTERVAL_SEC = 0.5


def mb(x): return x / (1024 * 1024)


# ----------------- Resource Monitor -----------------
def monitor_self(out_csv, interval_sec, stop_event):
    p = psutil.Process(os.getpid())
    ncores = psutil.cpu_count(logical=True) or 1
    try: p.cpu_percent(None)
    except: pass

    psutil.cpu_percent(None)
    time.sleep(0.2)
    psutil.cpu_percent(None)

    start = time.time()
    peak_rss = 0.0

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_sec","cpu_raw","cpu_norm","rss_mb","peak_rss_mb"])

        while not stop_event.is_set():
            t = time.time() - start
            cpu_raw = p.cpu_percent(interval=None)
            cpu_norm = cpu_raw / ncores
            rss = mb(p.memory_info().rss)
            peak_rss = max(peak_rss, rss)

            w.writerow([f"{t:.2f}", f"{cpu_raw:.2f}",
                        f"{cpu_norm:.2f}",
                        f"{rss:.2f}",
                        f"{peak_rss:.2f}"])
            f.flush()
            time.sleep(interval_sec)


# ----------------- Aggregator -----------------
class FedAvgAggregator(myproto_pb2_grpc.AggregatorServicer):

    def __init__(self):
        self.current_round = 0
        self.client_updates = {}
        self.lock = threading.Lock()
        self.round_start_time = None

        self.global_weights = self._init_global_weights()

        if not os.path.exists(COMM_LOG):
            with open(COMM_LOG, "w", newline="") as f:
                csv.writer(f).writerow([
                    "server_round","client_id",
                    "bytes_received_update",
                    "bytes_sent_global",
                    "num_samples",
                    "round_duration_sec",
                    "timestamp"
                ])

        print("[PLAIN-AGG] Initialized.")


    def _build_model(self):
        # IDENTICAL TO TLS VERSION
        return models.Sequential([
            layers.Input(shape=INPUT_SHAPE),

            layers.Conv2D(8, 3),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(2),

            layers.Conv2D(8, 3),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(2),

            layers.GlobalMaxPooling2D(),
            layers.Dense(SHARED_DIM, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ])


    def _init_global_weights(self):
        m = self._build_model()
        _ = m(np.zeros((1, *INPUT_SHAPE), dtype=np.float32), training=False)
        return m.get_weights()


    def _fedavg(self, updates):
        total = sum(int(n) for _, n in updates)
        L = len(self.global_weights)

        new_w = []
        for li in range(L):
            layer_sum = None
            for (w_list, n) in updates:
                w = np.nan_to_num(w_list[li])
                layer_sum = n*w if layer_sum is None else layer_sum + n*w
            new_w.append(layer_sum / total)
        return new_w


    # ----------------- RPC -----------------
    def SendModelUpdate(self, request, context):

        md = dict(context.invocation_metadata())
        client_id = md.get("client_id", "")

        if int(request.round) != self.current_round:
            return myproto_pb2.Ack(
                status=f"REJECTED: server_round={self.current_round}",
                current_round=self.current_round
            )

        weights = pickle.loads(request.weights)
        num_samples = int(request.num_samples)
        bytes_received = len(request.weights)

        with self.lock:

            if len(self.client_updates) == 0:
                self.round_start_time = time.time()

            self.client_updates[client_id] = (weights, num_samples, bytes_received)

            if len(self.client_updates) < NUM_CLIENTS:
                return myproto_pb2.Ack(
                    status="WAITING",
                    current_round=self.current_round
                )

            # Barrier reached
            duration = time.time() - self.round_start_time
            finished_round = self.current_round

            updates_for_avg = [(w,n) for (_, (w,n,_)) in self.client_updates.items()]
            self.global_weights = self._fedavg(updates_for_avg)

            global_blob = pickle.dumps(
                self.global_weights,
                protocol=pickle.HIGHEST_PROTOCOL
            )
            bytes_sent = len(global_blob)

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(COMM_LOG, "a", newline="") as f:
                wr = csv.writer(f)
                for cid, (_, n, b_up) in self.client_updates.items():
                    wr.writerow([
                        finished_round,
                        cid,
                        b_up,
                        bytes_sent,
                        n,
                        f"{duration:.6f}",
                        ts
                    ])

            self.client_updates.clear()
            self.current_round += 1

            return myproto_pb2.Ack(
                status="OK",
                current_round=self.current_round
            )


    def GetGlobalModel(self, request, context):
        blob = pickle.dumps(
            self.global_weights,
            protocol=pickle.HIGHEST_PROTOCOL
        )
        return myproto_pb2.ModelResponse(
            weights=blob,
            round=self.current_round
        )


# ----------------- MAIN -----------------
def main():

    stop = threading.Event()
    mon = threading.Thread(
        target=monitor_self,
        args=(RES_LOG, INTERVAL_SEC, stop),
        daemon=True
    )
    mon.start()

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=30)
    )

    myproto_pb2_grpc.add_AggregatorServicer_to_server(
        FedAvgAggregator(), server
    )

    # 🔥 ONLY difference from TLS:
    server.add_insecure_port(f"[::]:{PORT}")

    server.start()
    print(f"[PLAIN-AGG] Listening on :{PORT}")

    try:
        server.wait_for_termination()
    finally:
        stop.set()
        mon.join()


if __name__ == "__main__":
    main()