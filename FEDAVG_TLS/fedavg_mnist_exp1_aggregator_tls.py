#!/usr/bin/env python3
"""
fedavg_mnist_exp1_aggregator_tls.py
- TLS gRPC server
- TRUE FedAvg FULL MODEL Aggregator (sync barrier, waits for NUM_CLIENTS)
- Logs:
    fedavg_tls_aggregator_comm_log.csv:
      server_round, client_id, bytes_received_update, bytes_sent_global, num_samples, round_duration_sec, timestamp
    fedavg_tls_aggregator_resources.csv:
      cpu/mem time series for aggregator process
NO args/env.
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
NUM_CLIENTS = 6          # set to actual running clients (e.g., 2 for your test)
PORT = 50052

INPUT_SHAPE = (28, 28, 1)
SHARED_DIM = 8  # MUST match client model

COMM_LOG = "fedavg_tls_aggregator_comm_log.csv"
RES_LOG  = "fedavg_tls_aggregator_resources.csv"
INTERVAL_SEC = 0.5

CERT_FILE = "certs/server.crt"
KEY_FILE  = "certs/server.key"


def mb(x): return x / (1024 * 1024)


def monitor_self(out_csv: str, interval_sec: float, stop_event: threading.Event):
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
        w.writerow(["t_sec","pid","cpu_raw","cpu_norm","rss_mb","peak_rss_mb","sys_cpu","sys_used_mb"])
        while not stop_event.is_set():
            try:
                t = time.time() - start
                cpu_raw = p.cpu_percent(interval=None)
                cpu_norm = cpu_raw / ncores
                rss = mb(p.memory_info().rss)
                peak_rss = max(peak_rss, rss)
                sys_cpu = psutil.cpu_percent(interval=None)
                sys_used = mb(psutil.virtual_memory().used)
                w.writerow([f"{t:.2f}", p.pid, f"{cpu_raw:.2f}", f"{cpu_norm:.2f}",
                            f"{rss:.2f}", f"{peak_rss:.2f}", f"{sys_cpu:.2f}", f"{sys_used:.2f}"])
                f.flush()
            except Exception:
                break
            time.sleep(interval_sec)


class FedAvgAggregator(myproto_pb2_grpc.AggregatorServicer):
    def __init__(self):
        self.current_round = 0
        # client_id -> (weights, num_samples, bytes_received_update)
        self.client_updates = {}
        self.lock = threading.Lock()
        self.round_start_time = None

        self.global_weights = self._init_global_weights()
        self.last_global_blob_len = len(pickle.dumps(self.global_weights, protocol=pickle.HIGHEST_PROTOCOL))

        if not os.path.exists(COMM_LOG):
            with open(COMM_LOG, "w", newline="") as f:
                csv.writer(f).writerow([
                    "server_round","client_id",
                    "bytes_received_update","bytes_sent_global",
                    "num_samples","round_duration_sec","timestamp"
                ])

        print("[TLS-AGG] Global model initialized.")

    def _build_model(self):
        return models.Sequential([
            layers.Input(shape=INPUT_SHAPE),
            layers.Conv2D(8, 3, activation=None),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(2),

            layers.Conv2D(8, 3, activation=None),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(2),

            layers.GlobalMaxPooling2D(),
            layers.Dense(SHARED_DIM, activation="relu", name="shared_dense"),
            layers.Dense(10, activation="softmax"),
        ])

    def _init_global_weights(self):
        m = self._build_model()
        _ = m(np.zeros((1, *INPUT_SHAPE), dtype=np.float32), training=False)
        return m.get_weights()

    def _fedavg(self, updates):
        total = sum(max(int(n), 0) for _, n in updates)
        if total <= 0:
            return self.global_weights

        L = len(self.global_weights)
        new_w = []
        for li in range(L):
            layer_sum = None
            for (w_list, n) in updates:
                n = max(int(n), 0)
                w = np.nan_to_num(w_list[li])
                layer_sum = (n * w) if layer_sum is None else (layer_sum + n * w)
            new_w.append(layer_sum / total)
        return new_w

    # ----------------- RPCs -----------------
    def SendModelUpdate(self, request, context):
        md = dict(context.invocation_metadata())
        client_id = md.get("client_id", "")
        if not client_id:
            return myproto_pb2.Ack(status="ERROR: Missing client_id", current_round=self.current_round)

        client_round = int(request.round)
        if client_round != self.current_round:
            return myproto_pb2.Ack(
                status=f"REJECTED: server_round={self.current_round}, client_round={client_round}",
                current_round=self.current_round
            )

        try:
            weights = pickle.loads(request.weights)
        except Exception as e:
            return myproto_pb2.Ack(status=f"ERROR: Bad weights ({e})", current_round=self.current_round)

        num_samples = int(request.num_samples)
        bytes_received_update = len(request.weights)

        with self.lock:
            if len(self.client_updates) == 0:
                self.round_start_time = time.time()

            self.client_updates[client_id] = (weights, num_samples, bytes_received_update)

            print(f"[TLS-AGG] Round {self.current_round}: {len(self.client_updates)}/{NUM_CLIENTS} "
                  f"(from {client_id}, update_bytes={bytes_received_update})")

            if len(self.client_updates) < NUM_CLIENTS:
                return myproto_pb2.Ack(status="WAITING", current_round=self.current_round)

            # barrier met -> aggregate
            duration = (time.time() - self.round_start_time) if self.round_start_time else 0.0
            finished_round = self.current_round

            updates_for_avg = [(w, n) for (_, (w, n, _)) in self.client_updates.items()]
            self.global_weights = self._fedavg(updates_for_avg)

            # update global blob size (bytes sent to client via GetGlobalModel)
            self.last_global_blob_len = len(pickle.dumps(self.global_weights, protocol=pickle.HIGHEST_PROTOCOL))

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(COMM_LOG, "a", newline="") as f:
                wr = csv.writer(f)
                for cid, (_, n, b_up) in self.client_updates.items():
                    wr.writerow([
                        finished_round, cid,
                        b_up, self.last_global_blob_len,
                        n, f"{duration:.6f}", ts
                    ])

            self.client_updates.clear()
            self.current_round += 1
            self.round_start_time = None

            print(f"[TLS-AGG] Round {finished_round} done in {duration:.2f}s -> server_round={self.current_round}")
            return myproto_pb2.Ack(status="OK", current_round=self.current_round)

    def GetGlobalModel(self, request, context):
        blob = pickle.dumps(self.global_weights, protocol=pickle.HIGHEST_PROTOCOL)
        # (optional) keep updated
        self.last_global_blob_len = len(blob)
        return myproto_pb2.ModelResponse(weights=blob, round=self.current_round)


def main():
    # resource monitor
    stop = threading.Event()
    mon = threading.Thread(target=monitor_self, args=(RES_LOG, INTERVAL_SEC, stop), daemon=True)
    mon.start()

    # TLS creds
    with open(KEY_FILE, "rb") as f:
        private_key = f.read()
    with open(CERT_FILE, "rb") as f:
        certificate_chain = f.read()

    server_creds = grpc.ssl_server_credentials(((private_key, certificate_chain),))

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=30))
    myproto_pb2_grpc.add_AggregatorServicer_to_server(FedAvgAggregator(), server)
    server.add_secure_port(f"[::]:{PORT}", server_creds)
    server.start()

    print(f"[TLS-AGG] Listening on :{PORT} (TLS). Waiting for {NUM_CLIENTS} clients/round...")

    try:
        server.wait_for_termination()
    finally:
        stop.set()
        mon.join(timeout=5)


if __name__ == "__main__":
    main()