#!/usr/bin/env python3
"""
fedavg_mnist_client_plain_pftlstyle.py

PLAIN gRPC (no TLS), but SAME logic style as your PFTL client:

  pull_global -> train -> push_update -> wait_new_round -> next

Logs:
  client_0_plain_comm_log.csv  (PFTL-style: bytes_sent/bytes_received/rtt aligned after wait)
  client_0_plain_resources.csv
  client_0_final_metrics.csv
"""

import os, time, csv, pickle, threading
from datetime import datetime

import numpy as np
import psutil
import grpc
import tensorflow as tf

import myproto_pb2
import myproto_pb2_grpc


# ===================== CONFIG =====================
CLIENT_ID = "client_1"
TARGET = "localhost:50051"

NUM_ROUNDS = 15
LOCAL_EPOCHS = 1
BATCH_SIZE = 128
SEED = 42

DATA_DIR = "data"
MNIST_LOCAL_PATH = os.path.join(DATA_DIR, "mnist.npz")

INTERVAL_SEC = 0.5
POLL_SEC = 0.3
SYNC_TIMEOUT_SEC = 120.0   # like your PFTL SYNC_TIMEOUT_SEC

COMM_LOG  = f"{CLIENT_ID}_plain_comm_log.csv"
RES_LOG   = f"{CLIENT_ID}_plain_resources.csv"
FINAL_LOG = f"{CLIENT_ID}_final_metrics.csv"
# ==================================================


def mb(x): return x / (1024 * 1024)


def ensure_mnist_local():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(MNIST_LOCAL_PATH):
        return os.path.abspath(MNIST_LOCAL_PATH)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    np.savez_compressed(MNIST_LOCAL_PATH, x_train=x_train, y_train=y_train,
                        x_test=x_test, y_test=y_test)
    return os.path.abspath(MNIST_LOCAL_PATH)


def load_mnist(path):
    d = np.load(path)
    x_train = (d["x_train"].astype("float32") / 255.0)[..., None]
    y_train = d["y_train"].astype("int64")
    x_test  = (d["x_test"].astype("float32") / 255.0)[..., None]
    y_test  = d["y_test"].astype("int64")
    return (x_train, y_train), (x_test, y_test)


def build_model():
    # MUST MATCH AGGREGATOR EXACTLY
    from tensorflow.keras import models, layers
    INPUT_SHAPE = (28, 28, 1)
    SHARED_DIM = 8
    m = models.Sequential([
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
    m.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return m


def monitor_self(out_csv, interval_sec, stop_event):
    p = psutil.Process(os.getpid())
    ncores = psutil.cpu_count(logical=True) or 1
    try: p.cpu_percent(None)
    except Exception: pass
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


def summarize_resources(resources_csv: str):
    peak_rss = None
    cpu_norm_vals = []
    with open(resources_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                rss = float(row["rss_mb"])
                cpu_norm = float(row["cpu_norm"])
                cpu_norm_vals.append(cpu_norm)
                peak_rss = rss if peak_rss is None else max(peak_rss, rss)
            except Exception:
                continue
    avg_cpu_norm = (sum(cpu_norm_vals) / len(cpu_norm_vals)) if cpu_norm_vals else None
    return {
        "peak_rss_mb": f"{peak_rss:.2f}" if peak_rss is not None else "",
        "avg_cpu_norm_percent": f"{avg_cpu_norm:.2f}" if avg_cpu_norm is not None else "",
    }


# ===================== PFTL-STYLE gRPC HELPERS =====================

def pull_global(stub):
    """Like pull_candidates(): returns (server_round, weights_list, bytes_received)."""
    resp = stub.GetGlobalModel(myproto_pb2.EmptyRequest())
    bytes_received = len(resp.weights) if resp.weights else 0
    weights = pickle.loads(resp.weights) if resp.weights else None
    return int(resp.round), weights, int(bytes_received)


def push_update(stub, server_round, weights_list, num_samples):
    """Like push_shared(): returns (bytes_sent, t0, ack_status, ack_current_round)."""
    payload = pickle.dumps(weights_list, protocol=pickle.HIGHEST_PROTOCOL)
    bytes_sent = len(payload)
    t0 = time.perf_counter()
    ack = stub.SendModelUpdate(
        myproto_pb2.ModelUpdate(weights=payload, round=int(server_round), num_samples=int(num_samples)),
        metadata=(("client_id", CLIENT_ID),)
    )
    return int(bytes_sent), float(t0), str(ack.status), int(ack.current_round)


def wait_new_round(stub, start_round, t0_perf):
    """
    Like your PFTL wait_new_round():
    Poll GetGlobalModel until round changes OR timeout.
    Returns: (new_round, new_weights, rtt_sec, bytes_received_last_poll)
    """
    wait_start = time.perf_counter()
    last_bytes_received = 0

    while True:
        time.sleep(POLL_SEC)
        resp = stub.GetGlobalModel(myproto_pb2.EmptyRequest())
        last_bytes_received = len(resp.weights) if resp.weights else 0
        r = int(resp.round)

        if r > int(start_round):
            rtt = time.perf_counter() - float(t0_perf)
            new_weights = pickle.loads(resp.weights) if resp.weights else None
            return r, new_weights, float(rtt), int(last_bytes_received)

        if (time.perf_counter() - wait_start) > float(SYNC_TIMEOUT_SEC):
            rtt = time.perf_counter() - float(t0_perf)
            # timeout: return same round and no new weights (matches your PFTL fallback)
            return int(start_round), None, float(rtt), int(last_bytes_received)


# ===================== MAIN =====================

def main():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    if not os.path.exists(COMM_LOG):
        with open(COMM_LOG, "w", newline="") as f:
            csv.writer(f).writerow([
                "round",
                "pulled_server_round",
                "bytes_received_global",
                "bytes_sent_update",
                "train_sec",
                "send_sec",
                "ack_status",
                "ack_current_round",
                "rtt_sec",
                "timestamp",
            ])

    mnist_path = ensure_mnist_local()
    (x_train, y_train), (x_test, y_test) = load_mnist(mnist_path)

    model = build_model()

    stop = threading.Event()
    mon = threading.Thread(target=monitor_self, args=(RES_LOG, INTERVAL_SEC, stop), daemon=True)
    mon.start()

    channel = grpc.insecure_channel(
        TARGET,
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ],
    )
    stub = myproto_pb2_grpc.AggregatorStub(channel)

    completed = 0
    print(f"[PLAIN-PFTLSTYLE {CLIENT_ID}] Need {NUM_ROUNDS} rounds. target={TARGET}")

    try:
        # initial pull (same as PFTL initial pull)
        server_round, w, bytes_in = pull_global(stub)
        if w is not None:
            model.set_weights(w)

        while completed < NUM_ROUNDS:
            # 1) pull global for this round
            server_round, w, bytes_in = pull_global(stub)
            if w is not None:
                model.set_weights(w)

            # 2) local train
            t_train0 = time.time()
            model.fit(x_train, y_train, epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE, verbose=0)
            train_sec = time.time() - t_train0

            # 3) push update (ONCE)
            update_weights = model.get_weights()
            t_send0 = time.time()
            bytes_out, t0_perf, ack_status, ack_cur = push_update(
                stub, server_round, update_weights, num_samples=len(x_train)
            )
            send_sec = time.time() - t_send0

            # 4) barrier wait (ONLY if WAITING; if OK we still can wait to align like PFTL if you want)
            rtt_sec = 0.0
            if ack_status == "WAITING":
                new_round, new_w, rtt_sec, bytes_in2 = wait_new_round(stub, server_round, t0_perf)
                # align received-bytes to the poll that advanced (PFTL style)
                bytes_in = bytes_in2
                if new_w is not None:
                    model.set_weights(new_w)
                # if server advanced, count as completed contribution
                if new_round > server_round:
                    completed += 1
            elif ack_status == "OK":
                # server already advanced (barrier met by others)
                # still do a single pull to align with "global after aggregation"
                new_round, new_w, rtt_sec, bytes_in2 = wait_new_round(stub, server_round, t0_perf)
                if new_round == server_round:
                    # if it didn't advance quickly, just keep rtt as measured and continue
                    bytes_in = bytes_in2
                else:
                    bytes_in = bytes_in2
                    if new_w is not None:
                        model.set_weights(new_w)
                completed += 1
            else:
                # REJECTED/ERROR -> do not count; just loop
                pass

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(COMM_LOG, "a", newline="") as f:
                csv.writer(f).writerow([
                    completed - 1 if completed > 0 else 0,
                    server_round,
                    bytes_in,
                    bytes_out,
                    f"{train_sec:.6f}",
                    f"{send_sec:.6f}",
                    ack_status,
                    ack_cur,
                    f"{rtt_sec:.6f}",
                    ts
                ])

            print(f"[PLAIN-PFTLSTYLE] completed={completed}/{NUM_ROUNDS} "
                  f"pulled_round={server_round} in={bytes_in} out={bytes_out} "
                  f"train={train_sec:.2f}s send={send_sec:.3f}s rtt={rtt_sec:.3f}s ack={ack_status}")

        loss, acc = model.evaluate(x_test, y_test, verbose=0)

    finally:
        stop.set()
        mon.join(timeout=5)
        channel.close()

    res = summarize_resources(RES_LOG)
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "client_id": CLIENT_ID,
        "mode": "plain_pftlstyle",
        "target": TARGET,
        "num_rounds": str(NUM_ROUNDS),
        "local_epochs": str(LOCAL_EPOCHS),
        "test_acc": f"{acc:.6f}",
        "test_loss": f"{loss:.6f}",
        **res
    }

    with open(FINAL_LOG, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)

    print(f"[FINAL] test_acc={row['test_acc']} peak_rss_mb={row['peak_rss_mb']} avg_cpu_norm%={row['avg_cpu_norm_percent']}")


if __name__ == "__main__":
    main()