#!/usr/bin/env python3
"""Run the independent per-epoch sealed ObjectRecovery service."""

import argparse
import json
from pathlib import Path

import torch

from Merak.core.checkpoint.object_replica_service import ObjectReplicaServiceStore
from Merak.core.checkpoint.replica_service import ReplicaTCPServer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29591)
    parser.add_argument("--max-store-mib", type=int, default=4096)
    parser.add_argument("--retained-epochs", type=int, default=2)
    parser.add_argument("--torch-num-threads", type=int, default=0)
    parser.add_argument("--ready-file", type=Path)
    args = parser.parse_args()
    if args.torch_num_threads < 0:
        raise ValueError("torch-num-threads must be non-negative")
    if args.torch_num_threads:
        torch.set_num_threads(args.torch_num_threads)
    store = ObjectReplicaServiceStore(
        max_bytes=args.max_store_mib * 1024 * 1024,
        retained_epochs_per_source=args.retained_epochs,
        sealed_incremental=True,
    )
    with ReplicaTCPServer((args.bind, args.port), store) as server:
        endpoint = f"{server.server_address[0]}:{server.server_address[1]}"
        if args.ready_file is not None:
            args.ready_file.parent.mkdir(parents=True, exist_ok=True)
            args.ready_file.write_text(endpoint + "\n")
        print(json.dumps({"status": "ready", "endpoint": endpoint}), flush=True)
        try:
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
