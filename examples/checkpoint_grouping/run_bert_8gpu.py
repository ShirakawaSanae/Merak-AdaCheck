#!/usr/bin/env python3
"""Small 8-GPU Merak/AdaCheck save-path smoke test."""

import os
import signal
import time

import torch

from transformers import BertConfig, BertForMaskedLM, HfArgumentParser, set_seed

import Merak
from Merak import MerakArguments, MerakTrainer
from Merak.utils.datasets import DynamicGenDataset


def main() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    Merak.init(pp=1, tp=1, dp=world_size)
    parser = HfArgumentParser(MerakArguments)
    training_args, = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    config = BertConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=64,
        use_cache=False,
    )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    model = BertForMaskedLM(config).to(torch.device("cuda", local_rank))
    dataset = DynamicGenDataset(config, mode="text_only", dataset_size=4096, seq_length=64)
    trainer = MerakTrainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()

    failed_rank = os.environ.get("CKPT_EXPERIMENT_KILL_RANK_AFTER_TRAIN")
    if failed_rank is not None:
        torch.distributed.barrier()
        if torch.distributed.get_rank() == int(failed_rank):
            os.kill(os.getpid(), signal.SIGKILL)
        time.sleep(30)
        raise RuntimeError("torchrun did not stop after the injected worker failure")


if __name__ == "__main__":
    main()
