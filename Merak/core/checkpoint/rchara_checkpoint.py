import json
import os
import socket
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.distributed as dist

from Merak import get_logger

from .. import mpu
from .checkpoint import CheckpointSaver, CheckpointLoader
from .utils import unwrap_model
from .grouping import (
    create_checkpoint_process_groups,
    grouping_config_from_env,
    plan_checkpoint_groups,
)
from .redundancy_checker import RedundancyChecker
from .recovery import RemoteCheckpointSession, ServiceRemoteCheckpointSession
from .catalog import iter_checkpoint_objects, serialize_checkpoint
from .async_checkpoint import AsyncCommitResult, AsyncServiceRemoteCheckpointSession



from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
import pickle
import datetime

# Build communication groups for remote checkpointing
def build_comm_groups_for_remote_checkpointing(k=2):
    """Build groups from explicit experiment topology settings.

    ``CKPT_RANKS_PER_NODE`` (or torchrun's ``LOCAL_WORLD_SIZE``) describes
    physical node layout. ``CKPT_GROUPING_SCOPE=rank`` is only for
    single-host transport experiments and does not protect against host loss.
    """
    logger = get_logger("simple")
    config = grouping_config_from_env(default_group_size=k)
    plan = plan_checkpoint_groups(dist.get_world_size(), config)
    try:
        group, group_ranks = create_checkpoint_process_groups(plan, backend="gloo")
    except Exception as error:
        logger.error(f"Failed to create remote checkpoint groups: {error}")
        raise
    logger.info(
        f"Remote checkpoint grouping: rank={dist.get_rank()} scope={config.scope} "
        f"policy={config.policy} rank_pairing={config.rank_pairing_policy} "
        f"ranks_per_node={config.ranks_per_node} group={group_ranks}"
    )
    return group, list(group_ranks)


# Get dictionaries for remote checkpointing.
def get_dicts_for_remote_checkpointing(transmission_time_dict, local_checkpoint, comm_group_for_rcheck, comm_group_global_ranks_for_rcheck):
    '''
        Get partition dictionaries for remote checkpointing.
    '''
    logger = get_logger("simple")

    # split transferred global_ckpt into a list according to profile result
    # vars for remote transmission
    group_for_remote_ckpt = comm_group_for_rcheck
    rank_in_group = torch.distributed.get_rank(group=group_for_remote_ckpt)
    group_size = torch.distributed.get_world_size(group=group_for_remote_ckpt)

    src_of_remote_ckpt = comm_group_global_ranks_for_rcheck[(rank_in_group - 1) % group_size]
    dst_of_remote_ckpt = comm_group_global_ranks_for_rcheck[(rank_in_group + 1) % group_size]


    # record the meta data for each parameter tensor in terms of (name, index, size), done only once thanks to the consistency
    param_meta_data = []
    global_ckpt_size = 0

    # since dict is ordered if python version >= 3.7, no need to record name and shape for weights,
    # while extra states are all empty bytes or None.
    # we can get necessary information from healthy ranks for recovery.
    for k, v in local_checkpoint.items():
        if isinstance(v, torch.Tensor):
            param_meta_data.append((k, v.numel()))
            global_ckpt_size += v.numel()
        else:
            logger.error(f"In the local checkpoint, the value of key {k} is not a tensor.")
            raise 

    # pre-alloc cpu buffer for global_ckpt and remote_ckpt, both on CPU
    remote_ckpt_dict = {}
    remote_ckpt_dict.update({f'rank{src_of_remote_ckpt}': torch.zeros(global_ckpt_size, dtype=torch.float16, device='cpu')})
    cpu_global_ckpt = torch.zeros(global_ckpt_size, dtype=torch.float16, device='cpu', pin_memory=True)
    logger.info(f"checkpoint size: {round(global_ckpt_size*2/1024**2, 3)} MB")

                
    # update splitting information to be used until next profiling results are available
    # -> partitions format: {stage_name : length}
    # -> index_dict format: {stage_name : (start_index, end_index)}
    # -> snapshot_chunks_dict format: {stage_name : [(name, start_offset, end_offset)]}
    def scale_dict_to_target_sum(int_dict, target_sum, alpha=0):
        """ adjust the sum of the input int dict to a given value, take threshold into account. """

        if alpha!=0:
            for k, v in int_dict.items():
                int_dict[k] = max(0, v - alpha)

        current_sum = sum(int_dict.values())
        if current_sum == target_sum:
            return int_dict

        scale_factor = target_sum / current_sum
        scaled_dict = {k:round(v * scale_factor) for k,v in int_dict.items()}
        scaled_sum = sum(scaled_dict.values())
        if scaled_sum != target_sum:
            for k, v in scaled_dict.items():
                if v + target_sum - scaled_sum > 0:
                    scaled_dict[k] += target_sum - scaled_sum
                    break

        return scaled_dict
    
    partitions = scale_dict_to_target_sum(transmission_time_dict.copy(), global_ckpt_size)


    def revert_to_chunk_info_dict(meta_data, length_dict):
        """
            Args:
                meta_data: a list of [(name, numel)] representing the meta_data of each paramter tensor.
                length_dict: a dict of {stage: length} representing the length of checkpoint chunk for each stage.
            Returns:
                index_dict: a dict of {stage_name : (start_index, end_index)} representing the index ranges of the merged tensor for each stage. 
                chunks_dict: a dict of {stage: [(name, start_offset, end_offset)]} representing the set consist of index ranges of tensors constituting a chunk for each stage.

        """
        sum = 0
        id = 0          # id of current tensor
        offset = 0      # offset of current tensor
        index_dict = {}
        chunks_dict = {}
        for stage, length in length_dict.items():
            index_dict[stage] = (sum, sum + length)
            sum += length
            chunks_dict[stage] = []
            curr_length = 0
            while curr_length < length:
                curr_tensor_meta = meta_data[id]    # (name, id, numel)
                if curr_length + curr_tensor_meta[1] - offset <= length:
                    chunks_dict[stage].append((curr_tensor_meta[0], offset, curr_tensor_meta[1]))
                    curr_length += curr_tensor_meta[1] - offset
                    id += 1
                    offset = 0
                else:
                    chunks_dict[stage].append((curr_tensor_meta[0], offset, offset + length - curr_length))
                    offset += length - curr_length
                    break
        assert id == len(meta_data)
        return index_dict, chunks_dict

    index_dict = {}
    snapshot_chunks_dict = {}
    d1,d2 = revert_to_chunk_info_dict(param_meta_data, partitions)
    index_dict.update(d1)
    snapshot_chunks_dict.update(d2)

    # send/recv index_dict to/from remote node
    # -> remote_index_dict format: {stage_name : (start_index, end_index)}
    # p2p-comm for objects is not supported for torch<=2.2.0, thus replaced with broadcast.
    # torch.distributed.all_gather_object(object_list, obj, group=None)
    remote_index_dict = {}
    objects = [index_dict] if rank_in_group == 0 else [None]
    dist.broadcast_object_list(objects, src=dist.get_global_rank(group=group_for_remote_ckpt, group_rank=0), group=group_for_remote_ckpt)
    if rank_in_group == 1:
        remote_index_dict.update(objects[0])

    objects = [index_dict] if rank_in_group == 1 else [None]
    dist.broadcast_object_list(objects, \
        src=dist.get_global_rank(group=group_for_remote_ckpt, group_rank=1), group=group_for_remote_ckpt)
    if rank_in_group == 0:
        remote_index_dict.update(objects[0])

    logger.info(f'Dicts returned for remote checkpointing, param_meta_data:{param_meta_data}, remote_ckpt_dict:{remote_ckpt_dict}, index_dict:{index_dict}, snapshot_chunks_dict:{snapshot_chunks_dict}, remote_index_dict:{remote_index_dict}')

    return param_meta_data, remote_ckpt_dict, cpu_global_ckpt, index_dict, snapshot_chunks_dict, remote_index_dict

def logging_dict(state_dict, pre_fix=""):
    """
    Recursively logs information about a dictionary's contents, including:
    - Gradients (shape and dtype if present)
    - Shapes of tensors/arrays
    - Types of other values

    Handles nested dictionaries, lists, and tuples by recursively processing them.
    Designed for logging PyTorch model states but works with any dictionary-like object.

    Args:
        state_dict (dict): The dictionary to log information about
        pre_fix (str): Optional prefix string for log messages (used internally for recursion)

    Returns:
        None: Only produces log output
    """
    logger = get_logger("simple")

    for key, value in state_dict.items():
        current_prefix = f"{key} {pre_fix}".strip()

        # Handle objects with gradients (PyTorch tensors)
        if hasattr(value, "grad"):
            if value.grad is not None:
                logger.info(
                    f"{current_prefix} has grad {value.grad.shape} with {value.grad.dtype}"
                )
            else:
                logger.info(f"{current_prefix} has grad none but shape {value.shape}")

        # Handle nested dictionaries
        elif isinstance(value, dict):
            logging_dict(value, f"of {current_prefix}")

        # Handle sequences (lists/tuples)
        elif isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                item_prefix = f"{idx}th of {current_prefix}"

                if hasattr(item, "grad"):
                    if item.grad is not None:
                        logger.info(
                            f"{item_prefix} has grad {item.grad.shape} with {item.grad.dtype}"
                        )
                    else:
                        logger.info(
                            f"{item_prefix} has grad none but shape {item.shape}"
                        )
                elif isinstance(item, dict):
                    logging_dict(item, f"of {item_prefix}")
                elif hasattr(item, "shape"):
                    logger.info(f"{item_prefix} no grad but shape {item.shape}")
                else:
                    logger.info(f"{item_prefix} no grad no shape {type(item)}")

        # Handle other objects with shape attribute
        elif hasattr(value, "shape"):
            logger.info(f"{current_prefix} no grad but shape {value.shape}")

        # Fallback for other types
        else:
            logger.info(f"{current_prefix} no grad no shape {type(value)}")


def extract_grad_and_const(state_dict, grad_const_dict, exclude_list=None):
    """
    Recursively extracts gradients and constant values from a PyTorch state dictionary.

    Processes nested dictionaries, lists, and tuples to extract:
    - Gradient tensors (when available)
    - Constant tensors (when no gradient exists)
    - Preserves excluded keys without modification

    Args:
        state_dict (dict): The input dictionary containing model parameters/state
        grad_const_dict (dict): Output dictionary to store extracted gradients/constants
        exclude_list (list, optional): List of keys to preserve as-is without processing

    Returns:
        None: Modifies grad_const_dict in-place with extracted values

    Note:
        - All extracted tensors are cloned and detached for safety
        - Handles nested structures recursively
        - Preserves original values for excluded keys
    """
    if exclude_list is None:
        exclude_list = []

    for key, value in state_dict.items():
        if key != 'models':
            grad_const_dict[key] = value
            continue

        # Handle gradient tensors
        if hasattr(value, "grad") and value.grad is not None:
            grad_const_dict[key] = value.grad.data.clone().detach()

        # Handle nested dictionaries
        elif isinstance(value, dict):
            grad_const_dict[key] = {}
            extract_grad_and_const(value, grad_const_dict[key], exclude_list)

        # Handle sequences (lists/tuples)
        elif isinstance(value, (list, tuple)):
            grad_const_dict[key] = []
            for item in value:
                if hasattr(item, "grad") and item.grad is not None:
                    grad_const_dict[key].append(item.grad.data.clone().detach())
                elif isinstance(item, dict):
                    new_dict = {}
                    extract_grad_and_const(item, new_dict, exclude_list)
                    grad_const_dict[key].append(new_dict)
                elif not isinstance(item, torch.Tensor):
                    grad_const_dict[key].append(item.clone().detach())
                else:
                    grad_const_dict[key].append(item)

        # Handle regular tensors
        elif not isinstance(value, torch.Tensor):
            grad_const_dict[key] = value.clone().detach()

        # Preserve other types
        else:
            grad_const_dict[key] = value


def clean_grad(state_dict):
    """
    Recursively cleans gradients from a PyTorch state dictionary in-place.

    Processes nested dictionaries, lists, and tuples to:
    - Remove gradient references from Parameters and Tensors
    - Ensure all tensors are cloned and detached
    - Preserve the original structure of the state_dict

    Args:
        state_dict (dict): The input dictionary containing model parameters/state
                          (modified in-place)

    Returns:
        None: Modifies the input state_dict directly

    Note:
        - Handles both Parameters and regular Tensors
        - Processes nested structures recursively
        - All operations are performed in-place
        - Ensures memory safety by cloning and detaching tensors
    """
    for key, value in state_dict.items():
        # Handle Parameters and Tensors with gradients
        if isinstance(value, torch.nn.parameter.Parameter):
            state_dict[key] = value.clone().detach()

        # Handle nested dictionaries
        elif isinstance(value, dict):
            clean_grad(value)

        # Handle sequences (lists/tuples)
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                if isinstance(item, torch.nn.parameter.Parameter):
                    value[i] = item.clone().detach()
                elif isinstance(item, dict):
                    clean_grad(item)


class RcharaCheckpointSaver(CheckpointSaver):
    """
    Redundancy-aware Checkpoint Saver with differential saving capability

    Extends base CheckpointSaver to provide advanced checkpointing features including
    differential saving, redundancy analysis, and Zero-Redundancy Optimizer support.
    """

    def __init__(self):
        super().__init__()
        self.logger = get_logger("simple")
        self.ablation_study = False # setting to ablation_study
        self.checkpoint_version = 1
        self.redundancy_checker = None

        # Variables for remote checkpointing
        self.rcheckpoint_comm_group = None
        self.rcheckpoint_group_global_ranks = None
        self.remote_checkpoint_session = None
        self.async_remote_checkpoint_session = None
        self.remote_checkpoint_epochs = set()
        self.remote_checkpoint_pending_epochs = set()
        self.reduced_capture_epochs = set()


    def _remote_checkpoint_enabled(self):
        return os.environ.get("CKPT_REMOTE_ENABLE", "0") == "1"

    def _remote_async_enabled(self):
        return (
            self._remote_checkpoint_enabled()
            and os.environ.get("CKPT_REMOTE_ASYNC", "0") == "1"
        )

    def prepare_remote_checkpointing(self, args):
        """Create service clients before the measured training loop."""
        self.args = args
        if not self._remote_async_enabled():
            return
        if os.environ.get("CKPT_REMOTE_BACKEND", "gloo") != "service":
            raise ValueError("CKPT_REMOTE_ASYNC currently requires the service backend")
        self._ensure_remote_checkpoint_session()
        max_inflight = int(os.environ.get("CKPT_REMOTE_MAX_INFLIGHT", "2"))
        self.async_remote_checkpoint_session = AsyncServiceRemoteCheckpointSession(
            self.remote_checkpoint_session,
            max_inflight=max_inflight,
        )

    def _ensure_remote_checkpoint_session(self):
        if self.remote_checkpoint_session is not None:
            return
        manifest_root = Path(
            os.environ.get(
                "CKPT_REMOTE_MANIFEST_DIR",
                str(Path(self.args.output_dir) / "remote_checkpoint"),
            )
        )
        max_store_mib = int(os.environ.get("CKPT_REMOTE_STORE_MIB", "4096"))
        backend = os.environ.get("CKPT_REMOTE_BACKEND", "gloo")
        if backend == "gloo":
            if self.rcheckpoint_comm_group is None:
                (
                    self.rcheckpoint_comm_group,
                    self.rcheckpoint_group_global_ranks,
                ) = build_comm_groups_for_remote_checkpointing()
            self.remote_checkpoint_session = RemoteCheckpointSession(
                group=self.rcheckpoint_comm_group,
                group_ranks=tuple(self.rcheckpoint_group_global_ranks),
                manifest_root=manifest_root,
                max_store_bytes=max_store_mib * 1024 * 1024,
            )
            return
        if backend != "service":
            raise ValueError(f"unsupported CKPT_REMOTE_BACKEND: {backend}")

        config = grouping_config_from_env()
        plan = plan_checkpoint_groups(dist.get_world_size(), config)
        group_ranks = plan.group_for_rank(dist.get_rank())
        if len(group_ranks) != 2:
            raise ValueError(
                f"service backend requires two-rank groups, got {group_ranks}"
            )
        peer_rank = (
            group_ranks[1] if group_ranks[0] == dist.get_rank() else group_ranks[0]
        )
        endpoint = os.environ.get("CKPT_REPLICA_ENDPOINT")
        if endpoint is None:
            encoded_endpoints = os.environ.get("CKPT_REPLICA_ENDPOINTS")
            if encoded_endpoints is None:
                raise ValueError(
                    "service backend requires CKPT_REPLICA_ENDPOINT or "
                    "CKPT_REPLICA_ENDPOINTS"
                )
            endpoints = json.loads(encoded_endpoints)
            if not isinstance(endpoints, list):
                raise ValueError(
                    "CKPT_REPLICA_ENDPOINTS must be a JSON list per host"
                )
            peer_host_index = peer_rank // config.ranks_per_node
            try:
                endpoint = str(endpoints[peer_host_index])
            except IndexError as error:
                raise ValueError(
                    f"no replica endpoint for peer host index {peer_host_index}"
                ) from error
        self.rcheckpoint_group_global_ranks = list(group_ranks)
        self.remote_checkpoint_session = ServiceRemoteCheckpointSession(
            rank=dist.get_rank(),
            group_ranks=group_ranks,
            service_endpoint=endpoint,
            manifest_root=manifest_root,
            max_local_store_bytes=max_store_mib * 1024 * 1024,
        )
        self.logger.info(
            f"Remote checkpoint service: rank={dist.get_rank()} group={group_ranks} "
            f"endpoint={endpoint}"
        )

    def _save_remote_checkpoint(
        self, state_dict, zero_state_dict=None, peft_state_dict=None
    ):
        if (
            self.iteration in self.remote_checkpoint_epochs
            or self.iteration in self.remote_checkpoint_pending_epochs
        ):
            self.logger.info(
                f"[Rchara] Remote checkpoint epoch {self.iteration} already submitted; skipping"
            )
            return
        self._ensure_remote_checkpoint_session()
        remote_state = dict(state_dict)
        if zero_state_dict is not None:
            remote_state["zero_optimizer"] = zero_state_dict
        if peft_state_dict is not None:
            remote_state["peft_state_dict"] = peft_state_dict
        if self._remote_async_enabled():
            if self.async_remote_checkpoint_session is None:
                self.prepare_remote_checkpointing(self.args)
            self._collect_async_remote_results()
            capture_ms = self.async_remote_checkpoint_session.enqueue(
                self.iteration, remote_state
            )
            self.remote_checkpoint_pending_epochs.add(self.iteration)
            self.logger.info(
                f"[Rchara] Remote checkpoint captured: epoch={self.iteration} "
                f"capture_ms={capture_ms:.3f}"
            )
            return
        result = self.remote_checkpoint_session.save(self.iteration, remote_state)
        self._record_remote_result(result)

    def _record_remote_result(
        self, result, *, capture_ms=None, queue_ms=None, commit_lag=0
    ):
        metrics_dir = self.remote_checkpoint_session.repository.root / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        epoch = result.manifest.epoch
        metrics = {
            "epoch": epoch,
            "rank": dist.get_rank(),
            "group_ranks": list(result.manifest.group_ranks),
            "manifest_id": result.manifest.manifest_id,
            "object_count": len(result.manifest.objects),
            "logical_bytes": result.manifest.logical_bytes,
            "payload_bytes": result.manifest.payload_bytes,
            "wire_bytes": result.transport.wire_bytes,
            "serialize_ms": result.serialize_ms,
            "transport_ms": result.transport.duration_ms,
            "commit_ms": result.commit_ms,
            "store_bytes": result.store_bytes,
            "capture_ms": capture_ms,
            "queue_ms": queue_ms,
            "commit_lag": commit_lag,
            "asynchronous": self._remote_async_enabled(),
        }
        with (metrics_dir / f"rank_{dist.get_rank():05d}.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics, sort_keys=True) + "\n")
        self.remote_checkpoint_epochs.add(epoch)
        self.logger.info(
            f"[Rchara] Remote checkpoint committed: epoch={epoch} "
            f"group={result.manifest.group_ranks} payload={result.manifest.payload_bytes} "
            f"commit_ms={result.commit_ms:.3f}"
        )

    def _collect_async_remote_results(self):
        if self.async_remote_checkpoint_session is None:
            return
        for completed in self.async_remote_checkpoint_session.poll():
            self._record_async_remote_result(completed)

    def _record_async_remote_result(self, completed: AsyncCommitResult):
        self.remote_checkpoint_pending_epochs.discard(completed.epoch)
        self._record_remote_result(
            completed.result,
            capture_ms=completed.capture_ms,
            queue_ms=completed.queue_ms,
            commit_lag=max(0, int(self.iteration) - completed.epoch),
        )

    def finalize_remote_checkpointing(self):
        if self.async_remote_checkpoint_session is None:
            return
        for completed in self.async_remote_checkpoint_session.close():
            self._record_async_remote_result(completed)
        if self.remote_checkpoint_pending_epochs:
            raise RuntimeError(
                "remote checkpoint shutdown left uncommitted epochs: "
                f"{sorted(self.remote_checkpoint_pending_epochs)}"
            )
        self._save_tracker_file()

    @staticmethod
    def _atomic_capture(path: Path, payload: bytes):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, path)

    def _export_reduced_capture(
        self, state_dict, zero_state_dict=None, peft_state_dict=None
    ):
        capture_root = os.environ.get("CKPT_REDUCED_CAPTURE_DIR")
        if capture_root is None:
            return
        rank = dist.get_rank()
        state = dict(state_dict)
        if zero_state_dict is not None:
            state["zero_optimizer"] = zero_state_dict
        if peft_state_dict is not None:
            state["peft_state_dict"] = peft_state_dict
        metadata_only = os.environ.get(
            "CKPT_REDUCED_CAPTURE_METADATA_ONLY", "0"
        ) == "1"
        serialized = None if metadata_only else serialize_checkpoint(state)
        objects = (
            tuple(iter_checkpoint_objects(state))
            if metadata_only
            else serialized.objects
        )
        evidence = self.redundancy_checker.tensor_redundancy_evidence(rank)
        root = Path(capture_root) / f"epoch_{self.iteration:08d}"
        if metadata_only:
            self._atomic_capture(
                root / "source" / f"rank_{rank:05d}.objects.json",
                (
                    json.dumps(
                        [item.to_dict() for item in objects],
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                ).encode(),
            )
        else:
            self._atomic_capture(
                root / "source" / f"rank_{rank:05d}.bin", serialized.payload
            )
        evidence_roles = sorted({item.role for item in evidence})
        state_roles = sorted({item.role for item in objects})
        metadata = {
            "schema_version": 1,
            "capture_format": "metadata_only" if metadata_only else "payload",
            "rank": rank,
            "world_size": dist.get_world_size(),
            "local_world_size": int(os.environ.get("LOCAL_WORLD_SIZE", "1")),
            "host": socket.gethostname(),
            "epoch": self.iteration,
            "payload_bytes": 0 if metadata_only else len(serialized.payload),
            "payload_checksum": None if metadata_only else serialized.payload_checksum,
            "logical_bytes": sum(item.logical_bytes for item in objects),
            "object_count": len(objects),
            "detector_evidence_count": len(evidence),
            "detector_evidence_roles": evidence_roles,
            "conservative_unique_roles": sorted(set(state_roles) - set(evidence_roles)),
            "detector_comparison_scope": self.redundancy_checker.comparison_scope,
            "zero_stage": self.args.zero_stage,
        }
        self._atomic_capture(
            root / "source" / f"rank_{rank:05d}.json",
            (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode(),
        )
        self._atomic_capture(
            root / "detector" / f"rank_{rank:05d}.json",
            (
                json.dumps(
                    [item.to_dict() for item in evidence], indent=2, sort_keys=True
                )
                + "\n"
            ).encode(),
        )
        self.logger.info(
            f"[Rchara] Exported detector capture: epoch={self.iteration} "
            f"rank={rank} format={'metadata_only' if metadata_only else 'payload'} "
            f"payload={0 if metadata_only else len(serialized.payload)}"
        )

    def restore_remote_checkpoint(self, epoch):
        self._ensure_remote_checkpoint_session()
        return self.remote_checkpoint_session.restore_from_peer(epoch)

    def init_redundancy_checker(self, zero_state_dict):
        self.redundancy_checker = RedundancyChecker(
            self.model,
            self.optimizer,
            self.args.zero_stage,
            zero_state_dict,
        )

    def _save_state_dict(
            self,
            state_dict: Dict,
            checkpoint_name: str,
            zero_ckpt_name: Optional[str] = None,
            peft_state_dict: Optional[Dict] = None,
            zero_state_dict: Optional[Dict] = None,
            exclude_list: Optional[List] = None,
            **kwargs,
        ):
        """
        Save the state dictionary to the specified path.

        Args:
            state_dict: State dictionary to save.
            checkpoint_name: Path to save the main checkpoint.
            zero_ckpt_name: Path for saving Zero-Redundancy Optimizer states.
            peft_state_dict: LoRA state dictionary to save if applicable.
            zero_state_dict: Zero Optimizer state dictionary.
            **kwargs: Additional keyword arguments.
        """
        check_steps = set(self.args.check_steps or [])
        remote_state_dict = state_dict
        capture_only = os.environ.get("CKPT_CAPTURE_ONLY", "0") == "1"
        if capture_only and not os.environ.get("CKPT_REDUCED_CAPTURE_DIR"):
            raise ValueError("CKPT_CAPTURE_ONLY requires CKPT_REDUCED_CAPTURE_DIR")
        if capture_only and self.iteration not in check_steps:
            raise ValueError("CKPT_CAPTURE_ONLY requires this iteration in check_steps")
        if check_steps and self.iteration > max(check_steps):
            self.args.use_differential = True

        if self.args.use_differential:
            self.checkpoint_version = self.iteration

        if self.redundancy_checker is None:
            self.init_redundancy_checker(zero_state_dict)

        if self.iteration in check_steps:
            self.redundancy_checker.add_zero_state(zero_state_dict)

        if exclude_list is None:
            exclude_list = ["rng_state"]

        if (
            mpu.get_data_parallel_rank() == 0
            and not capture_only
            and not self._remote_async_enabled()
        ):
            if check_steps and self.iteration > max(check_steps) and \
                self.checkpoint_version % self.args.differential_interval != 0:
                self.redundancy_checker.get_dict_for_gradient_checkpointing()
                state_dict = self.redundancy_checker.parameter_dict_for_gradient_checkpointing
                # Debugging
                self.logger.debug(f"Keys for parameter_dict_for_gradient_checkpointing:{state_dict.keys()}")
            # else:
            #     if self.iteration in check_steps:
            #         tmp = state_dict['optimizer'].pop('fp16_groups')

            if self.args.use_differential:
                if self.iteration % self.args.differential_interval == 0:
                    # checkpoint_name = checkpoint_name.replace(
                    #     "iter_", "full_iter_"
                    # )
                    self.logger.info(f"[Rchara] Saving full {checkpoint_name}...")
                    clean_grad(state_dict)
                    # log_dist(f'state_dict:{state_dict}', ranks=[0])
                    self.logger.debug(f"checkpoint_name:{checkpoint_name}, keys of state dict after clean_grad:{state_dict.keys()}")
                    torch.save(
                        state_dict,
                        checkpoint_name,
                        _use_new_zipfile_serialization=False,
                    )
                    # logger.info(f"{path}...")
                    # logging_dict(state_dict=state_dict)
                else:
                    grad_const_dict = {}
                    extract_grad_and_const(state_dict, grad_const_dict, exclude_list)
                    self.logger.info(f"[Rchara] Saving grad {checkpoint_name}...")
                    self.logger.debug(f"checkpoint_name:{checkpoint_name}, keys of state dict after extract_grad_and_const:{state_dict.keys()}")
                    torch.save(
                        grad_const_dict,
                        checkpoint_name,
                        _use_new_zipfile_serialization=False,
                    )
                    # logger.info(f"{path}...")
                    # logging_dict(state_dict=grad_const_dict)
            else:
                self.logger.debug(f"checkpoint_name:{checkpoint_name}, keys of state dict when use_differential is false:{state_dict.keys()}")
                torch.save(
                    state_dict, checkpoint_name, _use_new_zipfile_serialization=False
                )

        # Save Zero Optimizer states if required
        if (
            self.args.zero_stage == 1
            and not capture_only
            and not self._remote_async_enabled()
        ):
            dist.barrier()
            torch.save(zero_state_dict, zero_ckpt_name)
            self.logger.info(
                f"zero checkpoint saved {zero_ckpt_name}", ranks=[dist.get_rank()]
            )

        if not self._remote_async_enabled():
            dist.barrier()  # Ensure all processes are synchronized

        # redundancy check
        capture_already_exported = (
            os.environ.get("CKPT_REDUCED_CAPTURE_DIR") is not None
            and self.iteration in self.reduced_capture_epochs
        )
        if self.iteration in check_steps and not capture_already_exported:
            self.redundancy_checker.add_model_state(state_dict['model'])
            self.redundancy_checker.add_optimizer_state(state_dict['optimizer'] 
                                                  if zero_state_dict is None else None)
            self.redundancy_check()
            self._export_reduced_capture(
                remote_state_dict,
                zero_state_dict=zero_state_dict,
                peft_state_dict=peft_state_dict,
            )
            if os.environ.get("CKPT_REDUCED_CAPTURE_DIR") is not None:
                self.reduced_capture_epochs.add(self.iteration)
        elif capture_already_exported:
            self.logger.info(
                f"[Rchara] Detector capture epoch {self.iteration} already exported; "
                "skipping"
            )

        if self._remote_checkpoint_enabled():
            self._save_remote_checkpoint(
                remote_state_dict,
                zero_state_dict=zero_state_dict,
                peft_state_dict=peft_state_dict,
            )

        # Publish progress only after every required remote replica is committed.
        if not self._remote_async_enabled():
            self._save_tracker_file()
        self.logger.info(f"[Rchara] Saved at iteration {self.iteration} to {self.save_path}.", ranks=[0])

        # update checkpoint version
        if self.args.use_differential:
            self.checkpoint_version = self.iteration

    def redundancy_check(self):
        """
        Perform redundancy analysis on model parameters at specified check steps.

        When ablation_study is disabled, runs the standard redundancy checking pipeline.
        When ablation_study is enabled, benchmarks different redundancy checking 
        configurations and algorithms for performance comparison.
        """
        if not self.ablation_study:
            phases = (
                ("name_mapping", self.redundancy_checker.build_name_value_mapping),
                ("state_hashing", self.redundancy_checker.get_hash_values),
                ("hash_packing", self.redundancy_checker.pack_to_tensor),
                ("cross_rank_compare", self.redundancy_checker.do_redundancy_checking),
                (
                    "name_resolution",
                    self.redundancy_checker.get_names_of_parameters_requiring_gradient_checkpointing,
                ),
            )
            for phase, operation in phases:
                started = time.perf_counter()
                self.logger.info(f"[Rchara] Redundancy phase started: {phase}")
                operation()
                self.logger.info(
                    f"[Rchara] Redundancy phase finished: {phase} "
                    f"duration_s={time.perf_counter() - started:.3f}"
                )
            self.logger.info("[Rchara] Redundancy check finished." )

            # Build communication group for remote checkpointing.
            if self.rcheckpoint_comm_group is None:
                (
                    self.rcheckpoint_comm_group,
                    self.rcheckpoint_group_global_ranks,
                ) = build_comm_groups_for_remote_checkpointing()

        else:
            self.redundancy_checker.build_name_value_mapping()
            self.redundancy_checker.get_hash_values()
            self.redundancy_checker.pack_to_tensor()
            start = time.time()
            self.redundancy_checker.do_redundancy_checking()
            end = time.time()
            self.logger.info(f'Redundancy checking time is {end - start} s', ranks=[0])

            self.redundancy_checker.no_redundant_cal = False
            start = time.time()
            self.redundancy_checker.do_redundancy_checking()
            end = time.time()
            self.logger.info(f'Redundancy checking without no_redundant_cal time is {end - start} s', ranks=[0])
            self.redundancy_checker.no_redundant_cal = True

            self.redundancy_checker.comm_groups_for_redundancy_checking = False
            start = time.time()
            self.redundancy_checker.do_redundancy_checking()
            end = time.time()
            self.logger.info(
                f'Redundancy checking without comm_groups_for_redundancy_checking cal time is {end - start} s',
                ranks=[0]
            )
            self.redundancy_checker.comm_groups_for_redundancy_checking = True

            self.redundancy_checker.redundancy_cache = False
            start = time.time()
            self.redundancy_checker.do_redundancy_checking()
            end = time.time()
            self.logger.info(f'Redundancy checking without redundancy_cache time is {end - start} s', ranks=[0])
            self.redundancy_checker.redundancy_cache = True

            self.redundancy_checker.get_baseline_packked_tensor()
            start = time.time()
            self.redundancy_checker.baseline_compare()
            end = time.time()
            self.logger.info(f'Baseline redundancy checking time is {end - start} s', ranks=[0])     

class RcharaCheckpointLoader(CheckpointLoader):
    """Apply a validated full-state replica returned by a remote session."""

    def load_remote_state(
        self,
        state_dict,
        model,
        optimizer,
        lr_scheduler,
        args,
        verbose=False,
        strict=True,
    ):
        if "model" not in state_dict:
            raise ValueError("remote checkpoint does not contain a full model state")
        self.model = unwrap_model(model)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.args = args
        self._load_state_dict(state_dict["model"], strict, verbose)

        if not args.finetune and not args.no_load_optim:
            if args.zero_stage == 1:
                self._load_zero_optimizer(state_dict.get("zero_optimizer"))
            else:
                self._load_optimizer_from_state_dict(state_dict)
        if not args.finetune:
            self._load_learning_rate_scheduler(state_dict)
            if not args.no_load_rng:
                self._load_rng_states(state_dict)
        self.version_manager.set_checkpoint_version(
            state_dict.get("checkpoint_version", 0)
        )
        return int(state_dict.get("iteration", 0))
