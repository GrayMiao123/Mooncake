import argparse
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.testing as testing


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "mooncake-wheel"))


@dataclass(frozen=True)
class PathologicalCase:
    name: str
    num_tokens: int
    hidden: int
    num_experts: int
    topk: int
    routing: str


CASES = (
    PathologicalCase("control_uniform", 64, 7168, 16, 2, "uniform"),
    PathologicalCase("hotspot_two_experts", 64, 7168, 16, 2, "hotspot"),
    PathologicalCase("single_rank_owner", 64, 7168, 16, 2, "single_owner"),
    PathologicalCase("zero_tokens_on_rank1", 64, 7168, 16, 2, "zero_tokens"),
)
BACKENDS = ("alltoall", "deepep", "mooncake")


def print_table(headers, rows):
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))

    def format_row(row):
        return "  ".join(
            str(cell).ljust(widths[index]) for index, cell in enumerate(row)
        )

    print(format_row(headers), flush=True)
    print(format_row(["-" * width for width in widths]), flush=True)
    for row in rows:
        print(format_row(row), flush=True)


def expert_factor(expert_id):
    return expert_id * 0.1 + 1.0


def wait_event_if_present(event):
    if event is not None and getattr(event, "event", None) is not None:
        event.current_stream_wait()


def make_topk_idx(case, rank):
    local_tokens = 0 if case.routing == "zero_tokens" and rank == 1 else case.num_tokens
    if local_tokens == 0:
        return torch.empty((0, case.topk), dtype=torch.int64, device="cuda")

    if case.routing in ("uniform", "zero_tokens"):
        scores = torch.rand((local_tokens, case.num_experts), device="cuda")
        return scores.topk(case.topk, dim=1).indices.to(torch.int64)
    if case.routing == "hotspot":
        return (
            torch.arange(case.topk, device="cuda", dtype=torch.int64)
            .expand(local_tokens, -1)
            .contiguous()
        )
    if case.routing == "single_owner":
        num_local_experts = case.num_experts // dist.get_world_size()
        scores = torch.rand((local_tokens, num_local_experts), device="cuda")
        return scores.topk(case.topk, dim=1).indices.to(torch.int64)
    raise ValueError(f"Unknown routing mode: {case.routing}")


def make_inputs(case, rank, seed):
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    topk_idx = make_topk_idx(case, rank)
    local_tokens = topk_idx.shape[0]
    x = torch.randn(
        (local_tokens, case.hidden), dtype=torch.bfloat16, device="cuda"
    )
    topk_weights = torch.rand(
        (local_tokens, case.topk), dtype=torch.float32, device="cuda"
    )
    if local_tokens:
        topk_weights /= topk_weights.sum(dim=1, keepdim=True)
    return x, topk_idx, topk_weights


class AlltoAllReference:
    def __init__(self, num_experts, world_size):
        self.num_experts = num_experts
        self.world_size = world_size
        self.num_local_experts = num_experts // world_size

    def dispatch(self, x, topk_idx):
        num_tokens, hidden = x.shape
        topk = topk_idx.shape[1]
        flat_experts = topk_idx.reshape(-1)
        local_counts = torch.bincount(flat_experts, minlength=self.num_experts)
        gathered_counts = torch.empty_like(local_counts)
        dist.all_to_all_single(gathered_counts, local_counts)

        send_counts = local_counts.view(self.world_size, -1).sum(dim=1)
        recv_counts = gathered_counts.view(self.world_size, -1).sum(dim=1)
        self.send_split = send_counts.tolist()
        self.recv_split = recv_counts.tolist()

        self.sorted_idx = torch.argsort(flat_experts)
        token_idx = torch.arange(num_tokens, device="cuda").repeat_interleave(topk)
        send_buffer = x[token_idx[self.sorted_idx]]
        recv_buffer = torch.empty(
            (sum(self.recv_split), hidden), dtype=x.dtype, device="cuda"
        )
        dist.all_to_all_single(
            recv_buffer,
            send_buffer,
            output_split_sizes=self.recv_split,
            input_split_sizes=self.send_split,
        )

        local_expert_ids = (
            torch.arange(self.num_experts, device="cuda") % self.num_local_experts
        )
        recv_expert_ids = torch.repeat_interleave(
            local_expert_ids, gathered_counts
        )
        self.sorted_eidx = torch.argsort(recv_expert_ids)
        recv_x = recv_buffer[self.sorted_eidx]
        recv_count = gathered_counts.view(self.world_size, -1).sum(dim=0)
        return recv_x, recv_count.to(torch.int32), None

    def combine(self, expert_out, topk_weights):
        restored = torch.empty_like(expert_out)
        restored[self.sorted_eidx] = expert_out
        recv_buffer = torch.empty(
            (sum(self.send_split), expert_out.shape[1]),
            dtype=expert_out.dtype,
            device="cuda",
        )
        dist.all_to_all_single(
            recv_buffer,
            restored,
            output_split_sizes=self.send_split,
            input_split_sizes=self.recv_split,
        )
        flat_out = torch.empty_like(recv_buffer)
        flat_out[self.sorted_idx] = recv_buffer
        return (
            flat_out.view(
                topk_weights.shape[0], topk_weights.shape[1], expert_out.shape[1]
            )
            * topk_weights.unsqueeze(-1)
        ).sum(dim=1)


def make_expert_output(recv_x, recv_count, rank, world_size, num_experts):
    num_local_experts = num_experts // world_size
    expert_out = torch.empty_like(recv_x)
    if recv_x.dim() == 3:
        for local_expert in range(num_local_experts):
            global_expert = rank * num_local_experts + local_expert
            expert_out[local_expert] = recv_x[local_expert] * expert_factor(
                global_expert
            )
        return expert_out.contiguous()

    offset = 0
    for local_expert in range(num_local_experts):
        count = int(recv_count[local_expert].item())
        global_expert = rank * num_local_experts + local_expert
        expert_out[offset : offset + count] = (
            recv_x[offset : offset + count] * expert_factor(global_expert)
        )
        offset += count
    return expert_out.contiguous()


def run_reference(x, topk_idx, topk_weights, case, rank, world_size):
    reference = AlltoAllReference(case.num_experts, world_size)
    recv_x, recv_count, handle = reference.dispatch(x, topk_idx)
    expert_out = make_expert_output(
        recv_x, recv_count, rank, world_size, case.num_experts
    )
    out = reference.combine(expert_out, topk_weights)
    return out, recv_count, handle


def run_deepep(x, topk_idx, topk_weights, case, rank, world_size):
    import deep_ep

    os.environ["NVSHMEM_QP_DEPTH"] = str(
        max(int(os.environ.get("NVSHMEM_QP_DEPTH", "1024")), (case.num_tokens + 1) * 2)
    )
    size = deep_ep.Buffer.get_low_latency_rdma_size_hint(
        case.num_tokens, case.hidden, world_size, case.num_experts
    )
    buffer = deep_ep.Buffer(
        dist.group.WORLD,
        num_rdma_bytes=size,
        low_latency_mode=True,
        num_qps_per_rank=case.num_experts // world_size,
    )
    recv_x, recv_count, handle, *_ = buffer.low_latency_dispatch(
        x, topk_idx, case.num_tokens, case.num_experts, use_fp8=False
    )
    expert_out = make_expert_output(
        recv_x, recv_count, rank, world_size, case.num_experts
    )
    out, *_ = buffer.low_latency_combine(
        expert_out, topk_idx, topk_weights, handle
    )
    return out, recv_count


def run_mooncake(x, topk_idx, topk_weights, case, rank, world_size):
    from mooncake.mooncake_ep_buffer import Buffer

    size = Buffer.get_ep_buffer_size_hint(
        case.num_tokens, case.hidden, world_size, case.num_experts
    )
    buffer = Buffer(dist.group.WORLD, size)
    active_ranks = torch.ones(world_size, dtype=torch.int32, device="cuda")
    recv_x, recv_count, handle, event, hook = buffer.dispatch(
        x,
        topk_idx,
        active_ranks,
        num_max_dispatch_tokens_per_rank=case.num_tokens,
        num_experts=case.num_experts,
        timeout_us=-1,
        use_fp8=False,
        async_finish=False,
        return_recv_hook=False,
    )
    if hook is not None:
        hook()
    wait_event_if_present(event)
    expert_out = make_expert_output(
        recv_x, recv_count, rank, world_size, case.num_experts
    )
    out, event, hook = buffer.combine(
        expert_out,
        topk_idx,
        topk_weights,
        active_ranks,
        timeout_us=-1,
        handle=handle,
        zero_copy=False,
        async_finish=False,
        return_recv_hook=False,
        out=torch.zeros_like(x),
    )
    if hook is not None:
        hook()
    wait_event_if_present(event)
    return out, recv_count


def run_child(args):
    case = next(case for case in CASES if case.name == args.case)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    torch.set_default_device("cuda")
    dist.init_process_group("nccl")
    try:
        x, topk_idx, topk_weights = make_inputs(case, rank, args.seed)
        ref_out, ref_count, _ = run_reference(
            x, topk_idx, topk_weights, case, rank, world_size
        )
        if args.backend == "alltoall":
            actual_out, actual_count = ref_out, ref_count
        elif args.backend == "deepep":
            actual_out, actual_count = run_deepep(
                x, topk_idx, topk_weights, case, rank, world_size
            )
        else:
            actual_out, actual_count = run_mooncake(
                x, topk_idx, topk_weights, case, rank, world_size
            )

        testing.assert_close(actual_count, ref_count)
        testing.assert_close(actual_out.float(), ref_out.float(), rtol=5e-2, atol=1e-3)
        if actual_out.numel():
            max_abs = (actual_out.float() - ref_out.float()).abs().max().item()
        else:
            max_abs = 0.0
        dist.barrier()
        if rank == 0:
            print(
                f"PATHO_RESULT case={case.name} backend={args.backend} "
                f"status=PASS max_abs={max_abs:.6g}",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


def run_isolated(case, backend, args):
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={args.world_size}",
        str(Path(__file__).resolve()),
        "--child",
        "--case",
        case.name,
        "--backend",
        backend,
        "--seed",
        str(args.seed),
    ]
    env = os.environ.copy()
    env.setdefault("DEVICE_FILTER", "")
    env.setdefault("EP_SUPPRESS_NCCL_CHECK", "1")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
        return "TIMEOUT", "-", output

    marker = next(
        (line for line in output.splitlines() if line.startswith("PATHO_RESULT")),
        None,
    )
    if process.returncode == 0 and marker:
        max_abs = marker.rsplit("max_abs=", 1)[-1]
        return "PASS", max_abs, output
    return "FAIL", "-", output


def run_parent(args):
    selected_cases = [case for case in CASES if not args.case or case.name in args.case]
    selected_backends = [
        backend for backend in BACKENDS if not args.backend or backend in args.backend
    ]
    results = {}
    logs = {}
    for case in selected_cases:
        for backend in selected_backends:
            print(f"Running {case.name} / {backend} ...", flush=True)
            status, max_abs, output = run_isolated(case, backend, args)
            results[(case.name, backend)] = (status, max_abs)
            if status != "PASS":
                logs[(case.name, backend)] = output

    print("\nPathological correctness and robustness matrix", flush=True)
    rows = []
    for case in selected_cases:
        row = [case.name, case.routing]
        for backend in selected_backends:
            status, max_abs = results[(case.name, backend)]
            row.append(f"{status} ({max_abs})" if status == "PASS" else status)
        rows.append(row)
    print_table(("Case", "Routing", *selected_backends), rows)
    print("\nPASS values show max absolute difference against torch all_to_all.")

    for key, output in logs.items():
        print(f"\nFailure log tail: {key[0]} / {key[1]}")
        print("\n".join(output.splitlines()[-20:]), flush=True)

    required_backends = {"alltoall", "mooncake"}
    if args.require_deepep:
        required_backends.add("deepep")
    failed = [
        (case, backend, status)
        for (case, backend), (status, _) in results.items()
        if backend in required_backends and status != "PASS"
    ]
    return 1 if failed else 0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare extreme MoE routing correctness in isolated processes. "
            "Mooncake and all_to_all are gating; DeepEP is informational by default."
        )
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", action="append", choices=[c.name for c in CASES])
    parser.add_argument("--backend", action="append", choices=BACKENDS)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--require-deepep",
        action="store_true",
        help="Make DeepEP failures and timeouts fail the parent test.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.child:
        if len(args.case or []) != 1 or len(args.backend or []) != 1:
            raise ValueError("Child mode requires exactly one --case and --backend")
        args.case = args.case[0]
        args.backend = args.backend[0]
        run_child(args)
        return 0
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
