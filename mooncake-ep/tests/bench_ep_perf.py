import argparse
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import deep_ep
import torch
import torch.distributed as dist
import torch.testing as testing

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "mooncake-wheel"))

from mooncake.mooncake_ep_buffer import Buffer as MooncakeBuffer


def wait_event_if_present(event):
    if event is not None and getattr(event, "event", None) is not None:
        event.current_stream_wait()


def expert_factor(expert_id):
    return expert_id * 0.1 + 1.0


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
        expert_out[offset : offset + count] = recv_x[
            offset : offset + count
        ] * expert_factor(global_expert)
        offset += count
    return expert_out.contiguous()


class AlltoAllReference:
    def __init__(self, num_experts, world_size):
        self.num_experts = num_experts
        self.world_size = world_size
        self.num_local_experts = num_experts // world_size
        self.sorted_idx = None
        self.sorted_eidx = None
        self.send_split = None
        self.recv_split = None

    def dispatch(self, x, topk_idx):
        num_tokens, hidden = x.shape
        _, topk = topk_idx.shape
        topk_idx_flat = topk_idx.reshape(-1)

        num_tokens_per_expert = torch.bincount(
            topk_idx_flat, minlength=self.num_experts
        )
        num_tokens_per_expert_group = torch.empty_like(num_tokens_per_expert)
        dist.all_to_all_single(num_tokens_per_expert_group, num_tokens_per_expert)

        send_count = num_tokens_per_expert.view(self.world_size, -1).sum(dim=1)
        recv_count = num_tokens_per_expert_group.view(self.world_size, -1).sum(dim=1)
        self.send_split = send_count.tolist()
        self.recv_split = recv_count.tolist()

        self.sorted_idx = torch.argsort(topk_idx_flat)
        src_token_idx = torch.arange(num_tokens, device=x.device).repeat_interleave(topk)
        send_buff = x[src_token_idx[self.sorted_idx]]

        recv_buff = torch.empty(
            (sum(self.recv_split), hidden), dtype=x.dtype, device=x.device
        )
        dist.all_to_all_single(
            recv_buff,
            send_buff,
            output_split_sizes=self.recv_split,
            input_split_sizes=self.send_split,
        )

        local_expert_ids = (
            torch.arange(self.world_size * self.num_local_experts, device=x.device)
            % self.num_local_experts
        )
        recv_buff_eid = torch.repeat_interleave(
            local_expert_ids, num_tokens_per_expert_group
        )
        self.sorted_eidx = torch.argsort(recv_buff_eid)
        recv_x = recv_buff[self.sorted_eidx]

        local_expert_count = num_tokens_per_expert_group.view(
            self.world_size, -1
        ).sum(dim=0)
        return recv_x, local_expert_count.to(torch.int32), None

    def combine(self, expert_out, topk_idx, topk_weights, handle=None):
        del topk_idx, handle
        num_tokens, topk = topk_weights.shape

        recv_buff_restored = torch.empty_like(expert_out)
        recv_buff_restored[self.sorted_eidx] = expert_out

        recv_buff = torch.empty(
            (sum(self.send_split), expert_out.shape[1]),
            dtype=expert_out.dtype,
            device=expert_out.device,
        )
        dist.all_to_all_single(
            recv_buff,
            recv_buff_restored,
            output_split_sizes=self.send_split,
            input_split_sizes=self.recv_split,
        )

        out_flat = torch.empty(
            (num_tokens * topk, expert_out.shape[1]),
            dtype=expert_out.dtype,
            device=expert_out.device,
        )
        out_flat[self.sorted_idx] = recv_buff
        return (
            out_flat.view(num_tokens, topk, -1) * topk_weights.unsqueeze(-1)
        ).sum(dim=1)


class DeepEPImpl:
    def __init__(self, group, max_tokens, hidden, world_size, num_experts):
        required_qp_depth = (max_tokens + 1) * 2
        current_qp_depth = int(os.environ.get("NVSHMEM_QP_DEPTH", "1024"))
        os.environ["NVSHMEM_QP_DEPTH"] = str(max(current_qp_depth, required_qp_depth))

        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
            max_tokens, hidden, world_size, num_experts
        )
        self.buffer = deep_ep.Buffer(
            group,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=num_experts // world_size,
        )
        self.max_tokens = max_tokens
        self.num_experts = num_experts

    def dispatch(self, x, topk_idx):
        recv_x, recv_count, handle, *_ = self.buffer.low_latency_dispatch(
            x, topk_idx, self.max_tokens, self.num_experts, use_fp8=False
        )
        return recv_x, recv_count, handle

    def combine(self, expert_out, topk_idx, topk_weights, handle):
        combined, *_ = self.buffer.low_latency_combine(
            expert_out, topk_idx, topk_weights, handle
        )
        return combined


class MooncakeEPImpl:
    def __init__(self, group, max_tokens, hidden, world_size, num_experts):
        num_ep_buffer_bytes = MooncakeBuffer.get_ep_buffer_size_hint(
            max_tokens, hidden, world_size, num_experts
        )
        self.buffer = MooncakeBuffer(group, num_ep_buffer_bytes)
        self.active_ranks = torch.ones((world_size,), dtype=torch.int32, device="cuda")
        self.max_tokens = max_tokens
        self.num_experts = num_experts

    def dispatch(self, x, topk_idx):
        recv_x, recv_count, handle, event, hook = self.buffer.dispatch(
            x,
            topk_idx,
            self.active_ranks,
            num_max_dispatch_tokens_per_rank=self.max_tokens,
            num_experts=self.num_experts,
            timeout_us=-1,
            use_fp8=False,
            async_finish=False,
            return_recv_hook=False,
        )
        if hook is not None:
            hook()
        wait_event_if_present(event)
        return recv_x, recv_count, handle

    def combine(self, expert_out, topk_idx, topk_weights, handle):
        combined, event, hook = self.buffer.combine(
            expert_out,
            topk_idx,
            topk_weights,
            self.active_ranks,
            timeout_us=-1,
            handle=handle,
            zero_copy=False,
            async_finish=False,
            return_recv_hook=False,
            out=None,
        )
        if hook is not None:
            hook()
        wait_event_if_present(event)
        return combined


@dataclass
class BenchResult:
    dispatch_us: float
    combine_us: float
    total_us: float


@dataclass(frozen=True)
class BenchCase:
    name: str
    num_tokens: int
    hidden: int
    num_experts: int
    topk: int
    routing: str = "uniform"


def print_section(title):
    print("\n" + "=" * 96, flush=True)
    print(title, flush=True)
    print("=" * 96, flush=True)


def print_table(headers, rows):
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))

    def format_row(row):
        return "  ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row))

    print(format_row(headers), flush=True)
    print(format_row(["-" * width for width in widths]), flush=True)
    for row in rows:
        print(format_row(row), flush=True)


def cuda_bench(fn, warmup, repeat):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()

    elapsed_ms = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_events, end_events)],
        dtype=torch.float64,
        device="cuda",
    )
    return elapsed_ms.median().item() * 1000.0


def max_across_ranks(value_us):
    value = torch.tensor(value_us, dtype=torch.float64, device="cuda")
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return value.item()


def bench_impl(impl, x, topk_idx, topk_weights, rank, world_size, num_experts, args):
    recv_x, recv_count, handle = impl.dispatch(x, topk_idx)
    expert_out = make_expert_output(recv_x, recv_count, rank, world_size, num_experts)
    impl.combine(expert_out, topk_idx, topk_weights, handle)
    torch.cuda.synchronize()
    dist.barrier()

    dispatch_us = cuda_bench(
        lambda: impl.dispatch(x, topk_idx),
        args.warmup,
        args.repeat,
    )

    recv_x, recv_count, handle = impl.dispatch(x, topk_idx)
    expert_out = make_expert_output(recv_x, recv_count, rank, world_size, num_experts)
    torch.cuda.synchronize()
    dist.barrier()

    combine_us = cuda_bench(
        lambda: impl.combine(expert_out, topk_idx, topk_weights, handle),
        args.warmup,
        args.repeat,
    )

    def full_pipeline():
        recv_x, _, handle = impl.dispatch(x, topk_idx)
        impl.combine(recv_x, topk_idx, topk_weights, handle)

    dist.barrier()
    total_us = cuda_bench(full_pipeline, args.warmup, args.repeat)

    return BenchResult(
        dispatch_us=max_across_ranks(dispatch_us),
        combine_us=max_across_ranks(combine_us),
        total_us=max_across_ranks(total_us),
    )


def check_correctness(results, rank):
    ref_out, ref_count = results["alltoall"]
    rows = []
    for name, (out, count) in results.items():
        testing.assert_close(count, ref_count)
        actual = out.float()
        expected = ref_out.float()
        abs_diff = (actual - expected).abs().max().item()
        denom = expected.abs().max().item()
        rel_diff = abs_diff / (denom if denom != 0.0 else 1.0)
        rows.append((name, f"{abs_diff:.6g}", f"{rel_diff:.6g}"))
        testing.assert_close(actual, expected, rtol=5e-2, atol=1e-3)
    if rank == 0:
        print_table(("Impl", "Max Abs Diff", "Max Rel Diff"), rows)


def make_topk_idx(num_tokens, num_experts, topk, routing):
    if routing == "uniform":
        return torch.randint(
            0, num_experts, (num_tokens, topk), dtype=torch.int64, device="cuda"
        )
    if routing == "hotspot":
        hot_experts = max(topk, num_experts // 8)
        return torch.randint(
            0, hot_experts, (num_tokens, topk), dtype=torch.int64, device="cuda"
        )
    if routing == "rank_skew":
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        num_local_experts = num_experts // world_size
        owner = rank if rank % 2 == 0 else (rank - 1)
        base = owner * num_local_experts
        return base + torch.randint(
            0, num_local_experts, (num_tokens, topk), dtype=torch.int64, device="cuda"
        )
    raise ValueError(f"Unknown routing mode: {routing}")


def make_inputs(rank, case, seed):
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    random.seed(seed + rank)

    x = torch.randn((case.num_tokens, case.hidden), dtype=torch.bfloat16, device="cuda")
    topk_idx = make_topk_idx(
        case.num_tokens, case.num_experts, case.topk, case.routing
    )
    topk_weights = torch.rand(
        (case.num_tokens, case.topk), dtype=torch.float32, device="cuda"
    )
    topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)
    return x, topk_idx, topk_weights


def measure_load_balance(topk_idx, num_experts):
    local_counts = torch.bincount(topk_idx.reshape(-1), minlength=num_experts).float()
    global_counts = local_counts.clone()
    dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)
    mean = global_counts.mean().item()
    max_count = global_counts.max().item()
    nonzero = int((global_counts > 0).sum().item())
    imbalance = max_count / mean if mean > 0 else 0.0
    return imbalance, nonzero


def run_once(args, rank, world_size, case):
    num_tokens = case.num_tokens
    hidden = case.hidden
    num_experts = case.num_experts
    topk = case.topk
    x, topk_idx, topk_weights = make_inputs(
        rank, case, args.seed
    )
    imbalance, active_experts = measure_load_balance(topk_idx, num_experts)

    impls = {
        "alltoall": AlltoAllReference(num_experts, world_size),
        "deepep": DeepEPImpl(
            dist.group.WORLD, num_tokens, hidden, world_size, num_experts
        ),
        "mooncake": MooncakeEPImpl(
            dist.group.WORLD, num_tokens, hidden, world_size, num_experts
        ),
    }

    correctness = {}
    for name, impl in impls.items():
        recv_x, recv_count, handle = impl.dispatch(x, topk_idx)
        expert_out = make_expert_output(
            recv_x, recv_count, rank, world_size, num_experts
        )
        out = impl.combine(expert_out, topk_idx, topk_weights, handle)
        correctness[name] = (out, recv_count)
    if rank == 0:
        print_section(
            f"Case: {case.name}"
        )
        print(
            f"Shape: tokens={num_tokens}, hidden={hidden}, experts={num_experts}, "
            f"topk={topk}, routing={case.routing}, active_experts={active_experts}/"
            f"{num_experts}, imbalance={imbalance:.2f}x",
            flush=True,
        )
        print("Correctness", flush=True)
    check_correctness(correctness, rank)

    byte_per_token = hidden * 2
    dispatch_bytes = num_tokens * topk * byte_per_token
    combine_bytes = num_tokens * topk * byte_per_token
    total_bytes = dispatch_bytes + combine_bytes

    bench_impls = impls
    if args.skip_alltoall_bench:
        bench_impls = {
            name: impl for name, impl in impls.items() if name != "alltoall"
        }

    bench_results = {}
    for name, impl in bench_impls.items():
        result = bench_impl(
            impl, x, topk_idx, topk_weights, rank, world_size, num_experts, args
        )
        bench_results[name] = result

    baseline_total = bench_results.get("alltoall", bench_results["deepep"]).total_us
    if rank == 0:
        rows = []
        for name, result in bench_results.items():
            dispatch_gbps = dispatch_bytes / result.dispatch_us / 1e3
            combine_gbps = combine_bytes / result.combine_us / 1e3
            total_gbps = total_bytes / result.total_us / 1e3
            speedup = baseline_total / result.total_us
            rows.append(
                (
                    name,
                    f"{result.dispatch_us:.2f}",
                    f"{result.combine_us:.2f}",
                    f"{result.total_us:.2f}",
                    f"{dispatch_gbps:.2f}",
                    f"{combine_gbps:.2f}",
                    f"{total_gbps:.2f}",
                    f"{speedup:.2f}x",
                )
            )
        print("\nPerformance", flush=True)
        print_table(
            (
                "Impl",
                "Dispatch p50 us",
                "Combine p50 us",
                "Comm E2E p50 us",
                "Dispatch GB/s",
                "Combine GB/s",
                "Comm E2E GB/s",
                "Speedup",
            ),
            rows,
        )
        print(
            "\nNotes: latency columns report median over repeat iterations. "
            "GB/s uses logical token payload bytes "
            "(num_tokens * topk * hidden * sizeof(bf16)).",
            flush=True,
        )
        print(
            "Notes: Comm E2E measures dispatch followed by identity-expert combine; "
            "expert computation is excluded.",
            flush=True,
        )
        if args.skip_alltoall_bench:
            print(
                "Notes: alltoall correctness is still checked; speedup baseline is DeepEP.",
                flush=True,
            )
    return bench_results


def parse_config(raw):
    fields = raw.split(",")
    if len(fields) not in (4, 5):
        raise argparse.ArgumentTypeError(
            "config must be num_tokens,hidden,num_experts,topk[,routing]"
        )
    num_tokens, hidden, num_experts, topk = [int(x) for x in fields[:4]]
    routing = fields[4] if len(fields) == 5 else "uniform"
    return BenchCase(
        name=f"custom_t{num_tokens}_h{hidden}_e{num_experts}_k{topk}_{routing}",
        num_tokens=num_tokens,
        hidden=hidden,
        num_experts=num_experts,
        topk=topk,
        routing=routing,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Mooncake EP, DeepEP, and torch all_to_all EP paths."
    )
    parser.add_argument(
        "--config",
        action="append",
        type=parse_config,
        default=None,
        help=(
            "Benchmark config: num_tokens,hidden,num_experts,topk[,routing]. "
            "Routing can be uniform, hotspot, or rank_skew. "
            "Non-uniform modes are pathological stress cases. Repeatable."
        ),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--preset",
        choices=("quick", "production", "stress"),
        default="quick",
        help="Shape suite to run when --config is not provided.",
    )
    parser.add_argument(
        "--skip-alltoall-bench",
        action="store_true",
        help="Check alltoall correctness but skip timing it for large shapes.",
    )
    return parser.parse_args()


def get_preset_configs(preset):
    if preset == "quick":
        return [
            BenchCase("quick_t128_e16_k2_uniform", 128, 7168, 16, 2, "uniform"),
            BenchCase("quick_t256_e32_k2_uniform", 256, 7168, 32, 2, "uniform"),
            BenchCase("quick_t256_e32_k4_uniform", 256, 7168, 32, 4, "uniform"),
            BenchCase("quick_t512_e64_k2_uniform", 512, 7168, 64, 2, "uniform"),
        ]
    if preset == "production":
        return [
            BenchCase("decode_t256_h7168_e32_k2_uniform", 256, 7168, 32, 2),
            BenchCase("decode_t512_h7168_e64_k2_uniform", 512, 7168, 64, 2),
            BenchCase("prefill_t1024_h7168_e64_k2_uniform", 1024, 7168, 64, 2),
            BenchCase("prefill_t2048_h7168_e64_k2_uniform", 2048, 7168, 64, 2),
            BenchCase("prefill_t4096_h7168_e64_k2_uniform", 4096, 7168, 64, 2),
            BenchCase("fanout_t1024_h7168_e64_k4_uniform", 1024, 7168, 64, 4),
            BenchCase("fanout_t2048_h7168_e64_k4_uniform", 2048, 7168, 64, 4),
            BenchCase("large_e_t1024_h7168_e128_k4_uniform", 1024, 7168, 128, 4),
            BenchCase("wide_t1024_h8192_e64_k2_uniform", 1024, 8192, 64, 2),
            BenchCase("wide_t2048_h8192_e64_k2_uniform", 2048, 8192, 64, 2),
        ]
    if preset == "stress":
        return [
            BenchCase("stress_t512_h7168_e64_k2_uniform", 512, 7168, 64, 2),
            BenchCase("stress_t1024_h7168_e64_k2_uniform", 1024, 7168, 64, 2),
            BenchCase("stress_t2048_h7168_e64_k2_uniform", 2048, 7168, 64, 2),
            BenchCase("stress_t4096_h7168_e64_k2_uniform", 4096, 7168, 64, 2),
            BenchCase("stress_t1024_h7168_e64_k4_uniform", 1024, 7168, 64, 4),
            BenchCase("stress_t2048_h7168_e64_k4_uniform", 2048, 7168, 64, 4),
            BenchCase("stress_t4096_h7168_e64_k4_uniform", 4096, 7168, 64, 4),
            BenchCase("stress_t1024_h7168_e128_k4_uniform", 1024, 7168, 128, 4),
            BenchCase("stress_t2048_h7168_e128_k4_uniform", 2048, 7168, 128, 4),
            BenchCase("stress_t4096_h7168_e128_k4_uniform", 4096, 7168, 128, 4),
            BenchCase("stress_t1024_h8192_e128_k4_uniform", 1024, 8192, 128, 4),
            BenchCase("stress_t2048_h8192_e128_k4_uniform", 2048, 8192, 128, 4),
            BenchCase("stress_t4096_h8192_e128_k4_uniform", 4096, 8192, 128, 4),
            BenchCase("stress_t1024_h7168_e128_k8_uniform", 1024, 7168, 128, 8),
            BenchCase("stress_t2048_h7168_e128_k8_uniform", 2048, 7168, 128, 8),
        ]
    raise ValueError(f"Unknown preset: {preset}")


def main():
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    torch.set_default_device("cuda")
    dist.init_process_group(backend="nccl")

    try:
        configs = args.config or get_preset_configs(args.preset)
        summary_rows = []
        for case in configs:
            assert case.num_experts % world_size == 0
            results = run_once(args, rank, world_size, case)
            if rank == 0 and "deepep" in results and "mooncake" in results:
                deepep = results["deepep"]
                mooncake = results["mooncake"]
                summary_rows.append(
                    (
                        case.name,
                        case.routing,
                        f"{mooncake.dispatch_us / deepep.dispatch_us:.2f}x",
                        f"{mooncake.combine_us / deepep.combine_us:.2f}x",
                        f"{mooncake.total_us / deepep.total_us:.2f}x",
                        f"{deepep.total_us / mooncake.total_us:.2f}x",
                    )
                )
            dist.barrier()
        if rank == 0 and summary_rows:
            print_section("Summary: Mooncake EP vs DeepEP")
            print_table(
                (
                    "Case",
                    "Routing",
                    "Dispatch Lat Ratio",
                    "Combine Lat Ratio",
                        "Comm E2E Lat Ratio",
                    "Mooncake/DeepEP Throughput",
                ),
                summary_rows,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
