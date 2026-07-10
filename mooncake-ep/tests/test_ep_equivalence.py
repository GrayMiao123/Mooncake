import os
import random
import traceback
import unittest

import deep_ep
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.testing as testing

from mooncake.mooncake_ep_buffer import Buffer as MooncakeBuffer


def wait_event_if_present(event):
    if event is not None and getattr(event, "event", None) is not None:
        event.current_stream_wait()


class AlltoAll:
    def __init__(self, num_experts, world_size, rank):
        self.num_experts = num_experts
        self.world_size = world_size
        self.rank = rank
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
            topk_idx_flat,
            minlength=self.num_experts,
        )
        num_tokens_per_expert_group = torch.empty_like(num_tokens_per_expert)
        dist.all_to_all_single(
            num_tokens_per_expert_group,
            num_tokens_per_expert,
        )

        send_count = num_tokens_per_expert.view(self.world_size, -1).sum(dim=1)
        recv_count = num_tokens_per_expert_group.view(
            self.world_size, -1
        ).sum(dim=1)
        self.send_split = send_count.tolist()
        self.recv_split = recv_count.tolist()

        self.sorted_idx = torch.argsort(topk_idx_flat)
        src_token_idx = torch.arange(
            num_tokens,
            device=x.device,
        ).repeat_interleave(topk)
        send_buff = x[src_token_idx[self.sorted_idx]]

        recv_buff = torch.empty(
            (sum(self.recv_split), hidden),
            dtype=x.dtype,
            device=x.device,
        )
        dist.all_to_all_single(
            recv_buff,
            send_buff,
            self.recv_split,
            self.send_split,
        )

        local_expert_ids = (
            torch.arange(
                self.world_size * self.num_local_experts,
                device=x.device,
            )
            % self.num_local_experts
        )
        recv_buff_eid = torch.repeat_interleave(
            local_expert_ids,
            num_tokens_per_expert_group,
        )
        self.sorted_eidx = torch.argsort(recv_buff_eid)
        recv_x = recv_buff[self.sorted_eidx]

        local_expert_count = num_tokens_per_expert_group.view(
            self.world_size, -1
        ).sum(dim=0)
        return recv_x, local_expert_count.to(torch.int32)

    def combine(self, expert_out, topk_weights):
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
        out = (
            out_flat.view(num_tokens, topk, -1)
            * topk_weights.unsqueeze(-1)
        ).sum(dim=1)
        return out


def make_inputs(rank, num_tokens, hidden, num_experts, topk, seed):
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    random.seed(seed + rank)

    x = torch.randn(
        (num_tokens, hidden),
        dtype=torch.bfloat16,
        device="cuda",
    )
    topk_idx = torch.randint(
        0,
        num_experts,
        (num_tokens, topk),
        dtype=torch.int64,
        device="cuda",
    )
    topk_weights = torch.rand(
        (num_tokens, topk),
        dtype=torch.float32,
        device="cuda",
    )
    topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)
    return x, topk_idx, topk_weights


def expert_factor(expert_id):
    return expert_id * 0.1 + 1.0


def run_deepep(x, topk_idx, topk_weights, max_tokens, num_experts, world_size, rank):
    required_qp_depth = (max_tokens + 1) * 2
    current_qp_depth = int(os.environ.get("NVSHMEM_QP_DEPTH", "1024"))
    os.environ["NVSHMEM_QP_DEPTH"] = str(max(current_qp_depth, required_qp_depth))

    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
        max_tokens,
        x.shape[1],
        world_size,
        num_experts,
    )
    buffer = deep_ep.Buffer(
        dist.group.WORLD,
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=num_experts // world_size,
    )
    recv_x, recv_count, handle, *_ = buffer.low_latency_dispatch(
        x,
        topk_idx,
        max_tokens,
        num_experts,
        use_fp8=False,
    )

    num_local_experts = num_experts // world_size
    expert_out = torch.empty_like(recv_x)
    for local_expert in range(num_local_experts):
        global_expert = rank * num_local_experts + local_expert
        expert_out[local_expert] = recv_x[local_expert] * expert_factor(
            global_expert
        )

    combined, *_ = buffer.low_latency_combine(
        expert_out.contiguous(),
        topk_idx,
        topk_weights,
        handle,
    )
    return combined, recv_count


def run_mooncake(x, topk_idx, topk_weights, max_tokens, num_experts, world_size, rank):
    num_ep_buffer_bytes = MooncakeBuffer.get_ep_buffer_size_hint(
        max_tokens,
        x.shape[1],
        world_size,
        num_experts,
    )
    buffer = MooncakeBuffer(dist.group.WORLD, num_ep_buffer_bytes)
    active_ranks = torch.ones((world_size,), dtype=torch.int32, device="cuda")

    recv_x, recv_count, handle, event, hook = buffer.dispatch(
        x,
        topk_idx,
        active_ranks,
        num_max_dispatch_tokens_per_rank=max_tokens,
        num_experts=num_experts,
        timeout_us=-1,
        use_fp8=False,
        async_finish=False,
        return_recv_hook=False,
    )
    if hook is not None:
        hook()
    wait_event_if_present(event)

    num_local_experts = num_experts // world_size
    expert_out = torch.empty_like(recv_x)
    for local_expert in range(num_local_experts):
        global_expert = rank * num_local_experts + local_expert
        expert_out[local_expert] = recv_x[local_expert] * expert_factor(
            global_expert
        )

    combined, event, hook = buffer.combine(
        expert_out.contiguous(),
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
    return combined, recv_count


def run_reference(x, topk_idx, topk_weights, num_experts, world_size, rank):
    ref = AlltoAll(num_experts, world_size, rank)
    recv_x, recv_count = ref.dispatch(x, topk_idx)

    expert_out = torch.empty_like(recv_x)
    offset = 0
    num_local_experts = num_experts // world_size
    for local_expert in range(num_local_experts):
        count = int(recv_count[local_expert].item())
        global_expert = rank * num_local_experts + local_expert
        expert_out[offset : offset + count] = recv_x[
            offset : offset + count
        ] * expert_factor(global_expert)
        offset += count

    combined = ref.combine(expert_out.contiguous(), topk_weights)
    return combined, recv_count


def assert_close_with_report(name, actual, expected, rank, rtol=5e-2, atol=1e-3):
    actual_f = actual.float()
    expected_f = expected.float()
    abs_diff = (actual_f - expected_f).abs().max().item()
    denom = expected_f.abs().max().item()
    rel_diff = abs_diff / (denom if denom != 0.0 else 1.0)
    print(
        f"[rank {rank}] {name}: abs_diff={abs_diff:.6g}, rel_diff={rel_diff:.6g}",
        flush=True,
    )
    testing.assert_close(actual_f, expected_f, rtol=rtol, atol=atol)


def run_equivalence(rank, world_size, config):
    torch.cuda.set_device(rank)
    torch.set_default_device("cuda")
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )

    try:
        num_experts = config["num_experts"]
        assert num_experts % world_size == 0

        x, topk_idx, topk_weights = make_inputs(
            rank,
            config["num_tokens"],
            config["hidden"],
            num_experts,
            config["topk"],
            config["seed"],
        )

        ref_out, ref_count = run_reference(
            x,
            topk_idx,
            topk_weights,
            num_experts,
            world_size,
            rank,
        )
        deepep_out, deepep_count = run_deepep(
            x,
            topk_idx,
            topk_weights,
            config["num_tokens"],
            num_experts,
            world_size,
            rank,
        )
        mooncake_out, mooncake_count = run_mooncake(
            x,
            topk_idx,
            topk_weights,
            config["num_tokens"],
            num_experts,
            world_size,
            rank,
        )

        print(
            f"[rank {rank}] ref_count={ref_count.tolist()} "
            f"deepep_count={deepep_count.tolist()} "
            f"mooncake_count={mooncake_count.tolist()}",
            flush=True,
        )

        testing.assert_close(deepep_count, ref_count)
        testing.assert_close(mooncake_count, ref_count)
        assert_close_with_report("DeepEP vs reference", deepep_out, ref_out, rank)
        assert_close_with_report(
            "Mooncake EP vs reference",
            mooncake_out,
            ref_out,
            rank,
        )
        assert_close_with_report(
            "Mooncake EP vs DeepEP",
            mooncake_out,
            deepep_out,
            rank,
        )

        dist.barrier()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        dist.destroy_process_group()


class TestMooncakeEPEquivalence(unittest.TestCase):
    def setUp(self):
        self.world_size = torch.cuda.device_count()
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        self.base_master_port = int(os.getenv("MASTER_PORT", "29511"))

    def test_deepep_mooncake_alltoall_equivalence(self):
        self.assertGreaterEqual(self.world_size, 2)
        # Mooncake EP dispatch uses one counting worker SM and requires
        # ceil(num_experts / kNumWarpGroups) > 1, so keep num_experts >= 16.
        configs = [
            {
                "name": "small_top2_e16",
                "num_tokens": 64,
                "hidden": 7168,
                "num_experts": 16,
                "topk": 2,
                "seed": 2026,
            },
            {
                "name": "medium_top2_e32",
                "num_tokens": 256,
                "hidden": 7168,
                "num_experts": 32,
                "topk": 2,
                "seed": 2027,
            },
            {
                "name": "medium_top4_e32",
                "num_tokens": 256,
                "hidden": 7168,
                "num_experts": 32,
                "topk": 4,
                "seed": 2028,
            },
            {
                "name": "large_top2_e64",
                "num_tokens": 512,
                "hidden": 7168,
                "num_experts": 64,
                "topk": 2,
                "seed": 2029,
            },
        ]

        for idx, config in enumerate(configs):
            with self.subTest(config=config["name"]):
                os.environ["MASTER_PORT"] = str(self.base_master_port + idx)
                mp.spawn(
                    run_equivalence,
                    args=(self.world_size, config),
                    nprocs=self.world_size,
                    join=True,
                    daemon=False,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
