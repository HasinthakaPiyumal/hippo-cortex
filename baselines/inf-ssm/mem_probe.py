"""Peak GPU memory and step time of one Inf-SSM training step (Vim-small, current + old model), on one GPU.

    python mem_probe.py                      # per-GPU batches 8 16 32 64, grad_ckpt on and off
    python mem_probe.py --batches 16 32 --ckpt true
    python mem_probe.py --batches 32 --ckpt true --reg true false --profile   # where the step time goes

Use it to pick --micro_batch / --grad_ckpt / --use_amp for a GPU: with --data_parallel each GPU sees
micro_batch / num_gpus images per step.
"""
import argparse
import time

import torch

from inf_ssm import InfSSMForward
from mamba_vim import KERNEL_IMPORT_ERRORS, vim_small


def probe(batch: int, ckpt: bool, amp: bool, reg: bool = True, profile: bool = False, steps: int = 3) -> str:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    net = vim_small(num_classes=100, drop_path_rate=0.0, grad_ckpt=ckpt).cuda().train()
    old = vim_small(num_classes=100, drop_path_rate=0.0).cuda().eval().requires_grad_(False)
    old.load_state_dict(net.state_dict())
    with torch.no_grad():  # make the two models differ, as after one update
        for p in net.parameters():
            p.add_(1e-4 * torch.randn_like(p))
    fwd = InfSSMForward(net, old if reg else None, 3e4, 'inf_ssm')
    opt = torch.optim.AdamW(net.parameters(), lr=1e-6)
    x = torch.randn(batch, 3, 224, 224, device='cuda')
    times = []
    try:
        for _ in range(steps):
            torch.cuda.synchronize()
            t0 = time.time()
            opt.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
                logits, pen = fwd(x)
                loss = logits.float().logsumexp(1).mean() + pen.mean()
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        res = f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:5.2f} GB | {min(times):5.2f} s/step | {batch / min(times):5.1f} img/s"
        if profile:
            from torch.profiler import ProfilerActivity, profile as tprofile
            with tprofile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                opt.zero_grad()
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
                    logits, pen = fwd(x)
                    loss = logits.float().logsumexp(1).mean() + pen.mean()
                loss.backward()
                torch.cuda.synchronize()
            res += "\n" + prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=60)
    except torch.cuda.OutOfMemoryError:
        res = f"OOM (peak before failing {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB)"
    del net, old, fwd, opt, x
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batches', type=int, nargs='+', default=[8, 16, 32, 64])
    p.add_argument('--ckpt', type=str, nargs='+', default=['true', 'false'])
    p.add_argument('--amp', type=str, default='false')
    p.add_argument('--reg', type=str, nargs='+', default=['true'], help='true = with the old model and penalty, false = plain step')
    p.add_argument('--profile', action='store_true', help='print the top CUDA ops of one step')
    args = p.parse_args()
    assert not KERNEL_IMPORT_ERRORS, KERNEL_IMPORT_ERRORS
    free, total = torch.cuda.mem_get_info()
    print(f"{torch.cuda.get_device_name()}: {free / 2 ** 30:.1f} / {total / 2 ** 30:.1f} GB free, amp={args.amp}")
    for reg in args.reg:
        for ckpt in args.ckpt:
            for b in args.batches:
                print(f"batch/GPU {b:3d} grad_ckpt={ckpt:5s} reg={reg:5s}: "
                      f"{probe(b, ckpt == 'true', args.amp == 'true', reg == 'true', args.profile)}", flush=True)


if __name__ == "__main__":
    main()
