import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Optional


class DDPBucketed(nn.Module):
    def __init__(self, module: nn.Module, bucket_size_mb: float):
        super().__init__()
        self.contained_module = module
        self.add_module("module", module)

        if not dist.is_initialized():
            raise RuntimeError("dist must be initialized before constructing DDPBucketed")

        self.world_size = dist.get_world_size()

        # 1) Broadcast initial params from rank 0
        with torch.no_grad():
            for p in self.contained_module.parameters():
                dist.broadcast(p, src=0)

        # 2) Bucket ONLY requires_grad params, in reverse order (for overlap)
        params = [p for p in self.contained_module.parameters() if p.requires_grad]
        params = list(reversed(params))

        cap_bytes = float(bucket_size_mb) * 1e6  # IMPORTANT: MB == 1e6 for these tests

        self.param_buckets: List[List[torch.nn.Parameter]] = []
        cur: List[torch.nn.Parameter] = []
        cur_bytes = 0.0

        for p in params:
            p_bytes = p.numel() * p.element_size()
            if cur and (cur_bytes + p_bytes > cap_bytes):
                self.param_buckets.append(cur)
                cur = []
                cur_bytes = 0.0
            cur.append(p)
            cur_bytes += p_bytes
        if cur:
            self.param_buckets.append(cur)

        # 3) Preallocate flat buffers + views
        self._bucket_flat: List[torch.Tensor] = []
        self._bucket_views: List[List[torch.Tensor]] = []
        self._bucket_ready: List[int] = []
        self._bucket_expected: List[int] = []
        self._bucket_work: List[Optional[dist.Work]] = []

        for bucket in self.param_buckets:
            device = bucket[0].device
            dtype = bucket[0].dtype
            total_elems = sum(p.numel() for p in bucket)
            flat = torch.empty(total_elems, device=device, dtype=dtype)

            views = []
            off = 0
            for p in bucket:
                n = p.numel()
                views.append(flat[off:off+n].view_as(p))
                off += n

            self._bucket_flat.append(flat)
            self._bucket_views.append(views)
            self._bucket_ready.append(0)
            self._bucket_expected.append(len(bucket))
            self._bucket_work.append(None)

        # 4) Hooks: copy grad -> view, return view so param.grad uses bucket buffer.
        self._hooks = []
        for b_idx, bucket in enumerate(self.param_buckets):
            for p_idx, p in enumerate(bucket):

                def _make_hook(bi: int, pi: int, param: torch.nn.Parameter):
                    def hook(grad: torch.Tensor):
                        view = self._bucket_views[bi][pi]
                        view.copy_(grad)

                        # Make optimizer see bucket-backed grad
                        param.grad = view

                        self._bucket_ready[bi] += 1
                        if self._bucket_ready[bi] == self._bucket_expected[bi]:
                            self._bucket_work[bi] = dist.all_reduce(
                                self._bucket_flat[bi], op=dist.ReduceOp.SUM, async_op=True
                            )
                        return grad
                    return hook

                self._hooks.append(p.register_hook(_make_hook(b_idx, p_idx, p)))

    def forward(self, *args, **kwargs):
        return self.contained_module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        # Wait for all async reductions
        for work in self._bucket_work:
            if work is not None:
                work.wait()

        # Average grads in-place (SUM -> MEAN)
        if self.world_size > 1:
            scale = 1.0 / float(self.world_size)
            for flat in self._bucket_flat:
                flat.mul_(scale)

        # Reset per-iteration state
        for i in range(len(self._bucket_ready)):
            self._bucket_ready[i] = 0
            self._bucket_work[i] = None
