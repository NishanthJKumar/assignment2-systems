import torch
import torch.nn as nn

import torch.distributed as dist

class DDP(nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.contained_module = module
        self.add_module('module', module)
        # Broadcast parameters from rank 0 to all other ranks
        for param in self.contained_module.parameters():
            dist.broadcast(param.data, src=0)
        # Register hooks for per-parameter gradient all-reduce
        self._hooks = []
        for param in self.contained_module.parameters():
            if param.requires_grad:
                def _make_hook(p):
                    def hook(grad):
                        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                        grad /= dist.get_world_size()
                        return grad
                    return hook
                h = param.register_hook(_make_hook(param))
                self._hooks.append(h)
        self._grad_synced = False

    def forward(self, *inputs, **kwargs):
        return self.contained_module(*inputs, **kwargs)
