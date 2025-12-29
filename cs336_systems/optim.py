import torch
import torch.distributed as dist
from torch.optim import Optimizer
from typing import Any, Type, Dict, List, Iterable

class ShardedOptimizer(Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any): 
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        param_groups: List[Dict[str, Any]] = list(params)
        if len(param_groups) == 0:
            raise ValueError("ShardedOptimizer received no parameters")
        if not isinstance(param_groups[0], dict):
            param_groups = [{"params": param_groups}]

        self.all_params: List[torch.Tensor] = []
        self.param_to_rank: Dict[int, int] = {}

        def register_param(param: torch.Tensor) -> int:
            if id(param) not in self.param_to_rank:
                self.param_to_rank[id(param)] = len(self.all_params) % self.world_size
                self.all_params.append(param)
            return self.param_to_rank[id(param)]

        local_param_groups: List[Dict[str, Any]] = []
        for group in param_groups:
            group_params: List[torch.Tensor] = []
            for param in self._unique_params(group["params"]):
                owner_rank = register_param(param)
                if owner_rank == self.rank:
                    group_params.append(param)
            if group_params:
                group_copy = {k: v for k, v in group.items() if k != "params"}
                group_copy["params"] = group_params
                local_param_groups.append(group_copy)

        # Build the wrapped optimizer directly with the local param groups.
        self.contained_optimizer = optimizer_cls(local_param_groups, **kwargs)

        # Initialize the Optimizer base class with the local groups, bypassing our custom add_param_group.
        self._building_base = True
        super().__init__(local_param_groups, self.contained_optimizer.defaults)
        self._building_base = False

        self.param_groups = self.contained_optimizer.param_groups
        self.state = self.contained_optimizer.state
        self.local_params: List[torch.Tensor] = [p for group in self.param_groups for p in group["params"]]

    def step(self, closure=None, **kwargs): 
        def _average_grads():
            for param in self.all_params:
                if param.grad is None:
                    continue
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.mul_(1.0 / self.world_size)

        loss = None
        if closure is None:
            _average_grads()
            loss = self.contained_optimizer.step(**kwargs)
        else:
            def wrapped_closure():
                with torch.enable_grad():
                    loss_val = closure()
                _average_grads()
                return loss_val

            loss = self.contained_optimizer.step(wrapped_closure, **kwargs)

        for param in self.all_params:
            owner_rank = self.param_to_rank[id(param)]
            dist.broadcast(param.data, src=owner_rank)

        return loss

    def add_param_group(self, param_group: Dict[str, Any]): 
        if getattr(self, "_building_base", False):
            return super().add_param_group(param_group)

        params = param_group["params"]
        if isinstance(params, torch.Tensor):
            params = [params]
        params = self._unique_params(params)

        new_param_group = {k: v for k, v in param_group.items() if k != "params"}
        local_params: List[torch.Tensor] = []

        for param in params:
            if id(param) not in self.param_to_rank:
                self.param_to_rank[id(param)] = len(self.all_params) % self.world_size
                self.all_params.append(param)
            if self.param_to_rank[id(param)] == self.rank:
                local_params.append(param)

        if not local_params:
            return
        existing_ids = {id(p) for group in self.contained_optimizer.param_groups for p in group["params"]}
        local_params = [p for p in local_params if id(p) not in existing_ids]
        if not local_params:
            return

        new_param_group["params"] = local_params
        self.contained_optimizer.add_param_group(new_param_group)
        self.param_groups = self.contained_optimizer.param_groups
        self.state = self.contained_optimizer.state
        self.local_params.extend(local_params)

    def zero_grad(self, set_to_none: bool = False):
        for param in self.all_params:
            if param.grad is None:
                continue
            if set_to_none:
                param.grad = None
            else:
                param.grad.zero_()
        self.contained_optimizer.zero_grad(set_to_none=set_to_none)

    @staticmethod
    def _unique_params(params: Iterable[torch.Tensor]) -> List[torch.Tensor]:
        seen: set[int] = set()
        out: List[torch.Tensor] = []
        for p in params:
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            out.append(p)
        return out
