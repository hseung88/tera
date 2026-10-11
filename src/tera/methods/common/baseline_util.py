from typing import Any as any_type, Tuple as tuple_type

import torch
from omegaconf.dictconfig import DictConfig as dict_config_type
from torch.utils.data import DataLoader as data_loader_type, Dataset as dataset_type
from tqdm.auto import tqdm


def my_collate_fn(batch):
    data = [item[0] for item in batch]
    efs = [torch.cat([torch.tensor([item[1]["energy"]]), item[1]["neg_force"]]) for item in batch]
    data_tensor = torch.stack(data, dim=0)
    efs_tensor = torch.stack(efs, dim=0)
    return data_tensor, efs_tensor


def flatten_dataset(
    dataset: dataset_type, collate_fn=None, batch_size=8192
) -> tuple_type[torch.Tensor, torch.Tensor]:
    train_loader = data_loader_type(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    train_x = []
    train_y = []
    for batch_x, batch_y in tqdm(train_loader):
        train_x += [batch_x]
        train_y += [batch_y]
    train_x = torch.cat(train_x, dim=0)
    train_y = torch.cat(train_y, dim=0).squeeze(-1)
    return train_x, train_y


def filter_param(
    named_params: list[tuple_type[str, torch.nn.Parameter]], name: str
) -> list[tuple_type[str, torch.nn.Parameter]]:
    return [param for n, param in named_params if n != name]


def build_kernel(config: dict_config_type | dict) -> any_type:
    from gpytorch.kernels import LinearKernel as linear_kernel_type, MaternKernel as matern_kernel_type, RBFKernel as rbf_kernel_type, RBFKernelGrad as rbf_kernel_grad_type, ScaleKernel as scale_kernel_type

    from tera.methods.ddsvgp.kernel import rbf_kernel_directional_grad

    kernels = {
        cls.__name__: cls
        for cls in (
            linear_kernel_type,
            matern_kernel_type,
            rbf_kernel_type,
            rbf_kernel_grad_type,
            scale_kernel_type,
            rbf_kernel_directional_grad,
        )
    }
    name = config["_target_"]
    if name not in kernels:
        raise ValueError(f"Unknown kernel: {name!r}. Expected one of {tuple(kernels)}")
    return kernels[name](**{k: v for k, v in config.items() if k != "_target_"})
