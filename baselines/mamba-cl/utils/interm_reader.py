"""IntermReader: identity module that marks the input of a parameter for the null-space hooks.

Kept in its own module (no transformers import) so that both mamba_block.py (De-focus Mamba) and
vim_mamba.py (Vim-small) can use the same class, which train_eval.get_interm_tensor_dict checks with
isinstance(module, utils.mamba_block.IntermReader).
"""
from torch import nn, Tensor


class IntermReader(nn.Module):
    def __init__(self, dst_param_id: int, module_name: str, other_args: dict = {}) -> None:
        super().__init__()
        self.dst_param_id = dst_param_id
        self.module_name = module_name
        self.other_args = other_args

    def forward(self, x: Tensor):
        return x

    def extra_repr(self) -> str:
        return f"dst_param_id={self.dst_param_id}, module_name={self.module_name}"
