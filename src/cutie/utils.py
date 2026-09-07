import cutlass.cute as cute
import cutlass.torch as ctorch
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.testing import JitArguments

PEAK_MEM_BW = 8e12  # 8TB/s


def get_transferred_bytes(M: int, H: int) -> int:
    total_bytes = 0
    # we load M * H elements from gmem
    total_bytes += M * H * 4
    # we load H elements from gmem
    total_bytes += H * 4
    # we store the same amount of elements to gmem
    total_bytes += M * H * 4

    return total_bytes


def get_mem_util(time_us: float, M: int, H: int) -> float:
    time_s = time_us / 1e6
    transferred_bytes = get_transferred_bytes(M, H)

    return transferred_bytes / (PEAK_MEM_BW * time_s) * 100


def workspace_to_cute(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor
) -> tuple[cute.Tensor, cute.Tensor, cute.Tensor]:
    x = from_dlpack(x, assumed_align=32)
    weight = from_dlpack(weight, assumed_align=32)
    out = from_dlpack(out, assumed_align=32)
    return x, weight, out


def workspace_generator(
    M: int,
    H: int,
    to_cute: bool = False,
    torch_stream: torch.cuda.Stream | None = None,
) -> JitArguments:
    def create_workspace() -> JitArguments:
        x = torch.randn(M, H, dtype=torch.float32, device="cuda")
        weight = torch.randn(H, dtype=torch.float32, device="cuda")
        eps = 1e-5
        out = torch.empty(M, H, dtype=torch.float32, device="cuda")
        if to_cute:
            x_cute, weight_cute, out_cute = workspace_to_cute(x, weight, out)
            return JitArguments(x=x_cute, weight=weight_cute, eps=eps, out=out_cute)

        return JitArguments(x=x, weight=weight, eps=eps, out=out)

    if torch_stream is None:
        return create_workspace()

    with torch.cuda.stream(torch_stream):
        return create_workspace()
