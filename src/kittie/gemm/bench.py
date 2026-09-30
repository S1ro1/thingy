import functools
import importlib.util
import os
import subprocess
import sys
import sysconfig
from argparse import ArgumentParser
from pathlib import Path

import cuda.bindings.driver as cuda_driver
import torch
from cutlass.testing import benchmark

from cutie.gemm.bench import (
    gemm_flops,
    gemm_reference,
    test_configs,
    time_us_to_tflops,
    workspace_generator,
)

HERE = Path(__file__).resolve().parent


def load_extension():
    subprocess.run(['make', '-s', '-C', str(HERE), f'PYTHON={sys.executable}'], check=True)
    path = HERE / ('_C' + sysconfig.get_config_var('EXT_SUFFIX'))
    spec = importlib.util.spec_from_file_location('_C', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = ArgumentParser(description='Run GEMM benchmark')
    parser.add_argument('--M', type=int, help='Size for M dimension')
    parser.add_argument('--N', type=int, help='Size for N dimension')
    parser.add_argument('--K', type=int, help='Size for K dimension')
    args = parser.parse_args()
    dims = (args.M, args.N, args.K)
    if any(d is not None for d in dims) and not all(d is not None and d > 0 for d in dims):
        parser.error('provide positive --M, --N, and --K together')
    return args


def main():
    args = parse_args()
    debug = os.environ.get('DEBUG', '0') == '1'
    extension = load_extension()
    torch_stream = torch.cuda.current_stream()
    stream = cuda_driver.CUstream(torch_stream.cuda_stream)

    def kittie_gemm(A, B, out):
        extension.gemm(A, B, out, stream=torch_stream)

    if args.M and args.N and args.K:
        runnable = [(args.M, args.N, args.K)]
    else:
        runnable = test_configs

    for M, N, K in runnable:
        inputs = workspace_generator(M, N, K)
        flops = gemm_flops(M, N, K)
        baseline_result = gemm_reference(
            inputs.kwargs['A'], inputs.kwargs['B'],
            out=torch.empty_like(inputs.kwargs['out']),
        )
        workspace = functools.partial(workspace_generator, M=M, N=N, K=K)
        if not debug:
            time_us_reference = benchmark(
                gemm_reference, workspace_generator=workspace, stream=stream,
            )
            tflops_reference = time_us_to_tflops(time_us_reference, flops)

        if M % extension.BM or N % extension.BN or K % extension.BK:
            raise ValueError('M, N, K must be divisible by BM, BN, BK')
        inputs.kwargs['out'].fill_(float('nan'))
        kittie_gemm(**inputs.kwargs)
        if debug:
            torch_stream.synchronize()  # Flush device printf after the single launch.
        is_correct = torch.allclose(
            baseline_result, inputs.kwargs['out'], rtol=1e-2, atol=1e-2,
        )
        if not is_correct:
            raise AssertionError('Kernel output does not match PyTorch')

        if debug:
            print(f'gemm({M}, {N}, {K}) | ✅ | DEBUG: single launch, timing skipped')
            break

        time_us_kittie = benchmark(
            kittie_gemm, workspace_generator=workspace, stream=stream,
        )
        tflops_kittie = time_us_to_tflops(time_us_kittie, flops)
        print(
            f'gemm({M}, {N}, {K}) | ✅ | Torch: {tflops_reference:.2f} TFLOPS | '
            f'Kittie: {tflops_kittie:.2f} TFLOPS | Speedup: {(tflops_kittie / tflops_reference):.2f}X'
        )


if __name__ == '__main__':
    main()
