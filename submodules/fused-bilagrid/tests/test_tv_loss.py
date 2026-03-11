import sys
import os
sys.path += [os.path.dirname(os.path.abspath(__file__))]
from util import *

import torch

from fused_bilagrid_cuda import (
    tv_loss_forward,
    tv_loss_backward,
)


def _num_tensor_elems(t):
    return max(torch.prod(torch.tensor(t.size()[1:]).float()).item(), 1.0)


def tv_loss_torch(x):  # noqa: F811
    """Returns total variation on multi-dimensional tensors.

    Args:
        x (torch.Tensor): The input tensor with shape $(B, C, ...)$, where $B$ is the batch size and $C$ is the channel size.
    """
    batch_size = x.shape[0]
    tv = 0
    for i in range(2, len(x.shape)):
        n_res = x.shape[i]
        idx1 = torch.arange(1, n_res, device=x.device)
        idx2 = torch.arange(0, n_res - 1, device=x.device)
        x1 = x.index_select(i, idx1)
        x2 = x.index_select(i, idx2)
        count = _num_tensor_elems(x1)
        tv += torch.pow((x1 - x2), 2).sum() / count
    return tv / batch_size


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_tv_loss():

    print("# Test tv_loss")
    print()

    for (N, L, H, W) in [
        (3, 5, 7, 15),
        (26, 8, 16, 16)
    ]:
        print("(N, L, H, W) =", (N, L, H, W))

        torch.random.manual_seed(42)

        bilagrid = torch.randn((N, 12, L, H, W)).cuda()
        bilagrid = torch.nn.Parameter(bilagrid)

        output = tv_loss_torch(bilagrid)
        output.retain_grad()
        output.requires_grad_(True)

        output1 = tv_loss_forward(bilagrid)
        assert_close(output1, output, 1e-4, "tv_loss")

        weights = torch.randn_like(output)
        loss = (weights*output).mean()
        loss.backward()

        v_bilagrid = tv_loss_backward(bilagrid, output.grad)
        assert_close(v_bilagrid, bilagrid.grad, 1e-8, "bilagrid.grad")
        print()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def profile_tv_loss():

    L, H, W = 8, 16, 16

    print("# Profile tv_loss")
    print()

    for N in [1, 50, 100, 250, 600, 2000]:

        repeat = min(10000//N, 20)

        torch.random.manual_seed(42+N)

        print("N =", N)

        bilagrid = torch.randn((N, 12, L, H, W)).cuda()
        bilagrid = torch.nn.Parameter(bilagrid)

        timeits([
            (lambda: tv_loss_torch(bilagrid), "torch forward"),
            (lambda: tv_loss_forward(bilagrid), "fused forward"),
        ], repeat)

        bilagrid = torch.nn.Parameter(bilagrid)

        output = tv_loss_torch(bilagrid)
        output.retain_grad()
        output.requires_grad_(True)

        weight = 2.345
        loss = (weight*output).mean()

        timeits([
            (lambda: loss.backward(retain_graph=True), "torch backward"),
            (lambda: tv_loss_backward(bilagrid, output.grad), "fused backward"),
        ], repeat)
        print()


if __name__ == "__main__":

    test_tv_loss()
    print()

    profile_tv_loss()
