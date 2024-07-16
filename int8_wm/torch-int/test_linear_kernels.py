import torch
# from torch_int._CUDA import linear_a8_w8_b32_o32, linear_relu_a8_w8_b8_o8, linear_a8_w8_b8_o8, linear_a8_w8_b32_o32_with_scaling, linear_a8_w8_bfp32_ofp32
from icecream import ic
# from cublas_test._CUDA import linear_a8_w8_b32_o32_cublas, linear_a8_w8_b32_o32_with_scaling_cublas, linear_a8_w8_bfp32_ofp32_cublas, test

torch.ops.load_library('/home/hadoop-hmart-waimai-rank/int-linear/torch_int/ft_gemm/build/ftgemm.cpython-39-x86_64-linux-gnu.so')
gemm = torch.ops.ftgemm.FTGEMM()

# import ftgemm

@torch.no_grad()
def test_quant_linear_a8_w8_b32_o32():
    # B, M, N = 128, 512, 1024
    # B, M, N = 16, 16, 16
    B, M, N = 32, 32, 32
    # B, M, N = 4, 4, 4
    # B, M, N = 16, 32, 64
    # B, M, N = 64, 64, 64
    # weight = torch.randint(-128, 127, (N, M), dtype=torch.int8)
    weight = torch.randint(0, 2, (N, M), dtype=torch.int8)
    # weight = torch.ones((N, M), dtype=torch.int8)
    # bias = torch.randint(torch.iinfo(torch.int32).min, torch.iinfo(
    #     torch.int32).max, (N,), dtype=torch.int32)
    bias = torch.randint(torch.iinfo(torch.int8).min, torch.iinfo(
        torch.int8).max, (N,), dtype=torch.int8)
    # x = torch.randint(-128, 127, (B, M), dtype=torch.int8)
    x = torch.randint(0, 2, (B, M), dtype=torch.int8)
    # x = torch.ones((B, M), dtype=torch.int8)
    # linear = torch.nn.Linear(M, N, bias=True)
    linear = torch.nn.Linear(M, N, bias=False)
    linear.weight.data = weight.float()
    # linear.bias.data = bias.float()
    torch.cuda.synchronize()

    y_gt = linear(x.float())
    # torch.cuda.synchronize()
    # y = linear_a8_w8_b32_o32(x.cuda(), weight.cuda(), bias.cuda())
    torch.cuda.synchronize()
    # z = linear_a8_w8_b32_o32_cublas(x.cuda(), weight.cuda(), bias.cuda())

    z = test(x.cuda(), weight.cuda(), bias.cuda())
    torch.cuda.synchronize()
    print(y_gt)
    # print(y)
    print(z)
    # ic(torch.allclose(y_gt, y.float().cpu(), atol=1e-3))
    ic(torch.allclose(y_gt, z.float().cpu(), atol=1e-3))
    # print(y==z)


@torch.no_grad()
def test_quant_linear_a8_w8_b32_o32_with_scaling():
    # B, M, N = 128, 512, 1024
    B, M, N = 16, 16, 16
    weight = torch.randint(-128, 127, (N, M), dtype=torch.int8)
    bias = torch.randint(torch.iinfo(torch.int32).min, torch.iinfo(
        torch.int32).max, (N,), dtype=torch.int32)
    x = torch.randint(-128, 127, (B, M), dtype=torch.int8)
    alpha, beta = 0.01, 0.0001
    # alpha, beta = 0.0, 0.0
    linear = torch.nn.Linear(M, N, bias=True)
    linear.weight.data = weight.float() * alpha
    linear.bias.data = bias.float() * beta
    torch.cuda.synchronize()
    y_gt = linear(x.float())
    torch.cuda.synchronize()
    y = linear_a8_w8_b32_o32_with_scaling(
        x.cuda(), weight.cuda(), bias.cuda(), alpha, beta)
    torch.cuda.synchronize()
    z = linear_a8_w8_b32_o32_with_scaling_cublas(x.cuda(), weight.cuda(), bias.cuda(), alpha, beta)
    torch.cuda.synchronize()
    print(y)
    print(z)
    ic(torch.allclose(y_gt, y.float().cpu(), atol=0.5))
    ic(torch.allclose(y_gt, z.float().cpu(), atol=1.0))
    print(sum(y==z))


@torch.no_grad()
def test_quant_linear_a8_w8_bfp32_ofp32():
    B, M, N = 128, 512, 1024
    weight = torch.randint(-128, 127, (N, M), dtype=torch.int8)
    bias = torch.randn(N, dtype=torch.float32)
    x = torch.randint(-128, 127, (B, M), dtype=torch.int8)
    alpha, beta = 0.001, 1
    linear = torch.nn.Linear(M, N, bias=True)
    linear.weight.data = weight.float() * alpha
    linear.bias.data = bias.float() * beta
    torch.cuda.synchronize()
    y_gt = linear(x.float())
    torch.cuda.synchronize()
    y = linear_a8_w8_bfp32_ofp32(
        x.cuda(), weight.cuda(), bias.cuda(), alpha, beta)
    torch.cuda.synchronize()
    z = linear_a8_w8_bfp32_ofp32_cublas(x.cuda(), weight.cuda(), bias.cuda(), alpha, beta);
    torch.cuda.synchronize()
    print(y.shape)
    print(y)
    print(z)
    ic(torch.allclose(y_gt, y.cpu(), atol=0.5))
    ic(torch.allclose(y_gt, z.cpu(), atol=0.5))
    


@torch.no_grad()
def test_quant_linear_a8_w8_b8_o8():
    B, M, N = 128, 512, 1024
    weight = torch.randint(-128, 127, (N, M), dtype=torch.int8)
    bias = torch.randint(-128, 127, (N,), dtype=torch.int8)
    x = torch.randint(-128, 127, (B, M), dtype=torch.int8)
    alpha, beta = 0.001, 0.01
    linear = torch.nn.Linear(M, N, bias=True)
    linear.weight.data = weight.float() * alpha
    linear.bias.data = bias.float() * beta
    y_gt = linear(x.float()).clamp(-128, 127).round().long()
    y = linear_a8_w8_b8_o8(x.cuda(), weight.cuda(),
                           bias.cuda(), alpha, beta).cpu().long()
    ic(torch.allclose(y_gt.float(), y.float().cpu(), atol=1))


@torch.no_grad()
def test_quant_linear_relu_a8_w8_b8_o8():
    B, M, N = 128, 512, 1024
    weight = torch.randint(-128, 127, (N, M), dtype=torch.int8)
    bias = torch.randint(-128, 127, (N,), dtype=torch.int8)
    x = torch.randint(-128, 127, (B, M), dtype=torch.int8)
    alpha, beta = 0.001, 0.01
    linear = torch.nn.Linear(M, N, bias=True)
    linear.weight.data = weight.float() * alpha
    linear.bias.data = bias.float() * beta
    y_gt = linear(x.float())
    y_gt = y_gt.clamp(0, 127).round().long()
    y = linear_relu_a8_w8_b8_o8(x.cuda(), weight.cuda(),
                                bias.cuda(), alpha, beta).cpu().long()
    ic(torch.allclose(y_gt.float(), y.float().cpu(), atol=1))


# if __name__ == '__main__':
    # print('test_quant_linear_a8_w8_b32_o32')
    # test_quant_linear_a8_w8_b32_o32()
    # print('test_quant_linear_a8_w8_b32_o32_with_scaling')
    # test_quant_linear_a8_w8_b32_o32_with_scaling()
    # print('test_quant_linear_a8_w8_bfp32_ofp32')
    # test_quant_linear_a8_w8_bfp32_ofp32()
    # print('test_quant_linear_a8_w8_b8_o8')
    # test_quant_linear_a8_w8_b8_o8()
    # print('test_quant_linear_relu_a8_w8_b8_o8')
    # test_quant_linear_relu_a8_w8_b8_o8()
