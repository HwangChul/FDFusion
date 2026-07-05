# !/user/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import torch
import warnings
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from net.FDFusion import Net
warnings.filterwarnings('ignore')  # 不显示warnings
from ptflops import get_model_complexity_info
import thop
from fvcore.nn import FlopCountAnalysis


def multi_input_constructor(input_res):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    imageA = torch.randn(1, 1, 480, 640).to(device)
    imageB = torch.randn(1, 1, 480, 640).to(device)
    textA = torch.randn(1, 300, 960).to(device)
    textB = torch.randn(1, 300, 960).to(device)
    return {'imageA': imageA, 'imageB': imageB, 'textA': textA, 'textB': textB}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net(hidden_dim=256).to(device)

model.eval()
imageA = torch.randn(1, 1, 480, 640).to(device)
imageB = torch.randn(1, 1, 480, 640).to(device)
textA = torch.randn(1, 300, 960).to(device)
textB = torch.randn(1, 300, 960).to(device)



with torch.no_grad():
    flops = FlopCountAnalysis(
        model,
        (imageA, imageB, textA, textB)
    )
    total_flops = flops.total()

print("Total FLOPs:", total_flops / 1e9, "GFLOPs")

with torch.no_grad():
    thop_flops, thop_params = thop.profile(model, inputs=(
        imageA, imageB, textA, textB))
    thop_flops, thop_params = thop.clever_format([thop_flops, thop_params], '%.3f')

    print('thop模型参数：', thop_params)
    print('thop每一个样本浮点运算量：', thop_flops)
    ptflops_flops, ptflops_params = get_model_complexity_info(model, (1, 512, 640),
                                                              input_constructor=multi_input_constructor,
                                                              as_strings=True,
                                                              print_per_layer_stat=False)

    print('ptflops每一个样本浮点运算量：', ptflops_flops)
    print('ptflops模型参数：', ptflops_params)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(n_parameters)

