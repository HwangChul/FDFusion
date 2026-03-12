from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# import kornia


def gaussian(window_size, sigma):
    """
    生成一维高斯分布的权重窗口。
    参数:
        window_size (int): 窗口的大小（即高斯分布的长度）。
        sigma (float): 高斯分布的标准差，控制分布的宽度。
    返回:
        torch.Tensor: 正规化的一维高斯分布权重张量，其元素之和为1。
    """
    # 使用列表推导式生成高斯分布权重
    gauss = torch.Tensor([
        exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()  # 对权重进行归一化，使得总和为1


def create_window(window_size, channel):  # 计算SSIM需要用到
    # 生成一维高斯分布权重
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)  # (5,)->(5,1)

    # 通过矩阵乘法生成二维高斯分布
    # (5,1)*(1,5)=(5,5)-->(1,1,5,5)
    _2D_window = _1D_window.mm(_1D_window.t()).double().unsqueeze(0).unsqueeze(0)

    # 将二维窗口扩展为适配多通道的卷积核形状（复制到多通道）
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())

    return window


class BCE_loss():  # Binary Cross-Entropy
    def __init__(self):
        super().__init__()

    def cal(self, predictlabel, truelabel):
        validindex = torch.where(torch.sum(truelabel, axis=2) == 1)
        # 对第二个维度求和，将第二维度的数值加到一起，求和后第二个维度消失，为什么取truelabel第二维求和为1，可能用于检查是否有某个类别的标注存在
        criteria = nn.BCELoss()
        loss = criteria(predictlabel[validindex[0], validindex[1], :, validindex[2]],
                        truelabel[validindex[0], validindex[1], :, validindex[2]])
        # 取出1所在的第二个维度
        return loss


class MSE_loss():
    def __init__(self):
        super().__init__()

    def cal(self, predictlabel, truelabel):
        validindex = torch.where(torch.sum(truelabel, axis=2) == 1)  # 人工标注的通道
        valid_predictlabel = predictlabel[validindex[0], validindex[1], :, validindex[2]]
        valid_truelabel = truelabel[validindex[0], validindex[1], :, validindex[2]]
        label_index = torch.argmax(valid_truelabel, dim=1) / valid_truelabel.shape[1]  # 把dim这个维度的值，变成这个维度的最大值的index
        predict_index = torch.argmax(valid_predictlabel, dim=1) / valid_truelabel.shape[1]
        criteria = nn.MSELoss()
        loss = criteria(label_index, predict_index)
        return loss


class BCE_MSE_loss():
    def __init__(self, balan_para):
        super().__init__()
        self.bp = balan_para
        self.BCEloss = BCE_loss()
        self.MSEloss = MSE_loss()

    def cal(self, predictlabel, truelabel):
        loss = self.BCEloss.cal(predictlabel, truelabel) + self.bp * self.MSEloss.cal(predictlabel, truelabel)
        return loss


def CE_Loss(inputs, target, num_classes=0):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss = nn.CrossEntropyLoss(ignore_index=num_classes)(temp_inputs, temp_target)  # 1.创建实例；2.计算CE
    return CE_loss


class Sobelxy(nn.Module):
    def __init__(self, device='cuda'):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        # shape: (3, 3)->(1,1,3,3), (out_channels, in_channels, kernel_height, kernel_width)
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).to(device)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).to(device)

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)  # 取绝对值,合并 X 和 Y 方向梯度


class Fusionloss(nn.Module):
    def __init__(self, coeff_int=1, coeff_grad=10, in_max=True, device='cuda'):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy(device=device)
        self.coeff_int = coeff_int  # 像素级损失的权重系数（用于调整强度）
        self.coeff_grad = coeff_grad  # 梯度损失的权重系数（用于调整对边缘的关注度）
        self.in_max = in_max

    def forward(self, image_vis, image_ir, generate_img):
        image_y = image_vis[:, :1, :, :]  # 只取可见光图像的第 1 个通道（亮度通道）, 第二个维度取第0个

        if self.in_max:
            x_in_max = torch.max(image_y, image_ir)
        else:
            x_in_max = (image_y + image_ir) / 2.0
        loss_in = F.l1_loss(x_in_max, generate_img)

        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        generate_img_grad = self.sobelconv(generate_img)

        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)
        loss_total = self.coeff_int * loss_in + self.coeff_grad * loss_grad

        return loss_total, loss_in, loss_grad


def _mef_ssim(X, Ys, window, ws, denom_g, denom_l, C1, C2, is_lum=False, full=False):  # 多曝光融合
    """
    X：输入图像（模型的预测输出）。
    Ys：参考图像（通常是目标图像，或高质量图像）。
    window：用于计算均值和方差的窗口（卷积核），通常是一个高斯窗口。
    ws：窗口大小。
    denom_g、denom_l：亮度和对比度相关的常数（控制亮度和对比度的加权程度）。
    C1、C2：常数项，用于数值稳定性，避免分母为零。
    is_lum：一个布尔值，决定是否使用亮度（Luminance）信息。如果为 True，则计算亮度映射；否则不使用亮度信息。
    full：一个布尔值，决定是否返回完整的输出，包括亮度图和对比度图。如果为 False，只返回最终的质量值 q。

    SSIM: mu:亮度（Luminance）, 对比度（Contrast）, sigma: 结构 (Structure)

    """
    K, C, H, W = list(Ys.size())

    # compute statistics of the reference latent image Y
    muY_seq = F.conv2d(Ys, window, padding=ws // 2, groups=C).view(K, C, H, W)
    # 计算参考图像 Ys 的均值(亮度)（通过卷积操作与窗口进行平滑）。
    muY_sq_seq = muY_seq * muY_seq
    # 计算局部方差；方差计算公式中的括号展开；Ys * Ys = 逐元素相乘
    sigmaY_sq_seq = F.conv2d(Ys * Ys, window, padding=ws // 2, groups=C).view(K, C, H, W) \
                    - muY_sq_seq
    sigmaY_sq, patch_index = torch.max(sigmaY_sq_seq, dim=0)
    """
    方差最大值、索引  减少维度：in:(K, C, H, W)->out:(C, H, W)
    patch_index = torch.tensor([
        [2, 1],  # 在(0,0)位置, K=2的值最大，在 (0,1) 位置 选K=1
        [0, 2]   # 在 (1,0) 位置 选K=0，在 (1,1) 位置 选K=2
    ])
    """

    # compute statistics of the test image X    亮度
    muX = F.conv2d(X, window, padding=ws // 2, groups=C).view(C, H, W)
    muX_sq = muX * muX
    sigmaX_sq = F.conv2d(X * X, window, padding=ws // 2,
                         groups=C).view(C, H, W) - muX_sq

    # compute correlation term   协方差：E[(X−μX)(Y−μY)] = E[XY]−E[X]E[Y]
    sigmaXY = F.conv2d(X.expand_as(Ys) * Ys, window, padding=ws // 2, groups=C).view(K, C, H, W) \
              - muX.expand_as(muY_seq) * muY_seq

    # compute quality map     对比度
    cs_seq = (2 * sigmaXY + C2) / (sigmaX_sq + sigmaY_sq_seq + C2)  # (K, C, H, W)
    cs_map = torch.gather(input=cs_seq.view(K, -1), dim=0,
                          index=patch_index.view(1, -1)).view(C, H, W)  # 从 input 的 dim 维度上，按照 index 选取数据
    """
    input: (K, C*H*W), index: (1, C*H*W); 在 dim=0 这个维度取哪一个K, 选择最大方差对应的 cs_seq 值
    cs_map: (1, C*H*W).view(C, H, W)->(C, H, W)
    cs_seq 记录了多个 K 参考图像的 contrast-structure 相似度
    patch_index 选择了方差最大的参考图像 K; torch.gather() 取出这个K处的 cs_seq
    """
    if is_lum:
        lY = torch.mean(muY_seq.view(K, -1), dim=1)  # 全局亮度均值 shape: (k, )
        lL = torch.exp(-((muY_seq - 0.5) ** 2) / denom_l)  # 局部亮度的加权因子 shape: (K, C, H, W)
        lG = torch.exp(- ((lY - 0.5) ** 2) /
                       denom_g)[:, None, None].expand_as(lL)  # 全局亮度的影响因子 shape: (k, )->(k, 1, 1)->(K, C, H, W)
        LY = lG * lL  # 亮度加权因子 shape: (K, C, H, W)
        muY = torch.sum((LY * muY_seq), dim=0) / torch.sum(LY, dim=0)  # 加权后的亮度均值，shape: (C, H, W)，用于最终的亮度计算。
        muY_sq = muY * muY
        l_map = (2 * muX * muY + C1) / (muX_sq + muY_sq + C1)  # 亮度对比函数
    else:
        l_map = torch.Tensor([1.0])
        if Ys.is_cuda:
            l_map = l_map.cuda(Ys.get_device())

    if full:
        l = torch.mean(l_map)
        cs = torch.mean(cs_map)
        return l, cs

    qmap = l_map * cs_map  # 亮度图与对比度图的乘积，代表图像的综合质量。
    q = qmap.mean()

    return q


class MEFSSIM(torch.nn.Module):
    def __init__(self, window_size=11, channel=3, sigma_g=0.2, sigma_l=0.2, c1=0.01, c2=0.03, is_lum=False):
        super(MEFSSIM, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = create_window(window_size, self.channel)
        self.denom_g = 2 * sigma_g ** 2
        self.denom_l = 2 * sigma_l ** 2
        self.C1 = c1 ** 2
        self.C2 = c2 ** 2
        self.is_lum = is_lum

    def forward(self, X, Ys):
        (_, channel, _, _) = Ys.size()

        if channel == self.channel and self.window.data.type() == Ys.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if Ys.is_cuda:
                window = window.cuda(Ys.get_device())
            window = window.type_as(Ys)

            self.window = window
            self.channel = channel

        return _mef_ssim(X, Ys, window, self.window_size,
                         self.denom_g, self.denom_l, self.C1, self.C2, self.is_lum)


class LpLssimLossweight(nn.Module):
    def __init__(self, window_size=5, size_average=True):
        """
            Constructor
        """
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        """
            Get the gaussian kernel which will be used in SSIM computation
        """
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        """
            Create the gaussian window
        """
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)  # [window_size, 1]
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)  # [1,1,window_size, window_size]
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        """
            Compute the SSIM for the given two image
            The original source is here: https://stackoverflow.com/questions/39051451/ssim-ms-ssim-for-tensorflow
        """
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)      # 局部均值
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq        # 局部方差
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2         # 协方差：E[(X−μX)(Y−μY)] = E[XY]−E[X]E[Y]

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)     # 依次对C, H, W取平均；即每张图片（batch 中的每个样本）单独计算一个 SSIM 值。

    def forward(self, image_in, image_out, weight):

        # Check if need to create the gaussian window
        (_, channel, _, _) = image_in.size()
        if channel == self.channel and self.window.data.type() == image_in.data.type():
            pass
        else:
            window = self.create_window(self.window_size, channel)
            window = window.to(image_out.get_device())
            window = window.type_as(image_in)
            self.window = window
            self.channel = channel

        # Lp
        Lp = torch.sqrt(torch.sum(torch.pow((image_in - image_out), 2)))  # 二范数
        # Lp = torch.sum(torch.abs(image_in - image_out))  # 一范数
        # Lssim
        Lssim = 1 - self._ssim(image_in, image_out, self.window, self.window_size, self.channel, self.size_average)
        return Lp + Lssim * weight, Lp, Lssim * weight
