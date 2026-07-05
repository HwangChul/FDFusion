import torch
import torch.nn as nn
from net.restormer import TransformerBlock as Restormer
import torch.nn.functional as F
from net.D_RAMiT import DRAMiTransformer
from einops import rearrange


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, query, key, value):
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        attn_output, _ = self.multihead_attn(query, key, value)
        attn_output = attn_output.transpose(0, 1)

        return attn_output


class imagefeature2textfeature(nn.Module):
    def __init__(self, in_channel, mid_channel, hidden_dim):
        super(imagefeature2textfeature, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=mid_channel, kernel_size=1)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(x, [288, 384], mode='nearest')
        x = x.contiguous().view(x.size(0), x.size().numel() // x.size(0) // self.hidden_dim, self.hidden_dim)       # reshape
        return x


class text_preprocess(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super(text_preprocess, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, mid_channel, 1, 1, 0)
        self.conv2 = nn.Conv1d(mid_channel, out_channel, 1, 1, 0)
        self.conv3 = nn.Conv1d(in_channel, out_channel, 1, 1, 0)

    def forward(self, x):
        # x = self.conv1(x.permute(0, 2, 1))
        # x = self.conv2(x)
        x = self.conv3(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels=16, out_channels=16):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels*2, out_channels, kernel_size=3, stride=1, padding=1)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(in_channels*3, out_channels, kernel_size=3, stride=1, padding=1)
        self.prelu3 = nn.PReLU()

    def forward(self, x):
        out = self.prelu1(self.conv1(x))

        out1 = self.conv2(torch.cat((x, out), 1))
        out1 = self.prelu2(out1)

        out2 = self.conv3(torch.cat((x, out, out1), 1))
        out2 = self.prelu3(out2)

        out = out + out1 + out2 + x
        return out


class JCR(nn.Module):
    def __init__(
            self,
            input_channel=1,
            conv_dim=32,
            res_dim=32,
            restormerhead=8,
            image2text_dim=32,
            ffn_expansion_factor=4,
            bias=False,
            LayerNorm_type='WithBias',
            hidden_dim=256,
    ):
        super().__init__()
        self.conv_dim = conv_dim
        self.res_dim = res_dim
        self.image2text_dim = image2text_dim
        self.dim = self.conv_dim + self.res_dim
        self.dropout = nn.Dropout2d(p=0.2)

        self.convA1_1 = nn.Conv2d(input_channel, self.dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.preluA1 = nn.PReLU()
        self.convA1_2 = nn.Conv2d(self.dim, self.conv_dim, kernel_size=1, stride=1, padding=0)
        self.denseA = DenseBlock(self.conv_dim, self.conv_dim)
        self.restormerA = Restormer(self.res_dim, restormerhead, ffn_expansion_factor, bias, LayerNorm_type)
        self.ramitA = DRAMiTransformer(dim=32, num_head=8, chsa_head_ratio=0.5)

        self.convB1_1 = nn.Conv2d(input_channel, self.dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.preluB1 = nn.PReLU()
        self.convB1_2 = nn.Conv2d(self.dim, self.conv_dim, kernel_size=1, stride=1, padding=0)
        self.denseB = DenseBlock(self.conv_dim, self.conv_dim)
        self.restormerB = Restormer(self.res_dim, restormerhead, ffn_expansion_factor, bias, LayerNorm_type)
        self.ramitB = DRAMiTransformer(dim=32, num_head=8, chsa_head_ratio=0.5)

        self.imageA2text_feature = imagefeature2textfeature(self.conv_dim, image2text_dim, hidden_dim)
        self.imageB2text_feature = imagefeature2textfeature(self.conv_dim, image2text_dim, hidden_dim)
        self.cross_attentionA = CrossAttention(embed_dim=hidden_dim, num_heads=8)
        self.cross_attentionB = CrossAttention(embed_dim=hidden_dim, num_heads=8)

        self.convA2 = nn.Conv2d(image2text_dim, self.conv_dim, kernel_size=1)
        self.preluA2 = nn.PReLU()
        self.convA3 = nn.Conv2d(self.dim, self.conv_dim, kernel_size=1)
        self.preluA3 = nn.PReLU()

        self.convB2 = nn.Conv2d(image2text_dim, self.conv_dim, kernel_size=1)
        self.preluB2 = nn.PReLU()
        self.convB3 = nn.Conv2d(self.dim, self.conv_dim, kernel_size=1)
        self.preluB3 = nn.PReLU()


    def forward(self, imageA, imageB, textA, textB):
        b, _, H, W = imageA.shape

        feaA = self.preluA1(self.convA1_1(imageA))
        convA, resA = torch.split(feaA, (self.conv_dim, self.res_dim), dim=1)           # b, c, H, W   1, 32, 288, 384  b, 10, him
        convA = self.denseA(convA)
        resA, sp, ch, attn0 = self.ramitA(resA)
        convAtotext = self.imageA2text_feature(convA)
        resAtotext = self.imageA2text_feature(resA)
        imageA = self.convA1_2(feaA)
        imageAtotext = self.imageA2text_feature(imageA)

        feaB = self.preluB1(self.convB1_1(imageB))
        convB, resB = torch.split(feaB, (self.conv_dim, self.res_dim), dim=1)
        convB = self.denseB(convB)
        resB, sp, ch, attn0 = self.ramitA(resB)
        convBtotext = self.imageB2text_feature(convB)
        resBtotext = self.imageB2text_feature(resB)
        imageB = self.convB1_2(feaB)
        imageBtotext = self.imageB2text_feature(imageB)


        ca_A = self.cross_attentionA(textA, resAtotext, convAtotext)
        ca_A = torch.nn.functional.adaptive_avg_pool1d(ca_A.permute(0, 2, 1), output_size=1).permute(0, 2, 1)           # 300,  256
        ca_A = F.normalize(ca_A, p=1, dim=2)
        ca_A = (imageAtotext * ca_A).view(imageA.shape[0], self.image2text_dim, 288, 384)
        ca_A = F.interpolate(ca_A, [H, W], mode='nearest')

        ca_A = self.preluA3(self.convA3(
            torch.cat((imageA, self.preluA2(self.convA2(ca_A) + imageA)), 1)))      # concat


        ca_B = self.cross_attentionB(textB, resBtotext, convBtotext)
        ca_B = torch.nn.functional.adaptive_avg_pool1d(ca_B.permute(0, 2, 1), 1).permute(0, 2, 1)
        ca_B = F.normalize(ca_B, p=1, dim=2)
        ca_B = (imageBtotext * ca_B).view(imageB.shape[0], self.image2text_dim, 288, 384)
        ca_B = F.interpolate(ca_B, [H, W], mode='nearest')

        ca_B = self.preluB3(
            self.convB3(torch.cat(
                (imageB, self.preluB2(self.convB2(ca_B)) + imageB), 1)))

        return ca_A, ca_B


class CLMA(nn.Module):
    def __init__(self, in_channels=32, num_heads=8, reduction_ratio=4):
        super(CLMA, self).__init__()
        self.num_heads = num_heads
        self.reduction_ratio = reduction_ratio
        self.hidden_dim = in_channels // reduction_ratio

        self.q_proj = nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1)
        self.k_proj = nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1)

        self.out_proj = nn.Conv2d(self.hidden_dim, in_channels, kernel_size=1)

        self.prelu = nn.PReLU()
        self.conv = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1)

    def forward(self, current, previous):
        B, C, H, W = current.shape

        Q = self.q_proj(current)
        K = self.k_proj(previous)
        V = self.v_proj(previous)

        Q = Q.reshape(B, self.num_heads, self.hidden_dim // self.num_heads, H * W)
        K = K.reshape(B, self.num_heads, self.hidden_dim // self.num_heads, H * W)
        V = V.reshape(B, self.num_heads, self.hidden_dim // self.num_heads, H * W)

        attn_out = F.scaled_dot_product_attention(Q, K, V)

        attn_out = attn_out.reshape(B, self.hidden_dim, H, W)

        attn_out = F.normalize(attn_out, p=2, dim=2)
        shortcut = self.out_proj(attn_out)
        # out = self.conv(torch.cat((current, shortcut), dim=1))
        out = shortcut + current

        return out, shortcut


class Net(nn.Module):
    def __init__(
            self,
            mid_channel=32,
            decoder_num_heads=8,
            ffn_factor=4,
            bias=False,
            LayerNorm_type='WithBias',
            out_channel=1,
            hidden_dim=256,
            image2text_dim=32,
            pooling='avg',
            normalization='l1'
    ):
        super().__init__()
        self.textA = text_preprocess(in_channel=960, mid_channel=512, out_channel=hidden_dim)
        self.textB = text_preprocess(in_channel=960, mid_channel=512, out_channel=hidden_dim)

        self.restormer1 = Restormer(2 * mid_channel, decoder_num_heads, ffn_factor, bias, LayerNorm_type)
        self.restormer2 = Restormer(mid_channel, decoder_num_heads, ffn_factor, bias, LayerNorm_type)
        self.restormer3 = Restormer(mid_channel, decoder_num_heads, ffn_factor, bias, LayerNorm_type)
        self.conv1 = nn.Conv2d(2 * mid_channel, mid_channel, kernel_size=1)
        self.conv2 = nn.Conv2d(mid_channel, out_channel, kernel_size=1)
        self.softmax = nn.Sigmoid()
        self.jcr1 = JCR()
        self.jcr2 = JCR(input_channel=mid_channel)
        self.jcr3 = JCR(input_channel=mid_channel)

        self.clma = CLMA()

    def forward(self, imageA, imageB, textA, textB):

        textA = self.textA(textA)
        textB = self.textB(textB)

        featureA, featureB = self.jcr1(imageA, imageB, textA, textB)
        featureA, featureB = self.jcr2(featureA, featureB, textA, textB)
        featureA, featureB = self.jcr3(featureA, featureB, textA, textB)


        fusionfeature = torch.cat((featureA, featureB),1)
        fusionfeature1 = self.restormer1(fusionfeature)
        fusionfeature1 = self.conv1(fusionfeature1)

        fusionfeature2 = self.restormer2(fusionfeature1)

        fusionfeature2, _ = self.clma(fusionfeature2, fusionfeature1)

        fusionfeature3 = self.restormer3(fusionfeature2)
        fusionfeature3 = self.conv2(fusionfeature3)

        fusionfeature = self.softmax(fusionfeature3)
        return fusionfeature


if __name__ == '__main__':
    imageA = torch.randn(1, 1, 288, 384)
    imageB = torch.randn(1, 1, 288, 384)
    textA = torch.randn(1, 300, 960)
    textB = torch.randn(1, 300, 960)
    model = Net(hidden_dim=256, image2text_dim=32)
    fusionfeature = model(imageA, imageB, textA, textB)
    print(fusionfeature.shape)
