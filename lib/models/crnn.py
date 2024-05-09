import torch.nn as nn
from torchsummary import summary

# LSTM类
class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)     # nIn：输入神经元个数
        self.embedding = nn.Linear(nHidden * 2, nOut)  # *2因为使用双向LSTM，两个方向隐层单元拼在一起

    def forward(self, input):
        # 经过RNN输出feature map特征结果
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()  # T:时间步长，b:batch size,h:hiden unit
        t_rec = recurrent.view(T * b, h)# 512×batch_size, 256

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        """
        :param imgH: 图片高度
        :param nc: 输入图片通道数
        :param nclass: 分类数目  cls_num+1需要增加一个空白符
        :param nh: rnn隐藏层神经元节点数
        :param n_rnn: rnn的层数
        :param leakyRelu: 是否使用LeakyRelu
        """
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16 图片高度必须为16的倍数'

        ks = [3, 3, 3, 3, 3, 3, 2]  # 卷积层卷积尺寸3表示3x3，2表示2x2
        ps = [1, 1, 1, 1, 1, 1, 0]  # padding大小
        ss = [1, 1, 1, 1, 1, 1, 1]  # stride大小
        nm = [64, 128, 256, 256, 512, 512, 512]  # 卷积核个数,卷积操作输出特征层的通道数

        cnn = nn.Sequential()
        def convRelu(i, batchNormalization=False):  # 创建卷积层
            nIn = nc if i == 0 else nm[i - 1]  # 确定输入channel维度,如果是第一层网络，输入通道数为图片通道数，输入特征层的通道数为上一个特征层的输出通道数
            nOut = nm[i]  # 确定输出channel维度
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))  # 添加卷积层
            # BN层
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            # Relu激活层
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))
        # 100×32×64
        convRelu(0)
        # 50×16×64
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))
        # 50×16×128
        convRelu(1)
        # 25×8×128
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))
        # 25×8×256
        convRelu(2, True)
        # 25×8×256
        convRelu(3)
        # 26×4×256
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))    # 参数 (h, w)
        # 26×4×512
        convRelu(4, True)
        convRelu(5)
        # 27×2×512
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        # 26×1×512
        convRelu(6, True)

        self.cnn = cnn
        # print(self.cnn)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh), # 输入的时间步长为512
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        # w作为时间步长
        conv = conv.permute(2, 0, 1)  # [w, b, c] [26, b, 512]

        # rnn features
        output = self.rnn(conv)
        # print(output.size())

        return output


if __name__ == '__main__':

    crnn = CRNN(32, 1, 37, 256)
    print(crnn)
