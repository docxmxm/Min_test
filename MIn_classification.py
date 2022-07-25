import numpy as np
import torch.nn as nn
import MinkowskiEngine as ME


class ExampleNetwork(ME.MinkowskiNetwork):
    def __init__(self, in_feat, out_feat, D=3):
        super(ExampleNetwork, self).__init__(D)
        self.conv = nn.Sequential(
                ME.MinkowskiConvolution(
                    in_channels=in_feat,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    dilation=1,
                    bias=False,
                    dimension=D),
                ME.MinkowskiBatchNorm(64),
                ME.MinkowskiReLU())
        self.pooling = ME.MinkowskiGlobalPooling(ME.PoolingMode.GLOBAL_AVG_POOLING_KERNEL)
        self.linear = ME.MinkowskiLinear(64, out_feat)

    def forward(self, x):
        out = self.conv(x)
        print('conv: ', out.coordinates.size(), out.features.size())
        out = self.pooling(out)
        print('pooling: ', out.coordinates.size(), out.features.size())
        out = self.linear(out)
        print('linear: ', out.coordinates.size(), out.features.size())
        return out


if __name__ == '__main__':
    origin_pc1 = 100 * np.random.uniform(0, 1, (10, 3))
    feat1 = np.ones((10, 3), dtype=np.float32)
    origin_pc2 = 100 * np.random.uniform(0, 1, (6, 3))
    feat2 = np.ones((6, 3), dtype=np.float32)

    coords, feats = ME.utils.sparse_collate([origin_pc1, origin_pc2], [feat1, feat2])
    input = ME.SparseTensor(feats, coordinates=coords)

    net = ExampleNetwork(in_feat=3, out_feat=32)
    output = net(input)
    print(net)
    #check the parameters of NN
    for k, v in net.named_parameters():
        #print(v.size())
        print(k, v.size())
