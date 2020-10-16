import torch.nn as nn

"""
Encoder Decoder Architecture
"""

class DownConv(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownConv, self).__init__()
        """
        Down convolution block, resembling Unet Architecture
        """
        self.downblock = nn.Sequential()