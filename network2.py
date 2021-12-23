import torch
import torch.nn as nn
import torch.nn.functional as nnf

class BlockConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0,
                 activation='relu', norm='none'):
        super(BlockConv2d, self).__init__()
        self.pad = nn.ZeroPad2d(padding)
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sig':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            self.activation = None
            
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            self.norm = None
            
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride)
        
    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        
        return x
        

class BlockConvTranspose2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0, out_padding=0,
                 activation='relu', norm='none'):
        super(BlockConvTranspose2d, self).__init__()
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            self.activation = None
            
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            self.norm = None
            
        self.conv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride,
                                       padding=padding, output_padding=out_padding)
        
    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        
        return x


        
class Unet(nn.Module):
    def __init__(self, mode_patch=True):
        super(Unet, self).__init__()
        self.mode_patch = mode_patch
        
        self.enc = nn.ModuleList()
        nb_channels_enc = [16, 32, 64, 128, 256]
        for i in range(len(nb_channels_enc)-1):
            self.enc.append(BlockConv2d(nb_channels_enc[i], nb_channels_enc[i+1], 4, 2, padding=1, activation='relu', norm='bn')) #208 -> 104 -> 52 -> 26 -> 13
        
        
        self.dec = nn.ModuleList()
        nb_channels_dec = [128, 64, 32, 16, 16, 16]
        self.dec.append(BlockConvTranspose2d(nb_channels_enc[-1], nb_channels_dec[0], 4, 2, 1, 0, 'relu', 'bn')) # 13 -> 26
        self.dec.append(BlockConvTranspose2d(nb_channels_dec[0], nb_channels_dec[1], 4, 2, 1, 0, 'relu', 'bn')) # 26 -> 52
        self.dec.append(BlockConvTranspose2d(nb_channels_dec[1], nb_channels_dec[2], 4, 2, 1, 0, 'relu', 'bn')) # 52 -> 104
        self.dec.append(BlockConvTranspose2d(nb_channels_dec[2], nb_channels_dec[3], 4, 2, 1, 0, 'relu', 'bn')) # 104 -> 208
        self.dec.append(BlockConv2d(nb_channels_dec[3], nb_channels_dec[4], 3, 1, 1, 'relu', 'bn'))
        
        
        self.conv0 = BlockConv2d(3, 16, 3, 1, 1, 'relu', 'bn')
        
        self.convs = nn.ModuleList()
        self.convs.append(BlockConv2d(256, 128, 3, 1, 1, 'relu', 'bn'))
        self.convs.append(BlockConv2d(128, 64, 3, 1, 1, 'relu', 'bn'))
        self.convs.append(BlockConv2d(64, 32, 3, 1, 1, 'relu', 'bn'))
        self.convs.append(BlockConv2d(32, 16, 3, 1, 1, 'relu', 'bn'))
        
        self.conv = BlockConv2d(16, 1, kernel_size=3, stride=1, padding=1, activation='sig')
        self.drop = nn.Dropout(p=0.2)
        
        if self.mode_patch:
            self.conv_patch = BlockConv2d(256, 64, kernel_size=3, stride=1, padding=1, activation='relu', norm='bn')
            self.conv_patch2 = BlockConv2d(64, 1, kernel_size=3, stride=1, padding=1, activation='sig')
    
    def forward(self, x):
        x = self.drop(self.conv0(x))
        x_enc = [x]
        for l in self.enc:
            x_enc.append(self.drop(l(x_enc[-1])))
            
        
        if self.mode_patch:
            y_patch = x_enc[-1]
            y_patch = self.drop(self.conv_patch(y_patch))
            y_patch = self.conv_patch2(y_patch)
        
        y = x_enc[-1]
        for i in range(4):
            y = self.drop(self.dec[i](y))
            y = torch.cat([y, x_enc[-(i+2)]], dim=1)
            y = self.convs[i](y)
            
        y = self.drop(self.dec[4](y))
        
        y = self.conv(y)
        
        if self.mode_patch:
            return y, y_patch
        
        return y