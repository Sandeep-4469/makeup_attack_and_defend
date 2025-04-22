import torch
import torch.nn as nn
import functools

class Identity(nn.Module):
    def forward(self, x): return x

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch': norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance': norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none': norm_layer = lambda x: Identity()
    else: raise NotImplementedError(f'normalization layer [{norm_type}] is not found')
    return norm_layer

def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal': nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier': nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming': nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal': nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else: raise NotImplementedError(f'initialization method [{init_type}] is not implemented')
            if hasattr(m, 'bias') and m.bias is not None: nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain); nn.init.constant_(m.bias.data, 0.0)
    print(f'Initializing network with {init_type}')
    net.apply(init_func); return net

# --- Residual Block ---
class ResidualBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super(ResidualBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect': conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate': conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero': p = 1
        else: raise NotImplementedError(f'padding [{padding_type}] is not implemented')
        use_bias = norm_layer == nn.InstanceNorm2d if not isinstance(norm_layer, functools.partial) else norm_layer.func == nn.InstanceNorm2d

        conv_block += [nn.Conv2d(dim, dim, 3, 1, p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout: conv_block += [nn.Dropout(0.5)]
        p = 0
        if padding_type == 'reflect': conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate': conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero': p = 1
        else: raise NotImplementedError(f'padding [{padding_type}] is not implemented')
        conv_block += [nn.Conv2d(dim, dim, 3, 1, p, bias=use_bias), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x): return x + self.conv_block(x)

# --- Generator (ResNet-based) ---
class GeneratorResNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):
        assert(n_blocks >= 0); super(GeneratorResNet, self).__init__()
        if type(norm_layer) == functools.partial: use_bias = norm_layer.func == nn.InstanceNorm2d
        else: use_bias = norm_layer == nn.InstanceNorm2d
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, 7, 1, 0, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling): mult = 2**i; model += [nn.Conv2d(ngf*mult, ngf*mult*2, 3, 2, 1, bias=use_bias), norm_layer(ngf*mult*2), nn.ReLU(True)]
        mult = 2**n_downsampling
        for i in range(n_blocks): model += [ResidualBlock(ngf*mult, padding_type, norm_layer, use_dropout)]
        for i in range(n_downsampling): mult = 2**(n_downsampling-i); model += [nn.ConvTranspose2d(ngf*mult, int(ngf*mult/2), 3, 2, 1, 1, bias=use_bias), norm_layer(int(ngf*mult/2)), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, 7, 1, 0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input): return self.model(input)

# --- Discriminator (PatchGAN) ---
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial: use_bias = norm_layer.func == nn.InstanceNorm2d
        else: use_bias = norm_layer == nn.InstanceNorm2d
        kw=4; padw=1
        sequence = [nn.Conv2d(input_nc, ndf, kw, 2, padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers): nf_mult_prev = nf_mult; nf_mult = min(2**n, 8); sequence += [nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kw, 2, padw, bias=use_bias), norm_layer(ndf*nf_mult), nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult; nf_mult = min(2**n_layers, 8)
        sequence += [nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kw, 1, padw, bias=use_bias), norm_layer(ndf*nf_mult), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(ndf*nf_mult, 1, kw, 1, padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input): return self.model(input)

class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):
        target = self.real_label if target_is_real else self.fake_label
        return target.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss