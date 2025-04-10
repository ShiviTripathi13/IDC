import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("..")

#############################
#   Missing Utility Classes #
#############################

class Swish(nn.Module):
    """Swish activation: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)

class Concat(nn.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))
        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)
            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3: diff3 + target_shape3])
        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)
    


def get_kernel(factor, kernel_type, phase, kernel_width, support=None, sigma=None):
    """
    Generate a simple kernel for downsampling.

    For demonstration, this function returns a normalized box kernel,
    regardless of the kernel_type. You can extend it to support 'lanczos',
    'gauss', etc., as needed.

    Arguments:
        factor (int): The downsampling factor.
        kernel_type (str): Type of kernel; e.g. 'box', 'lanczos', 'gauss', etc.
        phase (float): Either 0 or 0.5, indicating kernel phase.
        kernel_width (int): Width (and height) of the kernel.
        support: (Optional) Parameter for specific kernel types.
        sigma: (Optional) Standard deviation for Gaussian kernel.

    Returns:
        A numpy.ndarray representing the kernel.
    """
    # Example: a simple box filter
    kernel = np.ones((kernel_width, kernel_width), dtype=np.float32)
    kernel = kernel / np.sum(kernel)
    return kernel


class Downsampler(nn.Module):
    """
    Downsampler module as defined in the Deep Image Prior paper.
    See: http://www.realitypixels.com/turk/computergraphics/ResamplingFilters.pdf
    """
    def __init__(self, n_planes, factor, kernel_type, phase=0, kernel_width=None, support=None, sigma=None, preserve_size=False):
        super(Downsampler, self).__init__()
        assert phase in [0, 0.5], 'phase should be 0 or 0.5'
        if kernel_type == 'lanczos2':
            support = 2
            kernel_width = 4 * factor + 1
            kernel_type_ = 'lanczos'
        elif kernel_type == 'lanczos3':
            support = 3
            kernel_width = 6 * factor + 1
            kernel_type_ = 'lanczos'
        elif kernel_type == 'gauss12':
            kernel_width = 7
            sigma = 1 / 2
            kernel_type_ = 'gauss'
        elif kernel_type == 'gauss1sq2':
            kernel_width = 9
            sigma = 1. / np.sqrt(2)
            kernel_type_ = 'gauss'
        elif kernel_type in ['lanczos', 'gauss', 'box']:
            kernel_type_ = kernel_type
        else:
            assert False, 'wrong kernel name'

        # Here, get_kernel is assumed to be provided elsewhere.
        # If it is not defined, you should define it. For now, assume it's imported.

        self.kernel = get_kernel(factor, kernel_type_, phase, kernel_width, support=support, sigma=sigma)
        downsampler = nn.Conv2d(n_planes, n_planes, kernel_size=self.kernel.shape, stride=factor, padding=0)
        with torch.no_grad():
            downsampler.weight.zero_()
            if downsampler.bias is not None:
                downsampler.bias.zero_()
            kernel_torch = torch.from_numpy(self.kernel)
            for i in range(n_planes):
                downsampler.weight[i, i].copy_(kernel_torch)
        self.downsampler_ = downsampler
        self.preserve_size = preserve_size
        if preserve_size:
            if self.kernel.shape[0] % 2 == 1:
                pad = int((self.kernel.shape[0] - 1) / 2.)
            else:
                pad = int((self.kernel.shape[0] - factor) / 2.)
            self.padding = nn.ReplicationPad2d(pad)

    def forward(self, input):
        if self.preserve_size:
            x = self.padding(input)
        else:
            x = input
        return self.downsampler_(x)

#########################
#        Blocks         #
#########################

class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=stride, padding=padd)
        self.norm = nn.BatchNorm2d(out_channel)
        self.relu = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):
        # Use the built-in convolution forward.
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        try:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        except:
            print('no learnable params in BN')

#########################
#   Discriminators      #
#########################

class WDiscriminator(nn.Module):
    def __init__(self, opt):
        super(WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x

class GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        self.body = nn.Sequential()
        for i in range(int(opt.num_layer) - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
            nn.Tanh()
        )

    def forward(self, x, y):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, 0:3, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y

#########################
#     Mask Blocks       #
#########################

class MaskConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(MaskConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=stride, padding=padd)
        self.norm = nn.BatchNorm2d(out_channel)
        self.relu = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x, mask):
        # Use built-in conv forward.
        x = self.conv(x)
        diff = int((mask.shape[2] - x.shape[2]) / 2)
        im = torch.abs(1 - mask).cpu()
        im_tensor = torch.Tensor(np.expand_dims(im[:, 0, :, :], 0))
        kernel = np.ones([2 * diff + 1, 2 * diff + 1], dtype=np.float32)
        kernel_tensor = torch.Tensor(np.expand_dims(np.expand_dims(kernel, 0), 0))
        torch_result = F.conv2d(im_tensor, kernel_tensor)
        torch_result = torch.where(torch_result > 0,
                                   torch.tensor(1.0, dtype=torch_result.dtype, device=torch_result.device),
                                   torch_result)
        mask = torch.abs(1 - torch_result[0, 0, :, :]).to(x.device)

        # Apply mask to background pixels
        idx = np.where(mask.cpu() == 1)
        y = x[:, :, idx[0], idx[1]]
        x_vec = y.unsqueeze(-1)
        x_out = self.norm(x_vec)
        x_new = torch.zeros_like(x)
        x_new[:, :, idx[0], idx[1]] = x_out.squeeze(-1)
        x = x_new
        x = self.relu(x)
        return x

class WDiscriminatorMask(nn.Module):
    def __init__(self, opt):
        super(WDiscriminatorMask, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = MaskConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, (i + 1)))
            block = MaskConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones_like(x)
        x = self.head(x, mask)
        for block in self.body.children():
            x = block(x, mask)
        x = self.tail(x)
        diff = int((mask.shape[2] - x.shape[2]) / 2)
        im = torch.abs(1 - mask).cpu()
        im_tensor = torch.Tensor(np.expand_dims(im[:, 0, :, :], 0))
        kernel = np.ones([2 * diff + 1, 2 * diff + 1], dtype=np.float32)
        kernel_tensor = torch.Tensor(np.expand_dims(np.expand_dims(kernel, 0), 0))
        torch_result = F.conv2d(im_tensor, kernel_tensor)
        torch_result = torch.where(torch_result > 0,
                                   torch.tensor(1.0, dtype=torch_result.dtype, device=torch_result.device),
                                   torch_result)
        mask = torch.abs(1 - torch_result[0, 0, :, :]).to(x.device)
        x = x * mask
        return x

#########################
#      Skip Nets        #
#########################

def skip(num_input_channels=2, num_output_channels=3,
         num_channels_down=[16, 32, 64, 128, 128],
         num_channels_up=[16, 32, 64, 128, 128],
         num_channels_skip=[4, 4, 4, 4, 4],
         filter_size_down=3, filter_size_up=3, filter_skip_size=1,
         need_sigmoid=True, need_bias=True,
         pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU',
         need1x1_up=True):
    """Assembles encoder-decoder with skip connections."""
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)
    n_scales = len(num_channels_down)
    if not isinstance(upsample_mode, (list, tuple)):
        upsample_mode = [upsample_mode] * n_scales
    if not isinstance(downsample_mode, (list, tuple)):
        downsample_mode = [downsample_mode] * n_scales
    if not isinstance(filter_size_down, (list, tuple)):
        filter_size_down = [filter_size_down] * n_scales
    if not isinstance(filter_size_up, (list, tuple)):
        filter_size_up = [filter_size_up] * n_scales

    last_scale = n_scales - 1
    model = nn.Sequential()
    model_tmp = model
    input_depth = num_input_channels
    for i in range(n_scales):
        deeper = nn.Sequential()
        skip_module = nn.Sequential()
        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip_module, deeper))
        else:
            model_tmp.add(deeper)
        model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))
        if num_channels_skip[i] != 0:
            skip_module.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip_module.add(bn(num_channels_skip[i]))
            skip_module.add(act(act_fun))
        deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad,
                        downsample_mode=downsample_mode[i]))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))
        deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))
        deeper_main = nn.Sequential()
        if i == n_scales - 1:
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]
        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))
        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))
        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))
        input_depth = num_channels_down[i]
        model_tmp = deeper_main
    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())
    return model

def skip_interp(num_input_channels=2, num_output_channels=3,
                num_channels_down=[16, 32, 64, 128, 128],
                num_channels_up=[16, 32, 64, 128, 128],
                num_channels_skip=[4, 4, 4, 4, 4],
                filter_size_down=3, filter_size_up=3, filter_skip_size=1,
                need_sigmoid=True, need_bias=True,
                pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU',
                need1x1_up=True):
    """Assembles encoder-decoder with skip connections using interpolation."""
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)
    n_scales = len(num_channels_down)
    if not isinstance(upsample_mode, (list, tuple)):
        upsample_mode = [upsample_mode] * n_scales
    if not isinstance(downsample_mode, (list, tuple)):
        downsample_mode = [downsample_mode] * n_scales
    if not isinstance(filter_size_down, (list, tuple)):
        filter_size_down = [filter_size_down] * n_scales
    if not isinstance(filter_size_up, (list, tuple)):
        filter_size_up = [filter_size_up] * n_scales

    last_scale = n_scales - 1
    model = nn.Sequential()
    model_tmp = model
    input_depth = num_input_channels
    for i in range(n_scales):
        deeper = nn.Sequential()
        skip_module = nn.Sequential()
        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip_module, deeper))
        else:
            model_tmp.add(deeper)
        model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))
        if num_channels_skip[i] != 0:
            skip_module.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip_module.add(bn(num_channels_skip[i]))
            skip_module.add(act(act_fun))
        deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad,
                        downsample_mode=downsample_mode[i]))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))
        deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))
        deeper_main = nn.Sequential()
        if i == n_scales - 1:
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]
        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))
        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))
        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))
        input_depth = num_channels_down[i]
        model_tmp = deeper_main
    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())
    return model

#########################
#    Utility Functions  #
#########################

def act(act_fun='LeakyReLU'):
    '''
    Returns an activation function.
    '''
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=False)
        elif act_fun == 'Swish':
            return Swish()
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Sequential()
        else:
            assert False, "Unknown activation function: " + act_fun
    else:
        return act_fun()

def bn(num_features):
    return nn.BatchNorm2d(num_features)

def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':
        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride, padding=1)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode in ['lanczos2', 'lanczos3']:
            downsampler = Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5, preserve_size=True)
        else:
            assert False, "Unknown downsample mode"
        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)
    layers = [layer for layer in (padder, convolver, downsampler) if layer is not None]
    return nn.Sequential(*layers)

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)

torch.nn.Module.add = add_module
