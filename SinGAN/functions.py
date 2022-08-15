import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch.nn as nn
import math
import cv2
from PIL import Image
from skimage import io as img
from skimage import color, morphology, filters
from sklearn.neighbors import NearestNeighbors
from SinGAN.imresize import imresize, imresize_to_shape, imresize_in
import os
import random
from torchvision import transforms
import dill



def read_image(opt):
    x = img.imread('%s%s' % (opt.input_img,opt.ref_image))
    return np2torch(x)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def norm(x):
    out = (x -0.5) *2
    return out.clamp(-1, 1)

def convert_image_np(inp):
    if inp.shape[1]==3:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,:,:,:])
        inp = inp.numpy().transpose((1,2,0))
    else:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,-1,:,:])
        inp = inp.numpy().transpose((0,1))
    inp = np.clip(inp,0,1)
    return inp

def save_image(real_cpu,receptive_feild,ncs,epoch_num,file_name):
    fig,ax = plt.subplots(1)
    if ncs==1:
        ax.imshow(real_cpu.view(real_cpu.size(2),real_cpu.size(3)),cmap='gray')
    else:
        ax.imshow(convert_image_np(real_cpu.cpu()))
    rect = patches.Rectangle((0,0),receptive_feild,receptive_feild,linewidth=5,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    ax.axis('off')
    plt.savefig(file_name)
    plt.close(fig)

def convert_image_np_2d(inp):
    inp = denorm(inp)
    inp = inp.numpy()
    return inp

def generate_noise(size,num_samp=1,device='cpu',type='gaussian', scale=1):
    if type == 'gaussian':
        noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), device=device)
        noise = upsampling(noise,size[1], size[2])
    if type =='gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device)+5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1+noise2
    if type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    return noise

def plot_learning_curves(G_loss,D_loss,epochs,label1,label2,name):
    fig,ax = plt.subplots(1)
    n = np.arange(0,epochs)
    plt.plot(n,G_loss,n,D_loss)
    plt.xlabel('epochs')
    plt.legend([label1,label2],loc='upper right')
    plt.savefig('%s.png' % name)
    plt.close(fig)

def plot_learning_curve(loss,epochs,name):
    fig,ax = plt.subplots(1)
    n = np.arange(0,epochs)
    plt.plot(n,loss)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.savefig('%s.png' % name)
    plt.close(fig)

def upsampling(im,sx,sy):
    m = nn.Upsample(size=[round(sx),round(sy)],mode='bilinear',align_corners=True)
    return m(im)

def reset_grads(model,require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model

def move_to_gpu(t,device='cuda'):
    if (torch.cuda.is_available()):
        t = t.to(device)
    return t

def move_to_cpu(t):
    t = t.to(torch.device('cpu'))
    return t

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device,opt,vgg=False,q = 0):
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)#cuda() #gpu) #if use_cuda else alpha
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)#.cuda()
    if opt.mask_scale:
        disc_interpolates = netD(interpolates,mask=opt.mask_background)
    else:
        disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates , inputs=interpolates,
                              grad_outputs=torch.ones_like(disc_interpolates).to(device),#.cuda(), #if use_cuda else torch.ones(
                                  #disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.pixelImageToMaskRatio * LAMBDA
    return gradient_penalty

def read_image(opt):
    x = img.imread('%s/%s' % (opt.input_dir,opt.input_name))
    x = np2torch(x,opt)
    x = x[:,0:3,:,:]
    return x

def read_image_dir(dir,opt):
    x = img.imread('%s' % (dir))
    x = np2torch(x,opt)
    x = x[:,0:3,:,:]
    return x

def np2torch(x,opt=None):
    if opt is None:
        nc_im = len(x.shape)
    else:
        nc_im = opt.nc_im
    if nc_im >= 3:
        x = x[:,:,:,None]
        x = x.transpose((3, 2, 0, 1))
        if x.max() > 2:
            x = x / 255
    else:
        # x = color.rgb2gray(x)
        x = x[:,:,None,None]
        x = x.transpose(3, 2, 0, 1)
    x = torch.from_numpy(x)
    if not(opt.not_cuda):
        x = move_to_gpu(x,opt.device)
    x = x.type(torch.cuda.FloatTensor) if not(opt.not_cuda) else x.type(torch.FloatTensor)
    x = norm(x)
    return x

def torch2uint8(x):
    x = x[0,:,:,:]
    x = x.permute((1,2,0))
    x = 255*denorm(x)
    x = x.cpu().numpy()
    x = x.astype(np.uint8)
    return x

def read_image2np(opt):
    x = img.imread('%s/%s' % (opt.input_dir,opt.input_name))
    x = x[:, :, 0:3]
    return x


def save_networks(netG,netD,z,opt):
    torch.save(netG.state_dict(), '%s/netG.pth' % (opt.outf))
    torch.save(netD.state_dict(), '%s/netD.pth' % (opt.outf))
    torch.save(z, '%s/z_opt.pth' % (opt.outf))

def adjust_scales2image(real_,opt):
    opt.num_scales = math.ceil(
        (math.log(math.pow(opt.min_size / (min(real_.shape[2], real_.shape[3])), 1), opt.scale_factor_init))) + 1
    scale2stop = math.ceil(
        math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]),
                 opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    #long axis is set to max size
    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]),1)
    real = imresize(real_, opt.scale1, opt)
    opt.scale_factor = math.pow(opt.min_size / (min(real.shape[2], real.shape[3])), 1 / (opt.stop_scale))
    scale2stop = math.ceil(
        math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]),
                 opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    return real



def creat_reals_pyramid(real,reals,opt):
    reals = []
    for i in range(0,opt.stop_scale+1,1):
        scale = math.pow(opt.scale_factor,opt.stop_scale-i)
        curr_real = imresize(real,scale,opt)
        print(curr_real.shape)
        reals.append(curr_real.to(opt.device))
    return reals

def creat_reals_mask_pyramid(real,reals,opt,mask):
    reals = []
    masks = []
    for i in range(0,opt.stop_scale+1,1):
        scale = math.pow(opt.scale_factor,opt.stop_scale-i)
        curr_real = imresize(real,scale,opt)
        curr_mask = resize_and_tresh_mask(mask, scale)
        print(curr_real.shape)
        reals.append(curr_real.to(opt.device))
        masks.append(torch.from_numpy(curr_mask).to(opt.device))
    return reals,masks


def load_trained_pyramid(opt, mode_='train'):
    mode = opt.mode
    opt.mode = 'train'
    if (mode == 'animation_train') | (mode == 'SR_train') | (mode == 'paint_train'):
        opt.mode = mode
    dir = generate_dir2save(opt)
    if(os.path.exists(dir)):
        Gs = torch.load('%s/Gs.pth' % dir,map_location=opt.device)
        Ds = torch.load('%s/Ds.pth' % dir,map_location=opt.device)
        Zs = torch.load('%s/Zs.pth' % dir,map_location=opt.device)
        reals = torch.load('%s/reals.pth' % dir,map_location=opt.device)
        NoiseAmp = torch.load('%s/NoiseAmp.pth' % dir,map_location=opt.device, pickle_module=dill)
    else:
        print('no appropriate trained model is exist, please train first')
    opt.mode = mode
    return Gs,Ds,Zs,reals,NoiseAmp

def generate_in2coarsest(reals,scale_v,scale_h,opt):
    real = reals[opt.gen_start_scale]
    real_down = upsampling(real, scale_v * real.shape[2], scale_h * real.shape[3])
    if opt.gen_start_scale == 0:
        in_s = torch.full(real_down.shape, 0, device=opt.device)
    else:
        in_s = upsampling(real_down, real_down.shape[2], real_down.shape[3])
    return in_s

def generate_dir2save(opt):
    dir2save = None
    if (opt.mode == 'train') | (opt.mode == 'SR_train'):
        dir2save = 'TrainedModels/%s/scale_factor=%.3f,alpha=%.2f,lambda_grad=%.3f' % (opt.input_name[:-4], opt.scale_factor_init,opt.alpha,opt.lambda_grad)
    elif (opt.mode == 'animation_train') :
        dir2save = 'TrainedModels/%s/scale_factor=%f_noise_padding' % (opt.input_name[:-4], opt.scale_factor_init)
    elif (opt.mode == 'paint_train') :
        dir2save = 'TrainedModels/%s/scale_factor=%f_paint/start_scale=%d' % (opt.input_name[:-4], opt.scale_factor_init,opt.paint_start_scale)
    elif opt.mode == 'random_samples':
        dir2save = '%s/RandomSamples/%s/scale_factor=%.2f,alpha=%.2f,lambda_grad=%.2f' % (opt.out,opt.input_name[:-4], opt.scale_factor_init, opt.alpha,opt.lambda_grad)
    elif opt.mode == 'random_samples_arbitrary_sizes':
        dir2save = '%s/RandomSamples_ArbitrerySizes/%s/scale_v=%f_scale_h=%f' % (opt.out,opt.input_name[:-4], opt.scale_v, opt.scale_h)
    elif opt.mode == 'animation':
        dir2save = '%s/Animation/%s' % (opt.out, opt.input_name[:-4])
    elif opt.mode == 'SR':
        dir2save = '%s/SR/%s' % (opt.out, opt.sr_factor)
    elif opt.mode == 'harmonization':
        dir2save = '%s/Harmonization/%s/%s_out' % (opt.out, opt.input_name[:-4],opt.ref_name[:-4])
    elif opt.mode == 'editing':
        dir2save = '%s/Editing/%s/%s_out' % (opt.out, opt.input_name[:-4],opt.ref_name[:-4])
    elif opt.mode == 'paint2image':
        dir2save = '%s/Paint2image/%s/%s_out' % (opt.out, opt.input_name[:-4],opt.ref_name[:-4])
        if opt.quantization_flag:
            dir2save = '%s_quantized' % dir2save
    if opt.details:
        dir2save = dir2save + '_' + opt.details
    if 'background' in opt.details:
        dir2save += '_ratio%d' % opt.ratio
    return dir2save

def post_config(opt):
    # init fixed parameters
    opt.device = torch.device("cpu" if opt.not_cuda else "cuda:%d" % opt.gpu_num)
    opt.niter_init = opt.niter
    opt.noise_amp_init = opt.noise_amp
    opt.nfc_init = opt.nfc
    opt.min_nfc_init = opt.min_nfc
    opt.scale_factor_init = opt.scale_factor
    opt.out_ = 'TrainedModels/%s/scale_factor=%f/' % (opt.input_name[:-4], opt.scale_factor)
    if opt.mode == 'SR':
        opt.alpha = 100
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if torch.cuda.is_available() and opt.not_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return opt

def calc_init_scale(opt):
    in_scale = math.pow(1/2,1/3)
    iter_num = round(math.log(1 / opt.sr_factor, in_scale))
    in_scale = pow(opt.sr_factor, 1 / iter_num)
    return in_scale,iter_num


def dilate_mask(mask,opt,dilation=5,sigma_gaus=5):
    if opt.mode == "harmonization" or opt.mode == "random_samples" or opt.mode == 'train':
        element = morphology.disk(radius=dilation) #
    elif opt.mode == "editing":
        element = morphology.disk(radius=20)
    elif opt.mode == 'train':
        element = morphology.disk(radius=int(np.max((opt.mask_object.shape[2],opt.mask_object.shape[3]))*0.1))
    mask = torch2uint8(mask)
    mask = mask[:,:,0]
    mask = morphology.binary_dilation(mask,selem=element)
    mask = filters.gaussian(mask, sigma=sigma_gaus)
    nc_im = opt.nc_im
    opt.nc_im = 1
    mask = np2torch(mask,opt)
    opt.nc_im = nc_im
    mask = mask.expand(1, 3, mask.shape[2], mask.shape[3])
    if mask.min() != mask.max():
        mask = (mask-mask.min())/(mask.max()-mask.min())
    return mask


def image2mask(opt,dir_name,img_name):

    mask_model = opt.mask_model
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # perform pre-processing
    input_image = Image.open(dir_name+'/'+img_name)
    input_image = input_image.convert('RGB')
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch of size 1 as expected by the model
    # send to device
    model = mask_model.to(opt.device)
    input_batch = input_batch.to(opt.device)
    # forward pass
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)
    np.unique(output_predictions.cpu().numpy())
    # create a mask
    mask = torch.zeros_like(output_predictions).float().to(opt.device)
    labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
     'bus', 'car', 'cat', 'chair', 'cow',
     'diningtable','dog', 'horse', 'motorbike', 'person',
     'pottedplant','sheep', 'sofa', 'train', 'tvmonitor']
    number = []
    for j in range(len(labels)):
        if labels[j] in opt.input_name:
            number.append(j+1)
            mask[output_predictions == (j+1)] = 1
    if 'labrador' in opt.input_name or 'poodle' in opt.input_name or 'labradoodle' in opt.input_name or 'maltese' in opt.input_name:
        mask[output_predictions == 12] = 1
    return mask

def calc_mask(mask,real,opt,folder=''):
    opt.mask_object = mask.unsqueeze(0).unsqueeze(0).repeat(1,3,1,1)
    mask_background = (1-opt.mask_object)
    if not folder=='':
        plt.imsave('%s/object_mask.png' % (folder), opt.mask_object[0,0,:,:].cpu(), vmin=0, vmax=1,cmap='gray')
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(convert_image_np(real))
        plt.title('real')
        plt.subplot(2, 2, 2)
        plt.imshow(mask_background[0, 0, :, :].cpu(), cmap='gray')
        plt.colorbar()
        plt.title('background mask')
        plt.imsave('%s/background_mask.png' % (folder), mask_background[0,0,:,:].cpu(), vmin=0, vmax=1,cmap='gray')

    opt.idx_object = torch.where(mask_background[0, 0, :, :] == 0)
    opt.idx_background = torch.where(mask_background[0, 0, :, :] == 1)

    opt.mask_background = mask_background

    RF = opt.ker_size + ((opt.ker_size - 1) * (opt.num_layer - 1)) * opt.stride
    halfRF = int(RF/2)
    kernel = np.ones((RF, RF), np.uint8)
    mask_tmp = cv2.dilate(np.float32(opt.mask_object[0,0,:,:].cpu()), kernel)
    mask_background_forD = 1 - mask_tmp[halfRF:-halfRF,halfRF:-halfRF]
    if not folder=='':
        plt.subplot(2, 2, 3)
        tmp = real.cpu() * mask_background.cpu()
        plt.imshow(convert_image_np(tmp))
        plt.savefig('%s/background_mask' % (folder))
        plt.close()
    opt.idx_background_D = np.where(mask_background_forD == 1)
    opt.idx_object_D = np.where(mask_background_forD == 0)
    opt.mask_background_forD = mask_background_forD
    opt.mask_object_forD = 1-mask_background_forD
    return opt.mask_object

def torch2np(x):
    x = x[0,:,:,:]
    x = x.permute(1,2,0)
    x = denorm(x)
    x = x.cpu().numpy()
    return x

def load_mask(opt):
    opt.mask_original_image = img.imread('%s/%s' % (opt.input_dir, opt.input_mask))
    if len(opt.mask_original_image.shape) > 2:
        opt.mask_original_image = opt.mask_original_image[:, :, 0]
    opt.mask_original_image[opt.mask_original_image < 255 / 2] = 0
    opt.mask_original_image[opt.mask_original_image > 255 / 2] = 1
    opt.mask_original_image = torch.from_numpy(opt.mask_original_image).to(opt.device)



def RGB_1NN(img, opt,idx_img = None):
    query_values = img[0,:,idx_img[0],idx_img[1]].t()
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(opt.RGB_real.cpu())
    distances, idx_dist = nbrs.kneighbors(query_values.detach().cpu())
    idx_dist = idx_dist.reshape(-1)
    loss_mse = torch.nn.MSELoss()
    loss = loss_mse(query_values, opt.RGB_real[idx_dist, :])
    return loss


def calc_scale_to_start_masking(opt,real):
    mask = opt.mask_original_image.cpu()
    mask = resize_and_tresh_mask(mask,opt.scale1)
    for i in range(0,opt.stop_scale+1,1):
        scale = math.pow(opt.scale_factor,opt.stop_scale-i)
        mask_tmp = resize_and_tresh_mask(mask,scale)
        mask_tmp = dilate_mask_with_RF(mask_tmp, opt)
        RF = opt.ker_size + ((opt.ker_size - 1) * (opt.num_layer - 1)) * opt.stride
        halfRF = int(RF / 2)
        mask_tmp = mask_tmp[halfRF:-halfRF,halfRF:-halfRF]
        ratio = 100*mask_tmp.sum()/(mask_tmp.shape[0] * mask_tmp.shape[1])
        valid_patches = (1-mask_tmp).sum()
        if i == 0:
            if ratio < 25:
                break
        elif ratio < 60 or valid_patches > 5000:
            break
    scale_to_start_masking = i
    return scale_to_start_masking

def resize_and_tresh_mask(mask,scale):
    mask = imresize_in(mask,scale)
    mask[np.where(mask!=0)] = 1
    return mask

def dilate_mask_with_RF(mask,opt):
    RF = opt.ker_size + ((opt.ker_size - 1) * (opt.num_layer - 1)) * opt.stride
    kernel = np.ones((RF, RF), np.uint8)
    mask = cv2.dilate(np.float32(mask), kernel)
    return mask

