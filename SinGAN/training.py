import SinGAN.functions as functions
import SinGAN.models as models
import os
import sys
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import math
from tqdm import tqdm
import numpy as np
import copy
import matplotlib.pyplot as plt
from SinGAN.imresize import imresize, imresize_to_shape
from SinGAN.models import skip



sys.path.append("..")


def train(opt,Gs,Ds,Zs,reals,NoiseAmp):
    real_ = functions.read_image(opt)
    real = imresize(real_, opt.scale1, opt)
    nfc_prev = 0
    scale_num = len(Gs)
    mask_tmp = opt.mask_original_image.cpu()
    mask = functions.resize_and_tresh_mask(mask_tmp, opt.scale1)
    opt.mask = mask
    opt.real_=real_
    reals, masks = functions.creat_reals_mask_pyramid(real, reals, opt, opt.mask)
    opt.masks = masks
    tmp = functions.convert_image_np(reals[-1])
    idx = torch.where(masks[-1].cpu() == 1)
    tmp[idx[0], idx[1]] = 0
    plt.imsave('%s/real_masked.png' % (opt.dir2save),tmp)
    plt.imsave('%s/mask.png' % (opt.dir2save), masks[-1].cpu(),cmap='gray')
    if scale_num >= opt.stop_scale + 1:
        return reals, masks
    mask_tmp = functions.calc_mask(masks[opt.scale_to_start_masking-1], reals[opt.scale_to_start_masking-1], opt)
    if scale_num < opt.scale_to_start_masking:
        dir2save = functions.generate_dir2save(opt)
        init_inpaint = calc_init_inpaint(reals[opt.scale_to_start_masking-1],mask_tmp,opt).detach()
        plt.imsave('%s/init_inpaint.png' % dir2save,functions.convert_image_np(init_inpaint), vmin=0, vmax=1)
        tmp_init = functions.convert_image_np(reals[opt.scale_to_start_masking-1])
        idx = torch.where(masks[opt.scale_to_start_masking-1].cpu() == 1)
        tmp_init[idx[0], idx[1]] = 0

        for j in range(np.min((opt.scale_to_start_masking,len(reals)))):
            init_inpaint_resize = imresize_to_shape(init_inpaint,[reals[j].shape[2],reals[j].shape[3]],opt)
            mask_tmp = functions.calc_mask(masks[j], reals[j], opt)
            reals[j] = (init_inpaint_resize * mask_tmp + reals[j] * (1-mask_tmp)).float()

    in_s = torch.full(reals[0].shape, 0, device=opt.device)
    for scale_num in range(scale_num,opt.stop_scale+1):
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_, scale_num)
        try:
            os.makedirs(opt.outf)
        except OSError:
            pass

        plt.imsave('%s/real_scale.png' % (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)
        opt.num_scale = scale_num
        if opt.num_scale >= opt.scale_to_start_masking:
            opt.mask_scale = 1
            opt.lr = opt.lr_mask
            opt.decrease_LR = 3500
            opt.niter = opt.niter_mask
        else:
            opt.mask_scale = 0

        opt.paste_scale = 0
        D_curr, G_curr = init_models(opt)

        f = open(opt.file_name, "a")
        f.write('Dis:')
        f.write("\n")
        f.write(str(D_curr))
        f.write("\n")
        f.write('Gen:')
        f.write("\n")
        f.write(str(G_curr))
        f.write("\n")
        f.close()


        if (nfc_prev == opt.nfc):
            G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out_, scale_num - 1)))
            D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_, scale_num - 1)))
        z_curr, in_s, G_curr, D_curr = train_single_scale(D_curr, G_curr, reals, Gs, Zs, in_s, NoiseAmp, opt,masks)
        G_curr = functions.reset_grads(G_curr, False)
        G_curr.eval()
        D_curr = functions.reset_grads(D_curr, False)
        D_curr.eval()

        Gs.append(G_curr)
        Zs.append(z_curr)
        Ds.append(D_curr)
        NoiseAmp.append(opt.noise_amp)

        torch.save(Zs, '%s/Zs.pth' % (opt.out_))
        torch.save(Gs, '%s/Gs.pth' % (opt.out_))
        torch.save(Ds, '%s/Ds.pth' % (opt.out_))
        torch.save(reals, '%s/reals.pth' % (opt.out_))
        torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

        nfc_prev = opt.nfc
        del D_curr, G_curr

    return reals, masks



def train_single_scale(netD,netG,reals,Gs,Zs,in_s,NoiseAmp,opt,masks=None,centers=None):
    scale_num = opt.num_scale
    real = reals[scale_num]
    batch_size = real.shape[0]
    opt.nzx = real.shape[2]#+(opt.ker_size-1)*(opt.num_layer)
    opt.nzy = real.shape[3]#+(opt.ker_size-1)*(opt.num_layer)
    opt.receptive_field = opt.ker_size + ((opt.ker_size-1)*(opt.num_layer-1))*opt.stride
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    opt.pad1 = pad_noise
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    if opt.padding == 'zeros':
        m_noise = nn.ZeroPad2d(int(pad_noise))
        m_image = nn.ZeroPad2d(int(pad_image))
    elif opt.padding == 'reflect':
        m_noise = nn.ReflectionPad2d(int(pad_noise))
        m_image = nn.ReflectionPad2d(int(pad_image))
    elif opt.padding == 'replicate':
        m_noise = nn.ReplicationPad2d(int(pad_noise))
        m_image = nn.ReplicationPad2d(int(pad_image))
    elif opt.padding == 'ref_zero':
        opt.nzx = real.shape[2] + (opt.ker_size - 1) * (opt.num_layer)
        opt.nzy = real.shape[3] + (opt.ker_size - 1) * (opt.num_layer)
        pad_noise = 0
        m_noise = nn.ReflectionPad2d(int(pad_noise))
        m_image = nn.ZeroPad2d(int(pad_image))
    elif opt.padding == 'pad_with_noise':
        opt.nzx = real.shape[2] + (opt.ker_size - 1) * (opt.num_layer)
        opt.nzy = real.shape[3] + (opt.ker_size - 1) * (opt.num_layer)
        pad_noise = 0
        m_noise = nn.ZeroPad2d(int(pad_noise))
        m_image = nn.ZeroPad2d(int(pad_image))

    alpha = opt.alpha
    fixed_noise = functions.generate_noise([opt.nc_z,opt.nzx,opt.nzy],num_samp=batch_size,device=opt.device)
    z_opt = torch.full(fixed_noise.shape, 0, device=opt.device)
    z_opt = m_noise(z_opt)

    # setup optimizer

    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[opt.decrease_LR],gamma=opt.gamma)

    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD,milestones=[opt.decrease_LR],gamma=opt.gamma)

    errD2plot = []
    gradPenalty = []
    errG2plot = []
    D_real2plot = []
    D_fake2plot = []
    D_realfake2reclossplot = []
    D_realfake2GPlossplot = []
    z_opt2plot = []


    functions.calc_mask(masks[scale_num], real, opt)
    if opt.mask_scale:
        if scale_num >= opt.scale_to_start_grad:
            dilate = np.min((opt.num_scale,5))
            mask = (1-functions.dilate_mask(2*(opt.mask_object-0.5),opt,dilation=dilate,sigma_gaus=5))
            mask[:,:,opt.idx_object[0].cpu(), opt.idx_object[1].cpu()] = 0
            if scale_num > opt.stop_scale - opt.num_scales_to_enlarge_alpha:
                alpha = 100

        plt.imsave('%s/mask_for_rec_loss.png' % opt.outf, mask[0,0,:,:].cpu(),cmap='gray')
        opt.mask_gradual = mask
        opt.mask_object = functions.calc_mask(masks[scale_num], real[:, 0:3, :, :], opt, folder=opt.outf)
        opt.pixelImageToMaskRatio = torch.flatten(real[0, 0, :, :]).shape[0] / \
                                    torch.flatten(real[0, 0, opt.idx_background[0], opt.idx_background[1]]).shape[0]
        opt.pixelImageTorecMaskRatio = torch.flatten(real[0, 0, :, :]).shape[0] / opt.mask_gradual[0,0,:,:].sum()
    else:
        opt.pixelImageToMaskRatio = 1
        opt.pixelImageTorecMaskRatio = 1


    for epoch in tqdm(range(opt.niter)):
        opt.epoch = epoch
        if (Gs == []):
            if epoch==0:
                z_opt = functions.generate_noise([1,opt.nzx,opt.nzy], device=opt.device)
                z_opt = z_opt.expand(batch_size,opt.nc_z,opt.nzx,opt.nzy)
                z_opt = m_noise(z_opt)
            noise_ = functions.generate_noise([1,opt.nzx,opt.nzy], device=opt.device)
            noise_ = noise_.expand(1,opt.nc_z,opt.nzx,opt.nzy)
            noise_ = m_noise(noise_)
        else:
            if epoch==0:
                z_opt = functions.generate_noise([opt.nc_z,opt.nzx,opt.nzy],num_samp=batch_size, device=opt.device)
                z_opt = m_noise(z_opt)
            noise_ = functions.generate_noise([opt.nc_z,opt.nzx,opt.nzy],num_samp=batch_size, device=opt.device)
            noise_ = m_noise(noise_)
       


        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################
        for j in range(opt.Gsteps):
            netG.zero_grad()
            output = netD(fake)
            errG = -output.mean()
            errG.backward(retain_graph=True)

            if alpha!=0:
                loss = nn.MSELoss()
                Z_opt = opt.noise_amp * z_opt + z_prev
                tmp1 = netG(Z_opt.detach(),z_prev)
                tmp2 = real.clone()
                if opt.mask_scale:
                    tmp2[:, 0:3, opt.idx_object[0], opt.idx_object[1]] = 0
                    tmp1[:, 0:3, opt.idx_object[0], opt.idx_object[1]] = 0
                    tmp2 *= opt.mask_gradual
                    tmp1 *= opt.mask_gradual
                rec_loss = alpha*loss(tmp1[:,0:3,:,:],tmp2[:,0:3,:,:]) * opt.pixelImageTorecMaskRatio
                rec_loss.backward(retain_graph=True)
                rec_loss = rec_loss.detach()
                if epoch % 500 == 0 or epoch == (opt.niter - 1):
                    plt.figure(figsize=(10, 10))
                    plt.subplot(1, 2, 1)
                    plt.imshow(functions.convert_image_np(tmp1.detach()))
                    t = (alpha*loss(tmp1,tmp2) * opt.pixelImageToMaskRatio).detach()
                    plt.title('netG(noise)_loss_%.5f' % t)
                    plt.axis('off')
                    plt.subplot(1, 2, 2)
                    plt.imshow(functions.convert_image_np(tmp2))
                    plt.axis('off')
                    plt.title('real')
                    plt.savefig('%s/rec_loss_epoch_%d.png' % (opt.outf, epoch))
                    plt.close()
            else:
                Z_opt = z_opt
                rec_loss = 0
            optimizerG.step()


        errG2plot.append((errG.detach()+rec_loss).cpu())

         ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        for j in range(opt.Dsteps):
            netD.zero_grad()
            if opt.mask_scale:
                output = netD(real,opt.mask_background).to(opt.device)
            else:
                output = netD(real)

            if (epoch % 500 == 0 or epoch == opt.niter-1) and j == 0:
                plt.figure(figsize=(20, 10))
                plt.subplot(1,2,1)
                plt.imshow(functions.convert_image_np(real))
                plt.title('input 1')
                plt.axis('off')
                plt.subplot(1,2,2)
                plt.imshow(output[0,0,:,:].detach().cpu())
                plt.colorbar()
                plt.axis('off')
                plt.savefig('%s/real_and_D_real_epoch_%d' % (opt.outf, epoch))
                plt.close()

            if opt.mask_scale:
                errD_real = -output[:,:,opt.idx_background_D[0],opt.idx_background_D[1]].mean()
            else:
                errD_real = -output.mean()

            errD_real.backward(retain_graph=True)
            D_x = -errD_real.item()

            # train with fake
            if (j==0) & (epoch == 0):
                if (Gs == []):
                    prev = torch.full([batch_size,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)
                    z_prev = torch.full([batch_size, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
                    in_s = copy.copy(prev)
                    prev = m_image(prev)
                    z_prev = m_noise(z_prev)
                    opt.noise_amp = 1
                else:
                    prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rand',m_noise,m_image,opt)
                    criterion = nn.MSELoss()
                    z_prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rec',m_noise,m_image,opt)
                    if opt.mask_scale:
                        real_tmp = real.clone()
                        real_tmp[:, 0:3, opt.idx_object[0], opt.idx_object[1]] = 0
                        z_prev_tmp = z_prev.clone()
                        z_prev_tmp[:, 0:3, opt.idx_object[0], opt.idx_object[1]] = 0
                        RMSE = torch.sqrt(criterion(real_tmp,z_prev_tmp) * opt.pixelImageToMaskRatio)
                    else:
                        RMSE = torch.sqrt(criterion(real[:,0:3,:,:], z_prev[:,0:3,:,:]))

                    opt.noise_amp = opt.noise_amp_init*RMSE
                    z_prev = m_image(z_prev)
                    prev = m_image(prev)
            else:
                prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rand',m_noise,m_image,opt)
                prev = m_image(prev)


            if (Gs == []):
                noise = noise_
            else:
                noise = opt.noise_amp*noise_+prev
            fake = netG(noise.detach(), prev)
            output = netD(fake.detach())

            errD_fake = output.mean()

            errD_fake.backward(retain_graph=True)
            D_G_z = output.mean().item()

            gradient_penalty = functions.calc_gradient_penalty(netD, real, fake, opt.lambda_grad, opt.device,opt)
            gradient_penalty.backward(retain_graph=True)
            errD = errD_real + errD_fake + gradient_penalty

            optimizerD.step()

        errD2plot.append(errD.detach().cpu())

        gradPenalty.append(gradient_penalty.detach().cpu())
        
        
        D_real2plot.append(D_x)
        D_fake2plot.append(D_G_z)
        D_realfake2GPlossplot.append(((-D_x + D_G_z) / gradient_penalty.detach()).cpu())

        if alpha!=0:
            D_realfake2reclossplot.append(((-D_x+D_G_z)/rec_loss).cpu())
            z_opt2plot.append(rec_loss.cpu())
        else:
            z_opt2plot.append(rec_loss)

        if epoch % 500 == 0 or epoch == (opt.niter-1):
            plt.imsave('%s/fake_sample_%d.png' % (opt.outf,epoch), functions.convert_image_np(fake[:,0:3,:,:].detach()), vmin=0, vmax=1)
            img = netG(Z_opt.detach(), z_prev).detach()
            plt.imsave('%s/G(z_opt)_%d.png'    % (opt.outf,epoch),  functions.convert_image_np(img[:,0:3,:,:]), vmin=0, vmax=1)
            if epoch == opt.niter - 1:
                plt.imsave('%s/z_prev.png' % (opt.outf), functions.convert_image_np(z_prev[:, 0:3, :, :]),
                           vmin=0, vmax=1)

            functions.plot_learning_curve(z_opt2plot, epoch + 1, '%s/rec_loss' % (opt.outf))
            functions.plot_learning_curves(D_fake2plot,D_real2plot, epoch+1, 'D_fake','D_real', '%s/real_vs_fake_mean_output_of_discriminator'   % (opt.outf))
            functions.plot_learning_curve(errD2plot, epoch+1,  '%s/errD_(real_fake_GP)' % (opt.outf))
            functions.plot_learning_curve(errG2plot, epoch+1,  '%s/errG_(errG_rec_loss)' % (opt.outf))
            functions.plot_learning_curve(D_realfake2GPlossplot, epoch+1,  '%s/adversarial_to_GP' % (opt.outf))
            functions.plot_learning_curve(gradPenalty, epoch+1,  '%s/gradient_penalty_values' % (opt.outf))

            if alpha!=0:
                functions.plot_learning_curve(D_realfake2reclossplot, epoch + 1, '%s/adversarial_loss_to_rec_loss' % (opt.outf))
            torch.save(z_opt, '%s/z_opt.pth' % (opt.outf))


        schedulerD.step()
        schedulerG.step()

    functions.save_networks(netG,netD,z_opt,opt)
    return z_opt,in_s,netG,netD

def draw_concat(Gs,Zs,reals,NoiseAmp,in_s,mode,m_noise,m_image,opt):
    G_z = in_s
    if opt.num_scale > 0:
        if mode == 'rand':
            count = 0
            pad_noise = int(((opt.ker_size-1)*opt.num_layer)/2)
            if (opt.padding=='pad_with_noise'):
                pad_noise = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                if count == 0:
                    z = functions.generate_noise([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                    z = z.expand(1, opt.nc_z, z.shape[2], z.shape[3])
                else:
                    z = functions.generate_noise([opt.nc_z, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                z = m_noise(z)
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp*z+G_z
                G_z = G(z_in.detach(),G_z)
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                count += 1
        if mode == 'rec':
            count = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_tmp = Z_opt
                z_in = noise_amp*z_tmp+G_z
                G_z = G(z_in.detach(),G_z)
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]].to(opt.device)
                count += 1
    return G_z

def init_models(opt):
    if opt.mask_scale:
        netD = models.WDiscriminatorMask(opt).to(opt.device)
    else:
        netD = models.WDiscriminator(opt).to(opt.device)
    netG = models.GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
    netD.apply(models.weights_init)
    netG.apply(models.weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))

    return netD, netG

def calc_init_inpaint(img,mask,opt):
    mask = 1-mask
    input_depth = 2
    LR = 0.001
    num_iter = 1001
    reg_noise_std = 0.03
    net_input_saved = functions.generate_noise([input_depth,img.shape[2],img.shape[3]],device=opt.device)
    min_val = np.min((img.shape[2],img.shape[3]))
    depth = 0
    while True:
        min_val /= 2
        min_val = np.floor(min_val)
        if min_val <= 1:
            break
        depth += 1


    net = skip(input_depth, 3, num_channels_down=[128] * depth,
                num_channels_up=[128] * depth, num_channels_skip=[0] * depth, upsample_mode='nearest',
                filter_skip_size=1, filter_size_up=3, filter_size_down=3,
                need_sigmoid=True, need_bias=True, pad='zero', act_fun='LeakyReLU').to(opt.device)

    optimizer = optim.Adam(net.parameters(), lr=LR)
    mse = torch.nn.MSELoss()
    opt.RGB_real = img[0,:,opt.idx_background[0],opt.idx_background[1]].t()
    loss = []

    for j in range(num_iter):
        optimizer.zero_grad()
        net_input = net_input_saved + (net_input_saved.normal_() * reg_noise_std)
        out = net(net_input)
        out = out[:, :, 0:img.shape[2], 0:img.shape[3]]
        total_loss = mse(out * mask, img * mask)
        loss.append(total_loss.detach())
        tmp = 0.05 * functions.RGB_1NN(out, opt, opt.idx_object)
        total_loss += tmp
        total_loss.backward()
        optimizer.step()
    out = out * (1-mask) + img * mask
    return out
