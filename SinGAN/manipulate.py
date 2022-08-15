from __future__ import print_function
import torch.utils.data
import imageio
from SinGAN.training import *

def generate_gif(Gs,Zs,reals,NoiseAmp,opt,alpha=0.1,beta=0.9,start_scale=2,fps=10,erode_mask=False,masks=None):
    in_s = torch.full(Zs[0].shape, 0, device=opt.device)
    images_cur = []
    count = 0
    for G,Z_opt,noise_amp,real in zip(Gs,Zs,NoiseAmp,reals):
        pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
        if opt.padding == 'pad_with_noise':
            nzx = real.shape[2] + (opt.ker_size - 1) * (opt.num_layer)
            nzy = real.shape[3] + (opt.ker_size - 1) * (opt.num_layer)
            pad_noise = 0
            m_noise = nn.ZeroPad2d(int(pad_noise))
            m_image = nn.ZeroPad2d(int(pad_image))
        else:
            nzx = Z_opt.shape[2]
            nzy = Z_opt.shape[3]
            m_image = nn.ZeroPad2d(int(pad_image))
        images_prev = images_cur
        images_cur = []
        functions.calc_mask(masks[count], reals[count][:, 0:3, :, :], opt, folder='')
        if erode_mask:
            tmp = 2 * (opt.mask_background[:, 0:3, :, :] - 0.5)
            opt.mask_background = functions.dilate_mask(tmp, opt, pad_image, 0)
            opt.mask_object = (1 - opt.mask_background).abs()
        if count == 0:
            z_rand = functions.generate_noise([1,nzx,nzy], device=opt.device)
            z_rand = z_rand.expand(1,3,Z_opt.shape[2],Z_opt.shape[3])
            z_prev1 = 0.95*Z_opt +0.05*z_rand
            z_prev2 = Z_opt
        else:
            z_prev1 = 0.95*Z_opt +0.05*functions.generate_noise([opt.nc_z,nzx,nzy], device=opt.device)
            z_prev2 = Z_opt

        for i in range(0,100,1):
            if count == 0:
                z_rand = functions.generate_noise([1,nzx,nzy], device=opt.device)
                z_rand = z_rand.expand(1,3,Z_opt.shape[2],Z_opt.shape[3])
                diff_curr = beta*(z_prev1-z_prev2)+(1-beta)*z_rand
            else:
                diff_curr = beta*(z_prev1-z_prev2)+(1-beta)*(functions.generate_noise([opt.nc_z,nzx,nzy], device=opt.device))

            z_curr = alpha*Z_opt+(1-alpha)*(z_prev1+diff_curr)
            z_curr = Z_opt*m_image(opt.mask_background) + z_curr * m_image(1-opt.mask_background)
            z_prev2 = z_prev1
            z_prev1 = z_curr

            if images_prev == []:
                I_prev = in_s
            else:
                I_prev = images_prev[i]
                I_prev = imresize(I_prev, 1 / opt.scale_factor, opt)
                I_prev = I_prev[:, :, 0:real.shape[2], 0:real.shape[3]]
                I_prev = m_image(I_prev)
            if count < start_scale:
                z_curr = Z_opt

            z_in = (noise_amp*z_curr+I_prev).type(torch.FloatTensor).to(opt.device)
            I_curr = G(z_in.detach(),I_prev.type(torch.FloatTensor)).to(opt.device)

            if (count == len(Gs)-1):
                I_curr = functions.denorm(I_curr).detach()
                I_curr = I_curr[0,:,:,:].cpu().numpy()
                I_curr = I_curr.transpose(1, 2, 0)*255
                I_curr = I_curr.astype(np.uint8)

            images_cur.append(I_curr)
        count += 1
    dir2save = functions.generate_dir2save(opt)
    try:
        os.makedirs('%s/start_scale=%d' % (dir2save,start_scale) )
    except OSError:
        pass
    imageio.mimsave('%s/start_scale=%d/alpha=%f_beta=%f.gif' % (dir2save,start_scale,alpha,beta),images_cur,fps=fps)
    del images_cur

def SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt,in_s=None,scale_v=1,scale_h=1,n=0,gen_start_scale=0,num_samples=50,folder_to_save='',diversity=1,masks=None):
    if opt.mode == 'train':
        dir2save = '%s/RandomSamples/%s/gen_start_scale=%d' % (opt.out, opt.input_name[:-4], gen_start_scale)
    else:
        dir2save = functions.generate_dir2save(opt)
    if (diversity < 1) | (diversity > 8):
        print("diversity should be between 1 and 8" )
        if diversity > 8:
            diversity = 8
        elif diversity < 1:
            diversity = 1
    erode_val = 8 - diversity #diversity should be between 1-8.
    dir2save += '/diversity_%d' % (diversity)
    if not os.path.exists(dir2save):
        os.makedirs(dir2save)
    if not folder_to_save=='':
        dir2save = folder_to_save
    if in_s is None:
        in_s = torch.full(reals[0].shape, 0, device=opt.device)

    images_cur = []
    masks_sample = masks.copy()
    for scale in range(len(Gs)-1,-1,-1):
        mask_tmp = masks[scale].clone()
        tmp = (2 * ((1-mask_tmp) - 0.5)).repeat(1,3,1,1)
        mask_background = functions.dilate_mask(tmp, opt, erode_val, 0)
        masks_sample[scale] = (1 - mask_background).to(opt.device)
    if erode_val == 7:
        mask_merge = masks[-1].repeat(1,3,1,1)
    else:
        mask_tmp = masks_sample[0].clone()
        tmp_mask = (2 * (mask_tmp - 0.5))
        mask_object = functions.dilate_mask(tmp_mask, opt, 7, 0).cpu()
        mask_tmp = functions.resize_and_tresh_mask(mask_object[0,0,:,:], (1 / opt.scale_factor)** opt.stop_scale)
        mask_merge = torch.from_numpy(mask_tmp[:reals[-1].shape[2],:reals[-1].shape[3]]).repeat((1,3,1,1)).to(opt.device)
    tmp = functions.convert_image_np(reals[-1])
    idx = torch.where(mask_merge[0,0,:,:].cpu() == 1)
    tmp[idx[0],idx[1]] = 0
    idx = torch.where(masks[-1].cpu() == 1)
    tmp[idx[0], idx[1]] = 0
    plt.imsave('%s/real_final_masked.png' % dir2save, tmp)
    for G,Z_opt,noise_amp in zip(Gs,Zs,NoiseAmp):
        pad1 = int((opt.ker_size-1)*opt.num_layer)/2
        m_image = nn.ZeroPad2d(int(pad1))
        if opt.padding == 'pad_with_noise':
            pad1 = 0
        nzx = (Z_opt.shape[2]-pad1*2)*scale_v
        nzy = (Z_opt.shape[3]-pad1*2)*scale_h
        m_noise = nn.ZeroPad2d(int(pad1))
        images_prev = images_cur
        images_cur = []
        for i in range(0,num_samples,1):
            if n == 0:
                z_curr = functions.generate_noise([1,nzx,nzy], device=opt.device)
                z_curr = z_curr.expand(1,3,z_curr.shape[2],z_curr.shape[3])
                z_curr = m_noise(z_curr)
            else:
                z_curr = functions.generate_noise([opt.nc_z,nzx,nzy], device=opt.device)
                z_curr = m_noise(z_curr)
            z_curr = (1-m_image(masks_sample[n])) * Z_opt + m_image(masks_sample[n]) * z_curr
            z_curr = z_curr.type(torch.FloatTensor).to(opt.device)
            if images_prev == []:
                I_prev = m_image(in_s)
            else:
                I_prev = images_prev[i]
                I_prev = imresize(I_prev,1/opt.scale_factor, opt)
                I_prev = I_prev[:, :, 0:round(scale_v * reals[n].shape[2]), 0:round(scale_h * reals[n].shape[3])]
                I_prev = m_image(I_prev)
                I_prev = I_prev[:,:,0:z_curr.shape[2],0:z_curr.shape[3]]
                I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])

            if n < gen_start_scale:
                z_curr = Z_opt
            z_in = noise_amp*(z_curr)+I_prev
            I_curr = G(z_in.detach(),I_prev)
            images_cur.append(I_curr)
        n+=1

    functions.calc_mask(masks[-1], reals[-1][:, 0:3, :, :], opt, folder='')
    mask_merge[:,:,opt.idx_object[0],opt.idx_object[1]] = 1
    mask_to_dilate = 2*(mask_merge[:,0:3,:,:]-0.5)
    mask_gaus = functions.dilate_mask(mask_to_dilate, opt,5).to(opt.device)
    mask_gaus[:,:,opt.idx_object[0],opt.idx_object[1]] = 1

    for i in range(0, num_samples, 1):
        I_curr_tmp = images_cur[i]
        I_curr_tmp = I_curr_tmp * mask_gaus + reals[-1][:,0:3,:,:] * (1-mask_gaus)
        plt.imsave('%s/%d.png' % (dir2save, i), functions.convert_image_np(I_curr_tmp), vmin=0,vmax=1)
    return images_cur
