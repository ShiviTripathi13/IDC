import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default=0)
    parser.add_argument('--gpu_num', default=0, type=int, help='which gpu to send to')
    parser.add_argument('--diversity',help='diversity of the results, between 1 to 8', type=int, default=1)
    #load, input, save configurations:
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--nc_z',type=int,help='noise # channels',default=3)
    parser.add_argument('--nc_im',type=int,help='image # channels',default=3)
    parser.add_argument('--out',help='output folder',default='Output')
    parser.add_argument('--padding', default='pad_with_noise')
    #networks hyper parameters:
    parser.add_argument('--nfc', type=int, default=32)
    parser.add_argument('--min_nfc', type=int, default=32)
    parser.add_argument('--ker_size',type=int,help='kernel size',default=3)
    parser.add_argument('--num_layer',type=int,help='number of layers',default=5)
    parser.add_argument('--decrease_LR',type=int,help='number of layers',default=1600)
    parser.add_argument('--stride',help='stride',default=1)
    parser.add_argument('--padd_size',type=int,help='net pad size',default=0)
        
    #pyramid parameters:
    parser.add_argument('--scale_factor',type=float,help='pyramid scale factor',default=0.75)#pow(0.5,1/6))
    parser.add_argument('--noise_amp',type=float,help='addative noise cont weight',default=0.1)
    parser.add_argument('--min_size',type=int,help='image minimal size at the coarser scale',default=25)
    parser.add_argument('--max_size', type=int,help='image minimal size at the coarser scale', default=256)

    #optimization hyper parameters:
    parser.add_argument('--niter', type=int, default=2000, help='number of epochs to train per scale')
    parser.add_argument('--niter_mask', type=int, default=3000, help='number of epochs to train per scale')
    parser.add_argument('--gamma',type=float,help='scheduler gamma',default=0.1)
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--lr_mask', type=float, default=0.00005, help='learning rate, default=0.0005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--Gsteps',type=int, help='Generator inner steps',default=3)
    parser.add_argument('--Dsteps',type=int, help='Discriminator inner steps',default=3)
    parser.add_argument('--lambda_grad',type=float, help='gradient penalty weight',default=0.1)
    parser.add_argument('--alpha',type=float, help='reconstruction loss weight',default=10)



    parser.add_argument('--details',type=str,default='')
    parser.add_argument('--rec_loss', type=str, default='MSE')
    parser.add_argument('--num_scales_to_dilate_rec_loss', type=int, default=2)
    parser.add_argument('--scale_to_start_grad', type=int, default=0)
    parser.add_argument('--scale_to_start_masking', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)

    #remove before publishing
    parser.add_argument('--ratio', type=float, default=3, help='ratio of the mask compared to image size')
    return parser
