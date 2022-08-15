
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions
import time
from datetime import datetime
from config import get_arguments

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--input_mask', help='input mask name',default='')
    parser.add_argument('--mode', help='task to be done', default='train')
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Ds = []
    Zs = []
    reals = []
    NoiseAmp = []


    real = functions.read_image(opt)
    functions.adjust_scales2image(real, opt)
    now = datetime.now()
    opt.num_scales_to_enlarge_alpha = np.max((1,int(opt.stop_scale / 3)))
    functions.load_mask(opt)
    opt.scale_to_start_masking = functions.calc_scale_to_start_masking(opt,real)
    opt.scale_to_start_grad = opt.scale_to_start_masking
    dir2save = functions.generate_dir2save(opt)

    if (os.path.isfile('%s/Gs.pth' % dir2save)):
        Gs, Ds, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
    plt.imsave('%s/mask.png' % dir2save,opt.mask_original_image.cpu(),cmap='gray')
    file_name = dir2save + r'/inf.txt'
    opt.file_name = file_name
    if os.path.isfile(file_name):
        f = open(file_name,"a")
    else:
        f = open(file_name,"w")

    f.write(str(now.strftime("%d/%m/%Y %H:%M:%S")))
    f.write("\n")
    f.write(dir2save)
    f.write("\n")
    f.write(str(opt))
    f.write("\n")
    f.close()
    print(dir2save)
    print('Number of scales = %d' % (opt.stop_scale))
    opt.dir2save=dir2save
    t = time.time()
    reals, masks = train(opt, Gs,Ds, Zs, reals, NoiseAmp)
    elapsed = time.time() - t
    print('Training time = %s' % (elapsed))
    opt.mode = 'random_samples'
    opt.gen_start_scale = 0
    images_cur = SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt,masks=masks,diversity=opt.diversity)



