import time
from options.train_options import TrainOptions
from util.visualizer3d import Visualizer
from data import create_dataset
from models import create_model
import torch
import numpy as np

def MAE_(fake,real):
    mae = 0.0
    mae = np.mean(np.abs(fake-real))
    return mae

def Norm(a):
    max_ = torch.max(a)
    min_ = torch.min(a)
    a_0_1 = (a-min_)/(max_-min_)
    return (a_0_1-0.5)*2

opt = TrainOptions().parse()

dataset = create_dataset(opt, phase="train")  # create a dataset given opt.dataset_mode and other options
dataset_size = len(dataset)
print('#training images = %d' % dataset_size)

val_dataset = create_dataset(opt, phase="val") 
val_dataset_size = len(val_dataset)
print('#validation images = %d' % val_dataset_size)

model = create_model(opt)
model.setup(opt)             
visualizer = Visualizer(opt)
total_steps = 0
val_total_iters = 0 

global_mae = 100000000000000
for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
    epoch_start_time = time.time()
    iter_start_time = time.time()
    epoch_iter = 0
    if "adap" in opt.name:
        model.update_weight_alpha()
    for i, data in enumerate(dataset):
        
        total_steps += opt.batch_size
        epoch_iter += opt.batch_size
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)
            if "grad" in opt.name:
                grads = model.get_current_grads()
                visualizer.plot_current_grads(epoch, float(epoch_iter)/dataset_size, opt, grads)
                visualizer.print_current_grads(epoch, epoch_iter, grads)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batch_size
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)
    
    if epoch % opt.val_epoch_freq == 0: 
        device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        with torch.no_grad():
            MAE = 0
            num = 0
            for i, data in enumerate(val_dataset):
                AtoB = opt.direction == 'AtoB'
                real_A = data['A' if AtoB else 'B'].to(device,dtype=torch.float)
                real_B = data['B' if AtoB else 'A'].to(device,dtype=torch.float).detach().cpu().numpy()
                print(real_A.shape, real_B.shape)
                real_A_proj = Norm(torch.mean(real_A,3)) #torch.Size([1, 1, 256, 256])
                fake_B = model.netG(real_A).detach().cpu().numpy() 
                mae = MAE_(fake_B,real_B)
                MAE += mae
                num += 1

            print ('Val MAE:',MAE/num)
            if MAE/num < global_mae:
                global_mae = MAE/num
                # Save best models checkpoints
                print('saving the current best model at the end of epoch %d, iters %d' % (epoch, total_steps))
                model.save('best')
                model.save(epoch)
                print("saving best...")
            visualizer.print_current_metrics(epoch, MAE/num)
          

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

    if epoch > opt.n_epochs:
        model.update_learning_rate()

#python train3d.py --dataroot /home/slidm/OCTA/octa-500/OCT2OCTA3M_3D --name new_p2p_3D_pm_2g_seg_st_correctfix_seed7 --model TransPro --netG unet_256 --direction AtoB --lambda_A 10 --lambda_C 5 --dataset_mode alignedoct2octa3d --norm batch --pool_size 0 --load_size 304 --input_nc 1 --output_nc 1 --display_port 6031 --gpu_ids 0 --no_flip --seed 7