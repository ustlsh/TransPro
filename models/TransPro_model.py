import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util3d as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks3d
from . import networks_2g_st
from . import networks
from unet import UNet
import torch.nn.functional as F




class TransProModel(BaseModel):
    def name(self):
        return 'TransProModel'

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.isTrain = opt.isTrain

        # define 3D Generator
        self.netG = networks_2g_st.generator3D(opt.ngf)
        self.netG.weight_init(mean=0.0, std=0.02)
        self.netG.to(device=self.device)
        
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            # define 3D Discriminator and 2D Discriminator
            self.netD = networks3d.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, use_sigmoid)
            self.netD_proj = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # load 2D Generator for HCG module
            self.netG_t = networks_2g_st.generator(opt.ngf).to(self.device)
            self.netG_t.load_state_dict(torch.load("./pretrain_weights/hcg.pth")) 
            for p in self.netG_t.parameters():
                p.requires_grad=False
            # load 2D segmentation model for VPG module
            self.net = UNet(n_channels=1, n_classes=2, bilinear=False)
            self.net.load_state_dict(torch.load("./pretrain_weights/vpg.pth", map_location=self.device)) 
            self.net.to(device=self.device)
            for p in self.net.parameters():
                p.requires_grad=False
            
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)
                self.load_network(self.netD_proj, 'D_proj', opt.which_epoch)
        
        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks_2g_st.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam([
                {'params': self.netD.parameters(), 'lr': opt.lr, 'betas': (opt.beta1, 0.999)},
                {'params': self.netD_proj.parameters(), 'lr': opt.lr, 'betas': (opt.beta1, 0.999)}
            ])
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            
        print('---------- Networks initialized -------------')
        networks3d.print_network(self.netG)
        if self.isTrain:
            networks3d.print_network(self.netD)
            networks3d.print_network(self.netD_proj)
        print('-----------------------------------------------')
        
        

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].permute(1,0,2,3,4).to(self.device,dtype=torch.float) #torch.Size([1, 1, 256, 256, 256])
        self.real_B = input['B' if AtoB else 'A'].permute(1,0,2,3,4).to(self.device,dtype=torch.float)
        self.real_A_proj = torch.mean(self.real_A,3) #torch.Size([1, 1, 256, 256])
        self.real_A_proj = self.Norm(self.real_A_proj)
        self.real_B_proj = torch.mean(self.real_B,3)
        self.real_B_proj = self.Norm(self.real_B_proj)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG.forward(self.real_A) # torch.Size([1, 1, 256, 256, 256])
        self.fake_B_proj_t = self.netG_t.forward(self.real_A_proj)
        self.fake_B_proj_s = torch.mean(self.fake_B,3)
        self.fake_B_proj_s = self.Norm(self.fake_B_proj_s)


    # no backprop gradients
    def test(self):
        with torch.no_grad():
            self.fake_B = self.netG.forward(self.real_A)
        

    # get image paths
    def get_image_paths(self):
        #return "blksdf"
        return self.image_paths

    def backward_D(self):
        # Fake
        # fake_AB [1,2,256,256,256]
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        self.pred_fake = self.netD.forward(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        # Fake proj: [1,2,256,256]
        fake_AB_proj = torch.cat((self.real_A_proj, self.fake_B_proj_s), 1)  
        self.pred_fake_proj = self.netD_proj(fake_AB_proj.detach()) 
        self.loss_D_fake_proj = self.criterionGAN(self.pred_fake_proj, False)
        


        # Real
        # real_AB [1,2,256,256,256]
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        self.pred_real = self.netD.forward(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        # real_AB_proj [1,2,256,256]
        real_AB_proj = torch.cat((self.real_A_proj, self.real_B_proj), 1)
        self.pred_real_proj = self.netD_proj(real_AB_proj)
        self.loss_D_real_proj = self.criterionGAN(self.pred_real_proj, True)
        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_fake_proj + self.loss_D_real + self.loss_D_real_proj) * 0.5
    
        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD.forward(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        fake_AB_proj = torch.cat((self.real_A_proj, self.fake_B_proj_s), 1)
        pred_fake_proj = self.netD_proj(fake_AB_proj)
        self.loss_G_GAN_proj = self.criterionGAN(pred_fake_proj, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A

        self.fake_B_seg = F.softmax(self.net((self.fake_B_proj_s+1)/2), dim=1)[0] # 
        self.real_B_seg = F.softmax(self.net((self.real_B_proj+1)/2), dim=1)[0]

        # Third, proj(G(A)) = proj(B)
        self.loss_G_L1_pm = self.criterionL1(self.fake_B_proj_s, self.real_B_proj) * self.opt.lambda_C
        self.loss_G_L1_pm_st = self.criterionL1(self.fake_B_proj_s, self.fake_B_proj_t) * self.opt.lambda_C
        self.loss_G_L1_seg = self.criterionL1(self.fake_B_seg, self.real_B_seg) * self.opt.lambda_C
        
        self.loss_G = self.loss_G_GAN + self.loss_G_GAN_proj + self.loss_G_L1 + self.loss_G_L1_pm + self.loss_G_L1_pm_st + self.loss_G_L1_seg

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.set_requires_grad(self.netD_proj, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.set_requires_grad(self.netD, False) 
        self.set_requires_grad(self.netD_proj, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.item()),
                            ('G_L1', self.loss_G_L1.item()),
                            ('G_GAN_proj', self.loss_G_GAN_proj.item()),
                            ('G_L1_pm', self.loss_G_L1_pm.item()),
                            ('G_L1_pm_st', self.loss_G_L1_pm_st.item()),
                            ('G_L1_seg', self.loss_G_L1_seg.item()),
                            ('D_real_proj', self.loss_D_real_proj.item()),
                            ('D_fake_proj', self.loss_D_fake_proj.item()),
                            ('D_real', self.loss_D_real.item()),
                            ('D_fake', self.loss_D_fake.item())
                            ])

    def get_current_visuals(self):
        real_A = util.tensor2im3d(self.real_A.data)
        fake_B = util.tensor2im3d(self.fake_B.data)
        real_B = util.tensor2im3d(self.real_B.data)
        if self.isTrain:
            fake_B_proj = util.tensor2im(self.fake_B_proj_s.data)
            real_B_proj = util.tensor2im(self.real_B_proj.data)
            fake_B_seg = util.mask2im(self.fake_B_seg)
            real_B_seg = util.mask2im(self.real_B_seg)
            return OrderedDict([('fake_B', fake_B), ('real_B', real_B), ('fake_B_proj', fake_B_proj), ('real_B_proj', real_B_proj), ('fake_B_seg', fake_B_seg), ('real_B_seg', real_B_seg)])
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
        self.save_network(self.netD_proj, 'D_proj', label, self.gpu_ids)

    def Norm(self, a):
        max_ = torch.max(a)
        min_ = torch.min(a)
        a_0_1 = (a-min_)/(max_-min_)
        return (a_0_1-0.5)*2