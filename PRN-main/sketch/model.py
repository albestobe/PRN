import torch
import torch.optim as optim
from utils.io import load_ckpt
from utils.io import save_ckpt
from torchvision.utils import make_grid
from torchvision.utils import save_image
from modules.RFRNet import RFRNet, VGG16FeatureExtractor
import os
import time
# from visdom import Visdom


class RFRNetModel():
    def __init__(self):
        self.G = None
        self.lossNet = None
        self.iter = None
        self.optm_G = None
        self.device = None
        self.real_A = None
        self.real_B = None
        self.fake_B = None
        self.comp_B = None
        self.l1_loss_val = 0.0
    
    def initialize_model(self, path=None, train=True):
        self.G = RFRNet()
        self.optm_G = optim.Adam(self.G.parameters(), lr = 2e-4)
        if train:
            self.lossNet = VGG16FeatureExtractor()
        try:
            start_iter = load_ckpt(path, [('generator', self.G)], [('optimizer_G', self.optm_G)])
            if train:
                self.optm_G = optim.Adam(self.G.parameters(), lr = 2e-4)
                print('Model Initialized, iter: ', start_iter)
                self.iter = start_iter
        except:
            print('No trained model, from start')
            self.iter = 0
        
    def cuda(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Model moved to cuda")
            self.G.cuda()
            if self.lossNet is not None:
                self.lossNet.cuda()
        else:
            self.device = torch.device("cpu")
        
    def train(self, train_loader, save_path, finetune = False, iters=180000):
    #    writer = SummaryWriter(log_dir="log_info")
        self.G.train(finetune = finetune)
        if finetune:
            self.optm_G = optim.Adam(filter(lambda p:p.requires_grad, self.G.parameters()), lr = 5e-5)
        print("Starting training from iteration:{:d}".format(self.iter))
        s_time = time.time()
        while self.iter<iters:
            for items in train_loader:
                gt_images, gt_images_color, masks = self.__cuda__(*items)
                masked_images = gt_images * masks
                masked_images_color = gt_images_color * masks
                input_images=torch.cat((masked_images,masked_images_color),1)
                masks=torch.cat((masks,masks),1)
                self.forward(input_images, masks, masked_images_color, gt_images_color)
                self.update_parameters()
                self.iter += 1
                
                if self.iter % 50 == 0:
                    e_time = time.time()
                    int_time = e_time - s_time
                    print("Iteration:%d, l1_loss:%.4f, time_taken:%.2f" %(self.iter, self.l1_loss_val/50, int_time))
                    s_time = time.time()
                    self.l1_loss_val = 0.0
                
                if self.iter % 20000 == 0:
                    if not os.path.exists('{:s}'.format(save_path)):
                        os.makedirs('{:s}'.format(save_path))
                    save_ckpt('{:s}/g_{:d}.pth'.format(save_path, self.iter ), [('generator', self.G)], [('optimizer_G', self.optm_G)], self.iter)
        if not os.path.exists('{:s}'.format(save_path)):
            os.makedirs('{:s}'.format(save_path))
            save_ckpt('{:s}/g_{:s}.pth'.format(save_path, "final"), [('generator', self.G)], [('optimizer_G', self.optm_G)], self.iter)
    def test(self, test_loader, result_save_path):
        # viz = Visdom()
        self.G.eval()
        for para in self.G.parameters():
            para.requires_grad = False
        count = 0
        for items in test_loader:
            gt_images,gt_images_color, masks= self.__cuda__(*items)
            #masks = torch.cat([masks]*3, dim = 1)
            masks = masks[:, 0:3, :, :]
            masked_images = gt_images * masks
            masked_images_color =gt_images_color * masks
            input_images=torch.cat((masked_images,masked_images_color),1)
            masks3=masks
            masks=torch.cat((masks,masks),1)
            fake_B, mask = self.G(input_images, masks)
            masks1=masks[:,0:1,:,:]
            comp_B = fake_B * (1 - masks3) + gt_images_color * masks3
            # viz.images(torch.clamp(comp_B*255, 0, 255), win='c', env='main',opts=dict(title='Sketch Map Restoration'))
            if not os.path.exists('{:s}/results'.format(result_save_path)):
                os.makedirs('{:s}/results'.format(result_save_path))
            for k in range(comp_B.size(0)):
                count += 1
                # print(comp_B[k:k+1].shape)
                grid = make_grid(comp_B[k:k + 1])
                # print(grid.shape)
                file_path = '{:s}/results/img_{:d}.png'.format(result_save_path, count)
                save_image(grid, file_path)

                grid = make_grid(fake_B[k:k + 1])
                file_path = '{:s}/results/fake_img_{:d}.png'.format(result_save_path, count)
                save_image(grid, file_path)

                grid = make_grid(masked_images[k:k + 1] + 1 - masks3[k:k + 1])
                file_path = '{:s}/results/masked_img_{:d}.png'.format(result_save_path, count)
                save_image(grid, file_path)

                grid = make_grid(1 - masks1[k:k + 1])
                file_path = '{:s}/results/masks{:d}.png'.format(result_save_path, count)
                save_image(grid, file_path)

                grid = make_grid(masked_images_color[k:k + 1] + 1 - masks3[k:k + 1])
                file_path = '{:s}/results/masked_img_color{:d}.png'.format(result_save_path, count)
                save_image(grid, file_path)



            # if not os.path.exists('{:s}/cartoonresult'.format(result_save_path)):
            #     os.makedirs('{:s}/cartoonresult'.format(result_save_path))
            # if not os.path.exists('{:s}/originalimage'.format(result_save_path)):
            #     os.makedirs('{:s}/originalimage'.format(result_save_path))
            # if not os.path.exists('{:s}/originalmask'.format(result_save_path)):
            #     os.makedirs('{:s}/originalmask'.format(result_save_path))
            # for k in range(comp_B.size(0)):
            #     count += 1
            #
            #     grid = make_grid(comp_B[k:k+1])
            #     file_path = '{:s}/cartoonresult/{:d}.png'.format(result_save_path, count)
            #     save_image(grid, file_path)
            #
            #     grid = make_grid(gt_images[k:k+1])
            #     file_path = '{:s}/originalimage/{:d}.png'.format(result_save_path, count)
            #     save_image(grid, file_path)
            #
            #     grid = make_grid(1-masks1[k:k+1] )
            #     file_path = '{:s}/originalmask/{:d}.png'.format(result_save_path, count)
            #     save_image(grid, file_path)
    
    def forward(self, masked_image, mask,masked_images_color, gt_image):
        self.real_A = masked_images_color
        self.real_B = gt_image
        fake_B, _ = self.G(masked_image, mask)
        self.fake_B = fake_B
        self.mask=mask[:,0:3,:,:]
        mask3=mask[:,0:3,:,:]
        self.comp_B = self.fake_B * (1 - mask3) + self.real_B * mask3
    
    def update_parameters(self):
        self.update_G()
        self.update_D()
    
    def update_G(self):
        self.optm_G.zero_grad()
        loss_G = self.get_g_loss()
        loss_G.backward()
        self.optm_G.step()
    
    def update_D(self):
        return
    
    def get_g_loss(self):
        real_B = self.real_B
        fake_B = self.fake_B
        comp_B = self.comp_B
        
        real_B_feats = self.lossNet(real_B)
        fake_B_feats = self.lossNet(fake_B)
        comp_B_feats = self.lossNet(comp_B)
        
        tv_loss = self.TV_loss(comp_B * (1 - self.mask))
        style_loss = self.style_loss(real_B_feats, fake_B_feats) + self.style_loss(real_B_feats, comp_B_feats)
        preceptual_loss = self.preceptual_loss(real_B_feats, fake_B_feats) + self.preceptual_loss(real_B_feats, comp_B_feats)
        valid_loss = self.l1_loss(real_B, fake_B, self.mask)
        hole_loss = self.l1_loss(real_B, fake_B, (1 - self.mask))
        
        loss_G = (  tv_loss * 0.1
                  + style_loss * 120
                  + preceptual_loss * 0.05
                  + valid_loss * 1
                  + hole_loss * 6)
        
        self.l1_loss_val += valid_loss.detach() + hole_loss.detach()
        return loss_G
    
    def l1_loss(self, f1, f2, mask = 1):
        return torch.mean(torch.abs(f1 - f2)*mask)
    
    def style_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            _, c, w, h = A_feat.size()
            A_feat = A_feat.view(A_feat.size(0), A_feat.size(1), A_feat.size(2) * A_feat.size(3))
            B_feat = B_feat.view(B_feat.size(0), B_feat.size(1), B_feat.size(2) * B_feat.size(3))
            A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))
            B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))
            loss_value += torch.mean(torch.abs(A_style - B_style)/(c * w * h))
        return loss_value
    
    def TV_loss(self, x):
        h_x = x.size(2)
        w_x = x.size(3)
        h_tv = torch.mean(torch.abs(x[:,:,1:,:]-x[:,:,:h_x-1,:]))
        w_tv = torch.mean(torch.abs(x[:,:,:,1:]-x[:,:,:,:w_x-1]))
        return h_tv + w_tv
    
    def preceptual_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            loss_value += torch.mean(torch.abs(A_feat - B_feat))
        return loss_value
            
    def __cuda__(self, *args):
        return (item.to(self.device) for item in args)
            