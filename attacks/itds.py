import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from scipy import stats as st
from cus_logits import *
from attacks.config import *

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
trf_nor = transforms.Compose([
    transforms.Normalize(mean, std)
])

def imwrite(img, title, path):
    torchvision.utils.save_image(img, path + str(title) + ".png")


class ITDS(object):
    def __init__(self, model, eps=16/255, alpha=1.6/255, eta=1.0, steps=100, decay=1.0, m1=5, m2=3, trf=trf_nor, kernel_name='gaussian', len_kernel=7, nsig=3, di_prob=0.7, resize_rate=0.9, untg_steps=10, signmix=True, mimix=True, TI=False, DI=False, norm=True, config_idx=578):
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.eta = eta
        self.m1 = m1
        self.m2 = m2
        self.trf = trf
        self.model = model
        self.config_idx = config_idx
        self.di_prob = di_prob
        self.resize_rate = resize_rate
        self.kernel_name = kernel_name
        self.len_kernel = len_kernel
        self.nsig = nsig
        self.stacked_kernel = self.kernel_generation()
        self.signmix = signmix
        self.untg_steps = untg_steps
        self.TI = TI
        self.DI = DI
        self.norm = norm
        self.mimix = mimix
        self.device = next(model.parameters()).device


    def forward(self, images, labels, tg_images, tg_labels):
        
        images = images.clone().detach()
        labels = labels.clone().detach()
        tg_images = tg_images.clone().detach()
        tg_labels = tg_labels.clone().detach()
        exp_settings=exp_configuration[self.config_idx]
        untg_loss = LogitLoss(tg_labels, exp_settings['targeted'])

        if exp_settings['Logit']=='LogitLoss':
            loss_fn=LogitLoss(tg_labels, exp_settings['targeted'])
        elif exp_settings['Logit']=='CELoss':
            loss_fn=CELoss(tg_labels)
        elif exp_settings['Logit']=='TopkLoss':
            loss_fn=TopkLoss(tg_labels, exp_settings['targeted'], exp_settings['top_k'], exp_settings['p'])
        elif exp_settings['Logit']=='Topkdis':
            loss_fn=Topkdis(tg_labels, exp_settings['targeted'], exp_settings['top_k'], exp_settings['p'])
        
        momentum = torch.zeros_like(images).detach().to(self.device)
        momentum_tg = torch.zeros(self.m2, images.size(0), images.size(1), images.size(2), images.size(3)).detach().to(self.device)
        stacked_kernel = self.stacked_kernel.to(self.device)
        adv_images = images.clone().detach()
        rand_tgimgs = torch.zeros(self.m2, images.size(0), images.size(1), images.size(2), images.size(3)).detach().to(self.device)
        rand_tgimgs_adv = torch.zeros(self.m2, images.size(0), images.size(1), images.size(2), images.size(3)).detach().to(self.device)
     
        for t in range(self.steps):
            if t % self.untg_steps == 0:
                for k in range(self.m2):
                    random_indices = torch.randperm(tg_images.size(0))[:images.size(0)]
                    rand_tgimgs[k] = tg_images[random_indices]
                    rand_tgimgs_adv[k] = rand_tgimgs[k]
            adv_images.requires_grad = True
            g = torch.zeros_like(images).detach().to(self.device)

            for k in range(self.m2):
                rand_tgimg = rand_tgimgs_adv[k]
                rand_tgimg.requires_grad = True
                if self.DI:
                    outputs_tgimg = self.model(self.trf(self.input_diversity(rand_tgimg)))
                else:
                    outputs_tgimg = self.model(self.trf(rand_tgimg))
                # if exp_settings['Logit'] in ['Topkdis', 'Tgkdis']:                        
                #     cost_tg = loss_fn(rand_tgimg, outputs_tgimg)
                # else:
                #     cost_tg = loss_fn(outputs_tgimg)
                cost_tg = untg_loss(outputs_tgimg)
                g_tg = torch.autograd.grad(cost_tg, rand_tgimg, retain_graph=False, create_graph=False)[0]
                if self.TI:
                    g_tg = F.conv2d(g_tg, stacked_kernel, stride=1, padding=((self.len_kernel-1)//2,(self.len_kernel-1)//2), groups=3)
                if self.norm:
                    grad_tg = g_tg / torch.norm(g_tg, p=1, dim=[1,2,3], keepdim=True)
                    grad_tg = grad_tg + momentum_tg[k] * self.decay
                else:
                    grad_tg = g_tg + momentum_tg[k] * self.decay
                momentum_tg[k] = grad_tg        
                rand_tgimgs_adv[k] = rand_tgimgs_adv[k].detach() - self.alpha * grad_tg.sign()
                delta_tg = torch.clamp(rand_tgimgs_adv[k] - rand_tgimgs[k], min=-self.eps, max=self.eps)
                rand_tgimgs_adv[k] = torch.clamp(rand_tgimgs[k] + delta_tg, min=0, max=1).detach()

                for i in range(self.m1):
                    if self.signmix:
                        if self.mimix:
                            r_adv = (adv_images - self.eta * self.alpha * grad_tg.sign()) / (2**i)
                        else:
                            r_adv = (adv_images - self.eta * self.alpha * g_tg.sign()) / (2**i)
                    else:
                        if self.mimix:
                            r_adv = (adv_images - self.eta * grad_tg) / (2**i)
                        else:
                            r_adv = (adv_images - self.eta * g_tg) / (2**i)
                    r_adv = torch.clamp(r_adv, min=0, max=1)
                    if self.DI:
                        outputs_adv = self.model(self.trf(self.input_diversity(r_adv)))
                    else:
                        outputs_adv = self.model(self.trf(r_adv))

                    if exp_settings['Logit'] in ['Topkdis', 'Tgkdis']:                        
                        cost_adv = loss_fn(adv_images, outputs_adv)
                    else:
                        cost_adv = loss_fn(outputs_adv)
                    
                    # grad_nes = torch.autograd.grad(cost_adv, r_adv, retain_graph=False, create_graph=False)[0]  # Compute gradient for blended input
                    grad_nes = torch.autograd.grad(cost_adv, adv_images, retain_graph=False, create_graph=False)[0]   # Compute gradient for original input              
                    g = g + grad_nes
            g = g / (self.m1 * self.m2)
            if self.TI:
                g = F.conv2d(g, stacked_kernel, stride=1, padding=((self.len_kernel-1)//2,(self.len_kernel-1)//2), groups=3)
            if self.norm:
                grad = g / torch.norm(g, p=1, dim=[1,2,3], keepdim=True)
                grad = grad + momentum * self.decay
            else:
                grad = g + momentum * self.decay
            momentum = grad
            adv_images = adv_images.detach() + self.alpha * grad.sign()            
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        return adv_images
    
    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).type(torch.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).type(torch.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).type(torch.float32)
        else:
            raise NotImplementedError
        stack_kernel = torch.stack([kernel, kernel, kernel])
        stack_kernel = torch.unsqueeze(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = torch.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kern1d = torch.tensor(kern1d)
        kernel_raw = torch.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = torch.ones((kernlen, kernlen)) * 1.0 / (kernlen*kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1-torch.abs(torch.linspace((-kernlen+1)/2, (kernlen-1)/2, kernlen)/(kernlen+1)*2)
        kernel_raw = torch.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel
    
    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)
        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]
        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left
        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
        return padded if torch.rand(1) < self.di_prob else x

    def __call__(self, *input, **kwargs):
        adv_images = self.forward(*input, **kwargs)
        return adv_images
