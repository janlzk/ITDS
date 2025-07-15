import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from attacks.ODIM.render import *
from attacks.ODIM.config import *
from scipy import stats as st
from cus_logits import *
from attacks.config import *


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
trf_nor = transforms.Compose([
    transforms.Normalize(mean, std)
])


class ODIM(object):

    def __init__(self, model, eps=8/255, alpha=2/255, steps=20, decay=1.0, kernel_name='gaussian', trf=trf_nor, DI=False, TI=False, resize_rate=0.9,
                 len_kernel=7, nsig=3, di_prob=0.7, config_idx_odi=101, config_idx=578):
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.di_prob = di_prob
        self.kernel_name = kernel_name
        self.len_kernel = len_kernel
        self.nsig = nsig
        self.stacked_kernel = self.kernel_generation()
        self.trf = trf
        self.DI = DI
        self.TI = TI
        self.config_idx_odi = config_idx_odi
        self.config_idx = config_idx
        self.model = model
        self.resize_rate = resize_rate
        self.device = next(model.parameters()).device

    def render_3d_aug_input(self, x_adv, renderer, prob=0.7):
        c = np.random.rand()
        if c <= prob:
            x_ri=x_adv.clone()
            for i in range(x_adv.shape[0]):
                x_ri[i]=renderer.render(x_adv[i].unsqueeze(0))
            return  x_ri 
        else:
            return  x_adv
    
    def forward(self, images, tg_labels):
        
        images = images.clone().detach()
        tg_labels = tg_labels.clone().detach()
        renderer=Render3D(config_idx=self.config_idx_odi, count=0)
        # loss = nn.CrossEntropyLoss(reduction=self.reduction)
        exp_settings=exp_configuration[self.config_idx]

        if exp_settings['Logit']=='TemperatureLoss':
            loss_fn=TemperatureLoss(tg_labels,exp_settings['targeted'],20)
        elif exp_settings['Logit']=='MarginLoss':
            loss_fn=MarginLoss(tg_labels,exp_settings['targeted'])
        elif exp_settings['Logit']=='AngleLoss':
            loss_fn=AngleLoss(self.model, tg_labels, exp_settings['targeted'])
        elif exp_settings['Logit']=='oriAngleLoss':
            loss_fn=oriAngleLoss(self.model, tg_labels, exp_settings['targeted'])
        elif exp_settings['Logit']=='LogitLoss':
            loss_fn=LogitLoss(tg_labels,exp_settings['targeted'])
        elif exp_settings['Logit']=='CELoss':
            loss_fn=CELoss(tg_labels)
        elif exp_settings['Logit']=='RandDecayLoss':
            loss_fn=RandDecayLoss(tg_labels, exp_settings['alpha'], exp_settings['targeted'])
        elif exp_settings['Logit']=='NormLoss':
            loss_fn=NormLoss(tg_labels, exp_settings['p'], exp_settings['targeted'])

        momentum = torch.zeros_like(images).detach().to(self.device)
        stacked_kernel = self.stacked_kernel.to(self.device)
        adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            # outputs = self.model(self.trf(self.render_3d_aug_input(adv_images, renderer, self.di_prob)))
            if self.DI:
                outputs = self.model(self.input_diversity(self.render_3d_aug_input(adv_images, renderer, self.di_prob)))
            else:
                outputs = self.model(self.render_3d_aug_input(adv_images, renderer, self.di_prob))
            cost = loss_fn(outputs)
            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
            # depth wise conv2d  
            if self.TI:
                g = F.conv2d(g, stacked_kernel, stride=1, padding=((self.len_kernel-1)//2,(self.len_kernel-1)//2), groups=3)
            # grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad / torch.norm(grad, p=1)
            grad = grad + momentum*self.decay
            momentum = grad
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

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


