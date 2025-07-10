import torch
import torch
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch_dct as dct
import scipy.stats as st
from attacks.config import *
from cus_logits import *
import torchvision.transforms as transforms

img_max, img_min = 1., 0

def clamp(x, x_min, x_max):
    return torch.min(torch.max(x, x_min), x_max)
    
class Attack(object):
    """
    Base class for all attacks.
    """
    def __init__(self, attack, model, epsilon, targeted, random_start, norm, loss_type, mean, std, TI, resize_rate, di_prob, DI, ks, num_scale, num_admix, device=None):
        """
        Initialize the hyperparameters
        Arguments:
            attack (str): the name of attack.
            model (torch.nn.Module): the surrogate model for attack.
            epsilon (float): the perturbation budget.
            targeted (bool): targeted/untargeted attack.
            random_start (bool): whether using random initialization for delta.
            norm (str): the norm of perturbation, l2/linfty.
            loss (str): the loss function.
            device (torch.device): the device for data. If it is None, the device would be same as model
        """
        if norm not in ['l2', 'linfty']:
            raise Exception("Unsupported norm {}".format(norm))
        self.attack = attack
        self.model = model
        self.epsilon = epsilon
        self.targeted = targeted
        self.random_start = random_start
        self.norm = norm
        self.device = next(model.parameters()).device if device is None else device
        # self.loss = self.loss_function(loss_type)
        self.alpha = epsilon
        self.epoch = 10
        self.decay = 1.0
        self.mean = mean
        self.std = std
        self.TI = TI
        self.resize_rate = resize_rate
        self.di_prob = di_prob
        self.DI = DI
        self.ks = ks
        self.num_scale = num_scale
        self.num_admix = num_admix
    
    def gkern(kernlen=15, nsig=3):
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
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

    def forward(self, data, label, **kwargs):
        """
        The general attack procedure
        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        exp_settings = exp_configuration[578]
        
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        if exp_settings['Logit']=='LogitLoss':
            loss_fn=LogitLoss(label,exp_settings['targeted'])
        elif exp_settings['Logit']=='CELoss':
            loss_fn=CELoss(label)
        elif exp_settings['Logit']=='TopkLoss':
            loss_fn=TopkLoss(label, exp_settings['targeted'], exp_settings['top_k'], exp_settings['p'])
        elif exp_settings['Logit']=='Topkdis':
            loss_fn=Topkdis(label, exp_settings['targeted'], exp_settings['top_k'], exp_settings['p'])
    
        # Initialize adversarial perturbation
        delta = self.init_delta(data)
        trf_nor = transforms.Compose([
            transforms.Normalize(self.mean, self.std)
        ])
  
        momentum = 0.
        for _ in range(self.epoch):
            # data.requires_grad = True
            if self.DI:
                logits = self.get_logits(trf_nor(self.input_diversity(self.transform(data+delta, momentum=momentum))))
            else:
                logits = self.get_logits(trf_nor(self.transform(data+delta, momentum=momentum)))
            # Calculate the loss
            if exp_settings['Logit'] in ['Topkdis', 'Tgkdis']:  
                logits = logits.view(delta.size(0), self.num_scale*self.num_admix, logits.size(1))  
                logits_avg = logits.mean(dim=1)                 
                loss = loss_fn(delta, logits_avg)
            else:
                loss = loss_fn(logits)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)
            if self.TI:
                kernel = self.gkern(self.ks, 3).type(torch.float32)
                stack_kernel = torch.stack([kernel, kernel, kernel])
                stack_kernel = torch.unsqueeze(stack_kernel, 1)
                grad = F.conv2d(grad, stack_kernel, stride=1, padding='same', groups=3)
            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum,decay=self.decay)
            delta = self.update_delta(delta, data, momentum, self.alpha)
        adv_images = data + delta
        return adv_images.detach()

    def get_logits(self, x, **kwargs):
        """
        The inference stage, which should be overridden when the attack need to change the models (e.g., ensemble-model attack, ghost, etc.) or the input (e.g. DIM, SIM, etc.)
        """
        return self.model(x)

    def get_loss(self, logits, label):
        """
        The loss calculation, which should be overrideen when the attack change the loss calculation (e.g., ATA, etc.)
        """
        # Calculate the loss
        
        return self.loss(logits, label)
        
        

    def get_grad(self, loss, delta, **kwargs):
        """
        The gradient calculation, which should be overridden when the attack need to tune the gradient (e.g., TIM, variance tuning, enhanced momentum, etc.)
        """
        return torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]

    def get_momentum(self, grad, momentum, decay=None, **kwargs):
        """
        The momentum calculation
        """
        return momentum * decay + grad / (grad.abs().mean(dim=(1,2,3), keepdim=True))

    def init_delta(self, data, **kwargs):
        delta = torch.zeros_like(data).to(self.device)
        if self.random_start:
            if self.norm == 'linfty':
                delta.uniform_(-self.epsilon, self.epsilon)
            else:
                delta.normal_(-self.epsilon, self.epsilon)
                d_flat = delta.view(delta.size(0), -1)
                n = d_flat.norm(p=2, dim=10).view(delta.size(0), 1, 1, 1)
                r = torch.zeros_like(data).uniform_(0,1).to(self.device)
                delta *= r/n*self.epsilon
            delta = clamp(delta, img_min-data, img_max-data)
        delta.requires_grad = True
        return delta

    def update_delta(self, delta, data, grad, alpha, **kwargs):
        if self.norm == 'linfty':
            delta = torch.clamp(delta + alpha * grad.sign(), -self.epsilon, self.epsilon)
        else:
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1, 1)
            scaled_grad = grad / (grad_norm + 1e-20)
            delta = (delta + scaled_grad * alpha).view(delta.size(0), -1).renorm(p=2, dim=0, maxnorm=self.epsilon).view_as(delta)
        delta = clamp(delta, img_min-data, img_max-data)
        return delta




    def transform(self, data, **kwargs):
        return data

    def __call__(self, *input, **kwargs):
        self.model.eval()
        return self.forward(*input, **kwargs)
    
class MIFGSM(Attack):
    def __init__(self, model, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., targeted=False, random_start=False, 
                norm='linfty', loss_type='CE', device=None, attack='MI-FGSM',mean=None, std=None, TI=False, resize_rate=0.9, di_prob=0.7, DI=False, ks=5, num_scale=5, num_admix=3, **kwargs):
        super().__init__(attack, model, epsilon, targeted, random_start, norm, loss_type, mean, std, TI, resize_rate, di_prob, DI, ks, num_scale, num_admix, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay

class Admix(MIFGSM):
    def __init__(self, model, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., num_scale=5, num_admix=3, admix_strength=0.2, targeted=False, random_start=False, norm='linfty', loss_type='CE', device=None, attack='Admix', mean=None, std=None, TI=False, resize_rate=0.9, di_prob=0.7, DI=False, ks=5, **kwargs):
        super().__init__(model, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss_type, device, attack, mean, std, TI, resize_rate, di_prob, DI, ks, num_scale, num_admix, **kwargs)
        self.num_scale = num_scale
        self.num_admix = num_admix
        self.admix_strength = admix_strength
        
    def transform(self, x, **kwargs):
        """
        Admix the input for Admix Attack
        """
        admix_images = torch.concat([(x + self.admix_strength * x[torch.randperm(x.size(0))].detach()) for _ in range(self.num_admix)], dim=0)
        return torch.concat([admix_images / (2 ** i) for i in range(self.num_scale)])
