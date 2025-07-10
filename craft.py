import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import json
import torch
import torch.utils.data as Data
import torchvision.datasets as dsets
import torchvision.utils
from torchvision import models
import torchvision.transforms as transforms
from robustness.model_utils import make_and_restore_model
from robustness.datasets import ImageNet
from tqdm import tqdm
from attacks.itds import *
from attacks.tidim import *
from attacks.cfm import *
from attacks.config import *
from attacks.SIA.sia import *
from attacks.bsr import *
from attacks.admix import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_idx = json.load(open("eval_data/imagenet_class_index.json"))

idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
image_size = 224
num_workers = 64
config_idx = 578
exp_settings = exp_configuration[config_idx]
batch_size = 32

model_white_name = 'resnet50'
attacker = 'ITDS'   # ITDS CFM SIA Admix BSR DIM
tg_label = random.randint(0, 999)

targeted = True
tgimgs_path = exp_settings['tg_imgs_path'] # The image path corresponding to the target label
attack_type = exp_settings['attack_type']
eps = exp_settings['epsilon']
step_size = exp_settings['lr']
steps = exp_settings['steps']
di_prob = exp_settings['di_prob']
resize_rate = exp_settings['resize_rate']
DI_type = exp_settings['DI_type']
len_kernel = exp_settings['ks']
mix_prob = exp_settings['mix_prob']
eval_path = exp_settings['eval_path']
extract_path = exp_settings['extract_path']
mix_middle_type=exp_settings['mixed_image_type_feature']
fix_middle_type=exp_settings['blending_mode_feature']
mix_upper_bound_feature=exp_settings['mix_upper_bound_feature']
mix_lower_bound_feature=exp_settings['mix_lower_bound_feature']
channelwise=exp_settings['channelwise']
dataset = exp_settings['dataset']
fix_mode=exp_settings['fix_mode_input']
zeta=exp_settings['zeta']
nor=exp_settings['nor']
k = exp_settings['k']
drop = exp_settings['drop_ratio']
every_layer = exp_settings['every_layer']
logit = exp_settings['Logit']
rate = exp_settings['resize_rate']
alpha = exp_settings['alpha']
steps = exp_settings['steps']
norm = exp_settings['p']
itds_proportion = exp_settings['itds_proportion']
tg_samples = exp_settings['tg_samples']
tg_imgs = exp_settings['tg_imgs']
signmix = exp_settings['signmix']
mimix = exp_settings['mimix']
norm_mode = exp_settings['norm'] 
untg_steps = exp_settings['untg_steps']
noise_std = exp_settings['noise_std']
DI_mode = exp_settings['DI']
TI_mode = exp_settings['TI']
topk = exp_settings['top_k']
w = exp_settings['weight']
admix_portion = exp_settings['admix_portion']
num_admix = exp_settings['num_mix_samples']
loss_fn = 'logit'

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

trf = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

trf_nor = transforms.Compose([
    transforms.Normalize(mean, std)
])

def mkdir(path):
    """Check if the folder exists, if it does not exist, create it"""
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

def image_folder_custom_label(root, transform, custom_label):

    old_data = dsets.ImageFolder(root=root, transform=transform)
    old_classes = old_data.classes

    label2idx = {}

    for i, item in enumerate(idx2label):
        label2idx[item] = i

    new_data = dsets.ImageFolder(root=root, transform=transform,
                                 target_transform=lambda x: custom_label.index(old_classes[x]))
    new_data.classes = idx2label
    new_data.class_to_idx = label2idx

    return new_data

def get_adversarial_training_model(path):
    ds = ImageNet('/tmp')
    attack_model, _ = make_and_restore_model(arch='resnet50', dataset=ds, resume_path=path)
    model = attack_model.model
    return model

eval_data = image_folder_custom_label(root=eval_path, transform=trf, custom_label=idx2label)
eval_loader = Data.DataLoader(eval_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

image_count = sum(len(files) for _, _, files in os.walk(tgimgs_path))
if attacker in ['ITDS', 'ITDS_unsign']:
    tgimgs_data = image_folder_custom_label(root=tgimgs_path, transform=trf, custom_label=idx2label)
    tgimgs_loader = Data.DataLoader(tgimgs_data, batch_size=image_count, shuffle=True, pin_memory=True, num_workers=num_workers)
    for images, labels, _ in tgimgs_loader:
        tg_images = images.clone().detach()
        tg_labels = labels.clone().detach()

print('==> Loading model...')
if model_white_name in ['resnet50']:
    model = models.resnet50(pretrained=True).to(device).eval()
elif model_white_name in ['mobilenet']:    
    model = models.mobilenet_v3_large(pretrained=True).to(device).eval()
elif model_white_name in ['regnet']:
    model = models.regnet_y_32gf(pretrained=True).to(device).eval()
elif model_white_name in ['vgg16']:
    model = models.vgg16(pretrained=True).to(device).eval()

itds = ITDS(model=model, eps=eps/255, alpha=step_size/255, eta=itds_proportion, steps=steps, decay=1.0, m1=5, m2=tg_samples, di_prob=di_prob, resize_rate=resize_rate, len_kernel=len_kernel, trf=trf_nor, config_idx=config_idx, signmix=signmix, mimix=mimix, TI=TI_mode, DI=DI_mode, norm=norm_mode)
tidim = TIDIM(model=model, eps=eps/255, alpha=step_size/255, steps=steps, decay=1.0, trf=trf_nor, len_kernel=len_kernel, resize_rate=resize_rate, di_prob=di_prob, TI=TI_mode, DI=DI_mode, config_idx=config_idx)
sia = SIA(model, eps/255, step_size/255, steps, decay=1, targeted=targeted, ks=len_kernel, mean=mean, std=std, TI=TI_mode, loss_type=loss_fn)
bsr = BSR(model=model, epsilon=eps/255, alpha=step_size/255, epoch=steps, decay=1, targeted=targeted, ks=len_kernel, mean=mean, std=std, TI=TI_mode, resize_rate=resize_rate, di_prob=di_prob, DI=DI_mode, loss_type=loss_fn)
admix = Admix(model=model, epsilon=eps/255, alpha=step_size/255, epoch=steps, decay=1, targeted=targeted, ks=len_kernel, mean=mean, std=std, TI=TI_mode, resize_rate=resize_rate, di_prob=di_prob, DI=DI_mode, num_scale=5, num_admix=num_admix, admix_strength=admix_portion, loss_type=logit)

total = 0
count = 0
for index, (images, labels, paths) in enumerate(tqdm(eval_loader)):
    images = images.to(device)
    labels = labels.to(device)
    tg_labels = (torch.ones_like(labels) * tg_label).to(device)
    
    if attacker in ['CFM']:
        adv_images = advanced_fgsm(attack_type, model, images, labels, tg_labels, 
                                   num_iter=steps,max_epsilon=eps,step_size=step_size,DI_type=DI_type,di_prob=di_prob,resize_rate=resize_rate,
                                   kernel_size=len_kernel,mix_prob=mix_prob,count=count,config_idx=config_idx)  
    elif attacker in ['ITDS']:
        adv_images = itds(images, labels, tg_images, tg_labels)
    elif attacker in ['SIA']:
        adv_images = sia(images, tg_labels)
    elif attacker in ['BSR']:
        adv_images = bsr(images, tg_labels)
    elif attacker in ['DIM']:
        adv_images = tidim(images, tg_labels)
    elif attacker in ['Admix']:
        adv_images = admix(images, tg_labels)
    count = count + 1
    
    for i, adv_image in enumerate(adv_images):
        label = labels[i]
        folder_name = str(eval_data.classes[label])
        src_path = paths[i]
        filename = os.path.basename(src_path)
        path = f'./adv_examples/{attacker}_{tg_label}/{model_white_name}/{folder_name}/'
        mkdir(path)
        torchvision.utils.save_image(adv_image, path+filename)

