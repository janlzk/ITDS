import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import json
import torch
import torch.utils.data as Data
import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from robustness.model_utils import make_and_restore_model
from robustness.datasets import ImageNet
import timm


test_path = 'adv_examples/ITDS_99/rn50'
image_size = 224
batch_size = 128
num_workers = 64
tg_label = 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_idx = json.load(open("./eval_data/imagenet_class_index.json"))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

trf = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
trf_nor = transforms.Compose([
    transforms.Normalize(mean, std)
])

def imwrite(img, title, path):
    torchvision.utils.save_image(img, path + str(title) + ".png")

def mkdir(path):
    """Check if the folder exists, if it does not exist, create it"""
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

def image_folder_custom_label(root, transform, custom_label):  # 这里是做index和label的转换

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

# Todo: print('==>Loading pretrained model...')
print('==>Loading model...')

# # normal trained model (CNNs)
model_resnet50 = models.resnet50(pretrained=True).to(device).eval()
model_inceptionv3 = models.inception_v3(pretrained=True).to(device).eval()
model_resnet101 = models.resnet101(pretrained=True).to(device).eval()
model_mobilenet = models.mobilenet_v3_large(pretrained=True).to(device).eval()
model_regnet = models.regnet_y_32gf(pretrained=True).cuda().eval()
model_vgg16 = models.vgg16(pretrained=True).to(device).eval()
model_vgg19 = models.vgg19(pretrained=True).to(device).eval()
model_densenet161 = models.densenet161(pretrained=True).to(device).eval()
model_efficientnet = models.efficientnet_b7(pretrained=True).cuda().eval()

# # normal trained model (ViTs)
model_vit = models.vit_b_16(pretrained=True).cuda().eval()
model_deit = timm.create_model('deit_base_patch16_224', pretrained=True).cuda().eval()
model_convit = timm.create_model('convit_base', pretrained=True).cuda().eval()
model_pit = timm.create_model('pit_b_224', pretrained=True).cuda().eval()

# # adversarial trained defense model
# model_defense_lf_4 = get_adversarial_training_model('./defense_models/imagenet_linf_4.pt').cuda().eval()
# model_defense_lf_8 = get_adversarial_training_model('./defense_models/imagenet_linf_8.pt').cuda().eval()
# model_defense_l2_3 = get_adversarial_training_model('./defense_models/imagenet_l2_3_0.pt').cuda().eval()


# model_eval = [model_resnet50, model_defense_lf_4, model_defense_lf_8, model_defense_l2_3]
# model_eval_name = ['resnet50', 'defense_lf4', 'defense_lf8', 'defense_l2_3']

model_eval = [model_resnet50, model_vgg16, model_mobilenet, model_regnet, model_inceptionv3, model_resnet101, model_densenet161, model_efficientnet, model_vit, model_deit, model_convit, model_pit]
model_eval_name = ['resnet50', 'vgg16', 'mobilenet', 'regnet', 'inceptionv3', 'resnet101', 'densenet161', 'efficient',  'vit', 'deit', 'convit', 'pit']

print(test_path)
test_data = image_folder_custom_label(root=test_path, transform=trf, custom_label=idx2label)
test_loader = Data.DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)


for i in range(len(model_eval)):
    total = 0
    correct = 0
    for images, labels, _ in test_loader:
        adv_images = images.to(device)
        labels = labels.to(device)
        tg_labels = (torch.ones_like(labels) * tg_label).to(device)
        with torch.no_grad():
            output = model_eval[i](trf_nor(adv_images))
            _, adv_pre = torch.max(output.data, 1)
        total += adv_images.shape[0]
        correct += (adv_pre == tg_labels).sum()
    print(model_eval_name[i] + '\'s TASR is: %f %%' % (100 * float(correct / total)))
print('\n'+ str(total) + '\n')  