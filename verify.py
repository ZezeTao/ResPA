"""Implementation of evaluate attack result."""
import os
import torch
from torch.autograd import Variable as V
from torch import nn
from torchvision import transforms as T
from Normalize import Normalize, TfNormalize
from loader import ImageNet
from torch.utils.data import DataLoader
import pretrainedmodels

batch_size = 16

input_csv = './dataset/images.csv'
input_dir = './dataset/images'
adv_dir = './incv3_ResPA_outputs'

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def get_model(net_name, model_dir):
    """Load converted model"""
    model_path = os.path.join(model_dir, net_name + '.npy')

    if net_name == 'inception_v3':
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    torchvision.models.inception_v3(
                                        weights="DEFAULT").eval().cuda())


    elif net_name == 'densenet121':
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    torchvision.models.densenet121(pretrained=True).eval().cuda())

    elif net_name == 'resnet50':
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    torchvision.models.resnet50(pretrained=True).eval().cuda())

    elif net_name == 'vgg19':
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    torchvision.models.vgg19(pretrained=True).eval().cuda())

    elif net_name == 'resnet18':
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    torchvision.models.resnet101(pretrained=True).eval().cuda())



    elif net_name == 'inc_res_v2':
        model = torch.nn.Sequential(Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                    pretrainedmodels.inceptionresnetv2(num_classes=1000,
                                                                       pretrained='imagenet').eval().cuda())



    elif net_name == 'tf2torch_adv_inception_v3':
        from torch_nets import tf_adv_inception_v3
        net = tf_adv_inception_v3
        model = nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            TfNormalize('tensorflow'),
            net.KitModel(model_path).eval().cuda(), )
    elif net_name == 'tf2torch_ens3_adv_inc_v3':
        from torch_nets import tf_ens3_adv_inc_v3
        net = tf_ens3_adv_inc_v3
        model = nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            TfNormalize('tensorflow'),
            net.KitModel(model_path).eval().cuda(), )
    elif net_name == 'tf2torch_ens4_adv_inc_v3':
        from torch_nets import tf_ens4_adv_inc_v3
        net = tf_ens4_adv_inc_v3
        model = nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            TfNormalize('tensorflow'),
            net.KitModel(model_path).eval().cuda(), )
    elif net_name == 'tf2torch_ens_adv_inc_res_v2':
        from torch_nets import tf_ens_adv_inc_res_v2
        net = tf_ens_adv_inc_res_v2
        model = nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            TfNormalize('tensorflow'),
            net.KitModel(model_path).eval().cuda(), )
    else:
        print('Wrong model name!')

    return model


def verify(model_name, path):
    model = get_model(model_name, path)

    if model_name in ['inception_v3', 'tf2torch_adv_inception_v3', 'tf2torch_ens4_adv_inc_v3',
                      'tf2torch_ens3_adv_inc_v3', 'tf2torch_ens_adv_inc_res_v2']:
        img_size = 299
    else:
        img_size = 224
    transforms = T.Compose([T.Resize(img_size), T.ToTensor()])
    X = ImageNet(adv_dir, input_csv, transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)
    sum = 0
    for images, _, gt_cpu in data_loader:
        gt = gt_cpu.cuda()
        images = images.cuda()
        with torch.no_grad():
            sum += (model(images).argmax(1) != (gt)).detach().sum().cpu()

    print(model_name + '  acu = {:.2%}'.format(sum / 1000.0))


def verify_ensmodels(model_name, path):
    model = get_model(model_name, path)

    if model_name in ['inception_v3', 'tf2torch_adv_inception_v3', 'tf2torch_ens4_adv_inc_v3',
                      'tf2torch_ens3_adv_inc_v3', 'tf2torch_ens_adv_inc_res_v2']:
        img_size = 299
    else:
        img_size = 224
    transforms = T.Compose([T.Resize(img_size), T.ToTensor()])
    X = ImageNet(adv_dir, input_csv, transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)
    sum = 0
    for images, _, gt_cpu in data_loader:
        gt = gt_cpu.cuda()
        images = images.cuda()
        with torch.no_grad():
            # print(sum)
            sum += (model(images)[0].argmax(1) != (gt + 1)).detach().sum().cpu()

    print(model_name + '  acu = {:.2%}'.format(sum / 1000.0))


def main():
    model_names = ['inception_v3', "resnet50", "vgg19", 'densenet121']

    model_names_ens = ['tf2torch_ens4_adv_inc_v3', 'tf2torch_ens3_adv_inc_v3',
                       'tf2torch_ens_adv_inc_res_v2']
    models_path = './models/'
    for model_name in model_names:
        verify(model_name, models_path)
        print("===================================================")
    for model_name in model_names_ens:
        verify_ensmodels(model_name, models_path)
        print("===================================================")


if __name__ == '__main__':
    print("  ")
    print("  ")
    print("  ")
    print(adv_dir)
    main()

