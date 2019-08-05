import torch
from torch import nn

from .backbones.resnet import ResNet


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class PCBModel(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path,
                 local_conv_channel=256, num_stripes=6):
        super(PCBModel, self).__init__()
        self.base = ResNet(last_stride)
        self.base.load_param(model_path)
        self.num_classes = num_classes
        self.local_conv_channel = local_conv_channel
        self.num_stripes = num_stripes

        self.local_conv_list = nn.ModuleList()
        for _ in range(num_stripes):
            self.local_conv_list.append(nn.Sequential(
                nn.Conv2d(self.in_planes, local_conv_channel, 1),
                nn.AdaptiveAvgPool2d(1),
            ))

        self.local_bottleneck_list = nn.ModuleList()
        for _ in range(num_stripes):
            bnneck = nn.BatchNorm1d(local_conv_channel)
            bnneck.bias.requires_grad_(False)
            bnneck.apply(weights_init_kaiming)
            self.local_bottleneck_list.append(bnneck)

        self.local_fc_list = nn.ModuleList()
        for _ in range(num_stripes):
            cls = nn.Linear(local_conv_channel, num_classes, bias=False)
            cls.apply(weights_init_classifier)
            self.local_fc_list.append(cls)

    def load_param(self, model_path):
        param = torch.load(model_path)
        for i in param:
            if 'fc' in i: continue
            if i not in self.state_dict().keys(): continue
            if param[i].shape != self.state_dict()[i].shape: continue
            self.state_dict()[i].copy_(param[i])

    def forward(self, x):
        feat = self.base(x)
        assert feat.size(2) % self.num_stripes == 0
        stripe_h = int(feat.size(2) / self.num_stripes)

        local_feat_list, local_logits_list = [], []
        for i in range(self.num_stripes):
            local_feat = feat[:, :, i * stripe_h:(i + 1) * stripe_h, :]
            local_feat = self.local_conv_list[i](local_feat)
            local_feat = local_feat.view(local_feat.size(0), -1)
            local_feat_bn = self.local_bottleneck_list[i](local_feat)
            
            local_logits_list.append(self.local_fc_list[i](local_feat_bn))
            if self.training:
                local_feat_list.append(local_feat)
            else:
                local_feat_list.append(local_feat_bn)
        
        if self.training:
            return local_feat_list, local_logits_list
        return local_feat_list

