import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

from models.net import MobileNetV1 as MobileNetV1
from models.net import FPN as FPN
from models.net import SSH as SSH
from models.unet import UNet_X1



class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 10)

class RetinaFace(nn.Module):
    def __init__(self, cfg = None, phase = 'train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace,self).__init__()
        self.phase = phase
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if cfg['pretrain']:
                import os
                _folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                checkpoint_path = f"{_folder}/weights/mobilenetV1X0.25_pretrain.tar"
                if os.path.exists(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path, 
                                            map_location=torch.device('cpu'))
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in checkpoint['state_dict'].items():
                        name = k[7:]  # remove module.
                        new_state_dict[name] = v
                    # load params
                    backbone.load_state_dict(new_state_dict)
        elif cfg['name'] == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=cfg['pretrain'])

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list,out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def forward(self,inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output


class UNetRetina(RetinaFace):
    def __init__(self, cfg=None, phase='train', num_classes=2):
        super().__init__(cfg, phase)
        self.unet = UNet_X1(3, 1)
        self.dropout2d = torch.nn.Dropout2d(p=0.5)

    def forward(self, x):
        y = self.unet(x)
        x = self.dropout2d(x)
        y = super().forward(torch.sigmoid(y) * x)
        return y


class ClassHeadMultiClass(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3, num_classes=2):
        super(ClassHeadMultiClass, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * self.num_classes,
                                 kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, self.num_classes)


class UNetRetinaConcat(RetinaFace):
    def __init__(self, cfg=None, phase='train', use_batch_normalization=False, num_classes=2):
        self.use_batch_normalization = use_batch_normalization
        self.num_classes = num_classes
        super().__init__(cfg, phase)
        self.unet = UNet_X1(3, 3)
        self.body.stage1[0][0] = torch.nn.Conv2d(6, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.dropout2d = torch.nn.Dropout2d(p=0.05)
        self.batch_normalization = torch.nn.BatchNorm2d(3, momentum=0.01)

    def forward(self, x):
        if self.use_batch_normalization:
            x = self.batch_normalization(x)
        y = self.unet(x)

        x = torch.cat([x, torch.softmax(y, dim=1)], dim=1)

        # if np.random.choice([True, False], p=[0.1, 0.9]):
        #     y = y.detach()

        x = self.dropout2d(x)
        x = super().forward(x)
        # x = super().forward(torch.sigmoid(torch.mean(y, 1, keepdim=True)) * x)
        return x, y

    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHeadMultiClass(inchannels, anchor_num, self.num_classes))
        return classhead
