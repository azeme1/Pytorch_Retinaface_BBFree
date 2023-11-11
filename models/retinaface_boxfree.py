import torch
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict
from models.retinaface import RetinaFace
from models.net import MobileNetV1 as MobileNetV1
from models.net import FPN as FPN
from models.net import SSH as SSH


class RetinaFaceBoxFree(RetinaFace):
    def __init__(self, cfg=None, phase='train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFaceBoxFree, self).__init__()
        self.phase = phase
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if cfg['pretrain']:
                checkpoint = torch.load("./weights/mobilenetV1X0.25_pretrain.tar",
                                        map_location=torch.device('cpu'))
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
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

    def forward(self, inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (classifications, ldm_regressions)
        else:
            output = (F.softmax(classifications, dim=-1), ldm_regressions)
        return output
