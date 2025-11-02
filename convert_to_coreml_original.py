import argparse
import torch
from convert_to_onnx import load_model
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
import cv2
import os
from models.retinaface import RetinaFace
import torch.nn as nn
import torchvision
import onnxruntime as rt
# import ai_edge_torch
import coremltools as ct
from PIL import Image

from utils.box_utils import bounding_box_from_points_torch


def decode_landm_torch(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    _p = priors.unsqueeze(0).unsqueeze(2)
    _l = pre.reshape(pre.shape[0], pre.shape[1], -1, 2)
    #     print(_p.shape, _l.shape)

    landms = _p[..., 0:2] + _p[..., 2:4] * _l * variances[0]
    landms = landms.reshape(pre.shape[1], -1)
    return landms


def decode_torch(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    _p = priors.unsqueeze(0)

    boxes = torch.cat((
        _p[..., :2] + loc[..., :2] * variances[0] * _p[..., 2:],
        _p[..., 2:] * torch.exp(loc[..., 2:] * variances[1])), 2)

    # boxes[..., :2] = boxes[..., :2] - boxes[..., 2:] / 2
    # boxes[..., 2:] = boxes[..., 2:] + boxes[..., :2]

    boxes = torch.cat([boxes[..., :2] - boxes[..., 2:] / 2,
                       boxes[..., :2] + boxes[..., 2:] / 2], dim=2)

    return boxes


class RetinaStaticExportWrapper(nn.Module):
    def __init__(self, model, prior_box, config, bounding_box_from_points, return_mask=False):
        super(RetinaStaticExportWrapper, self).__init__()
        self.model = model
        self.color_scheme = config.get('color_scheme', 'BGR')
        self.variance = config['variance']
        # self.point_shape = model.point_count, model.point_dim
        self.point_shape = 5, 2
        self.point_count = np.prod(self.point_shape)

        self.top_k = config['top_k']
        self.confidence_threshold = config['confidence_threshold']
        self.nms_threshold = config['nms_threshold']

        self.mean = config.get('mean', [0.0, 0.0, 0.0])
        self.std = config.get('std', [1.0, 1.0, 1.0])

        assert self.color_scheme in ['RGB', 'BGR'], f"Wrong color scheme {self.color_scheme}"
        self.change_color_scheme = self.color_scheme != 'BGR'

        if self.mean == [0.0, 0.0, 0.0] and self.std == [1.0, 1.0, 1.0]:
            self.normalize = False
        else:
            self.normalize = True

        self.return_mask = return_mask

        if self.normalize:
            self.mean = torch.tensor(self.mean)
            self.std = torch.tensor(self.std)
            self.mean = self.mean[None, :, None, None]
            self.std = self.std[None, :, None, None]

            self.mean = torch.nn.Parameter(self.mean, requires_grad=False)
            self.std = torch.nn.Parameter(self.std, requires_grad=False)

        self.prior_box = prior_box
        self.bounding_box_from_points = bounding_box_from_points

    def forward(self, x):
        prior_box = self.prior_box
        x = torch.permute(x, (0, 3, 1, 2))

        if self.change_color_scheme:
            x = torch.flip(x, dims=[1])

        x = x.type(torch.float)

        if self.normalize:
            x = ((x - self.mean) / self.std)

        if self.bounding_box_from_points:
            _, conf, landms = self.model(x)
        else:
            (loc, conf, landms) = self.model(x)
            # (loc, conf, landms), mask_predicted = self.model(x)

        size_b, size_c, size_y, size_x = x.size()
        size_p, size_o = prior_box.size()
        coordinate_scale = torch.tensor([size_x, size_y]).view(1, 2).to(x.device)

        landms = decode_landm_torch(landms, prior_box, self.variance)
        landms = landms.reshape((size_b, size_p) + self.point_shape) * coordinate_scale
        landms = landms.reshape(size_b, size_p, self.point_count)

        if self.bounding_box_from_points:
            pass
        else:
            loc = decode_torch(loc, prior_box, self.variance)
            loc = loc.reshape((size_b, size_p) + (2, 2)) * coordinate_scale
            loc = loc.reshape(size_b, size_p, 4)

        if False:
            max_value, max_index = conf.max(dim=-1, keepdims=False)
            confidence_select = (max_value > self.confidence_threshold) * (max_index > 0)
            landms = landms[confidence_select, :]
            conf = conf[confidence_select, :]
            score = max_value[confidence_select]
        else:
            _score = conf[..., -1]
            confidence_select = _score > self.confidence_threshold
            landms = landms[confidence_select, :]
            conf = conf[confidence_select, :]
            score = confidence_select[confidence_select]

        if self.bounding_box_from_points:
            pass
        else:
            loc = loc[confidence_select]

        # NOT supported in ONNX opset 11  torch.argsort(score, descending=True)[:self.top_k]
        # _, top_k_select = score.sort(descending=True)
        # top_k_select = top_k_select[:self.top_k]
        # landms = landms[top_k_select]
        # conf = conf[top_k_select]
        # score = score[top_k_select]

        if self.bounding_box_from_points:
            loc = bounding_box_from_points_torch(landms, self.point_shape)
        else:
            loc = loc #[top_k_select]

        nms_select = torchvision.ops.nms(loc, score, self.nms_threshold)
        loc = loc[nms_select]
        landms = landms[nms_select]
        conf = conf[nms_select]
        score = score[nms_select]

        if self.return_mask:
            return conf, landms, loc, torch.softmax(mask_predicted, dim=1)
        else:
            return conf, landms, loc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--long_side', default=640, 
                        help='when origin_size is false, long_side is scaled size(320 or 640 for long side)')
    parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
    parser.add_argument('--test_image_path', 
                        default='./curve/ATC_2011_Graduation_Ceremony.jpg', help='Path to test image')
    parser.add_argument('--result_folder', action="store_true",
                        default='./__result', help='Folder to save the debug data')
    parser.add_argument('--bounding_box_from_points', action="store_true",
                        default=True, help='Construct bounding box from points')
    parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
    args = parser.parse_args()

    test_image_path = args.test_image_path
    long_side = args.long_side
    vis_thres = args.vis_thres
    result_folder = args.result_folder
    bounding_box_from_points = args.bounding_box_from_points
    bounding_box_from_points = False
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
        trained_model = './weights/mobilenet0.25_Final.pth'
    elif args.network == "resnet50":
        cfg = cfg_re50
        trained_model = './weights/Resnet50_Final.pth'

    # net and model
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    onnx_model_path = os.path.join(test_image_path, '', args.network + '.onnx')

    img_raw = cv2.imread(test_image_path)
    img_show = img_raw.copy()
    f_xy = (np.array(img_raw.shape) / long_side).max()
    img_raw_in = cv2.resize(img_raw, None, fx=1. / f_xy, fy=1. / f_xy)
    img_raw_in = cv2.copyMakeBorder(img_raw_in, 0, long_side-img_raw_in.shape[0],
                                    0, long_side - img_raw_in.shape[1],
                                    cv2.BORDER_CONSTANT)

    priorbox = PriorBox(cfg, image_size=(long_side, long_side))
    priors = priorbox.forward()
    priors = priors.to(device)

    export_config = {key: cfg[key] for key in ['variance']}
    export_config['nms_threshold'] = 0.35
    export_config['confidence_threshold'] = 0.02
    export_config['top_k'] = 512
    # export_config['color_scheme'] = 'BGR'
    # export_config['mean'] = (104, 117, 123)
    # flip = 1
    export_config['color_scheme'] = 'RGB'
    export_config['mean'] = (123, 104, 117)
    flip = -1

    export_model = RetinaStaticExportWrapper(net, priors, export_config, bounding_box_from_points)

    input_numpy = np.ascontiguousarray(img_raw_in[None, ..., ::flip])
    input_torch = torch.from_numpy(input_numpy)

    # predict_wrapper = export_model(input_torch)
    # os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)
    # torch.onnx.export(export_model, input_torch, onnx_model_path, export_params=True, verbose=False,
    #                   input_names=['input'], output_names=['output'], opset_version=11)

    # model_onnx = sess = rt.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    # input_name = sess.get_inputs()[0].name

    # predict_onnx = model_onnx.run(None, {input_name: input_numpy})

    if False:
        mlmodel_fp32_path = "retina_mobilenet_facedetector.mlmodel"

        trace_export_model_rgb = torch.jit.trace(export_model, input_torch)

        mlmodel_fp32 = ct.convert(
            trace_export_model_rgb,
            inputs=[ct.ImageType(name="input", shape=input_numpy.shape, channel_first=False)],
            outputs=[ct.TensorType(name="confidence"), ct.TensorType(name="landmarks"), ct.TensorType(name="box")],
            convert_to='neuralnetwork'
        )

        mlmodel_fp32.save(mlmodel_fp32_path)

        predict_onnx = mlmodel_fp32.predict({"input": Image.fromarray(img_raw_in[..., ::flip])})

        score_list = predict_onnx["confidence"][:, 1]
        landmarks_list = predict_onnx["landmarks"]
        box_list = predict_onnx["box"]

    elif False:
        mlmodel_fp32_path = "retina_mobilenet_facedetector.tflite"

        # Convert directly to TFLite
        edge_model = ai_edge_torch.convert(export_model, example_inputs=[input_torch])

        # Save the TFLite model
        edge_model.export(mlmodel_fp32_path)

        # Validate conversion
        # Load TFLite model and run inference
        interpreter = edge_model  # Can use directly or load from file

        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Prepare input
        input_data = input_torch.numpy() #.astype(np.float32)
        interpreter.set_input(0, input_data)

        # Run inference
        interpreter.invoke()

        # Get output
        tflite_output = interpreter.get_output(0)

    else:
        onnx_model_path = "retina_mobilenet_facedetector.onnx"

        torch.onnx.export(export_model, input_torch, onnx_model_path, export_params=True, verbose=False,
                          opset_version=17,
                          input_names=['input'], output_names=['confidence', 'landmarks', 'box'])

        model_onnx = sess = rt.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name

        predict_onnx = model_onnx.run(None, {input_name: input_numpy})

        print()

        # onnx_path = "temp_model.onnx"
        # torch.onnx.export(
        #     pytorch_model,
        #     dummy_input,
        #     onnx_path,
        #     input_names=['input'],
        #     output_names=['output'],
        #     opset_version=11
        # )

        # # Step 2: ONNX to TensorFlow
        # onnx_model = onnx.load(onnx_path)
        # tf_rep = onnx_tf.backend.prepare(onnx_model)
        # tf_model_path = "temp_tf_model"
        # tf_rep.export_graph(tf_model_path)

        # # Step 3: TensorFlow to TFLite
        # converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)

        # # Optional optimizations
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # # converter.target_spec.supported_types = [tf.float16]  # For FP16 quantization

        # tflite_model = converter.convert()

        # # Save TFLite model
        # with open(tflite_path, 'wb') as f:
        #     f.write(tflite_model)

        # print(f"Model successfully converted to {tflite_path}")


    for _score, _landm, _box in zip(score_list, landmarks_list, box_list):
        if _score < vis_thres:
            continue
        text = "{:.4f}".format(_score)
        _box = list(map(int, _box * f_xy + 0.5))
        _landm = list(map(int, _landm * f_xy + 0.5))
        cv2.rectangle(img_show, (_box[0], _box[1]), (_box[2], _box[3]), (0, 0, 255), 2)
        cx = _box[0]
        cy = _box[1] + 12
        cv2.putText(img_raw, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landms
        cv2.circle(img_show, (_landm[0], _landm[1]), 1, (0, 0, 255), 4)
        cv2.circle(img_show, (_landm[2], _landm[3]), 1, (0, 255, 255), 4)
        cv2.circle(img_show, (_landm[4], _landm[5]), 1, (255, 0, 255), 4)
        cv2.circle(img_show, (_landm[6], _landm[7]), 1, (0, 255, 0), 4)
        cv2.circle(img_show, (_landm[8], _landm[9]), 1, (255, 0, 0), 4)

    if bounding_box_from_points:
        out_file_path = os.path.join(result_folder, '', 'onnx_bounding_box_from_points.png')
    else:
        out_file_path = os.path.join(result_folder, '', 'onnx_original.png')

    os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
    cv2.imwrite(out_file_path, img_show)
