import argparse
import torch
from convert_to_onnx import load_model
import numpy as np
from convert_to_onnx_original import RetinaStaticExportWrapper
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
import cv2
import os
import coremltools as ct
from models.retinaface import UNetRetinaConcat
import onnxruntime as rt
from coremltools.models.neural_network import quantization_utils
import tensorflow as tf
import ai_edge_torch


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
                        default=False, help='Construct bounding box from points')
    parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
    args = parser.parse_args()

    test_image_path = args.test_image_path
    long_side = args.long_side
    vis_thres = args.vis_thres
    result_folder = args.result_folder
    bounding_box_from_points = args.bounding_box_from_points
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
        trained_model = './weights/mobilenet0.25_Final.pth'
    elif args.network == "resnet50":
        cfg = cfg_re50
        trained_model = './weights/Resnet50_Final.pth'

    test_image_path = "./test_002.png"
    test_image_path = "./usa-1-4-dollar-1937.jpg"
    trained_model = "/home/ubuntu/projects/work/Pytorch_Retinaface_BBFree/weights/groupe_aligned/mobilenet0.25_epoch_12_mAP0.89158_F1_0.96458.pth"
    trained_model = "/home/ubuntu/projects/work/Pytorch_Retinaface_BBFree/weights/groupe_aligned_baseline/mobilenet0.25_2025-04-06T17:00:55.215359_Final.pth"
    # trained_model = "/home/ubuntu/projects/work/Pytorch_Retinaface_BBFree/weights/groupe_aligned/mobilenet0.25_epoch_38_mAP0.89796_F1_0.95604.pth"
    # trained_model = "/home/ubuntu/projects/work/Pytorch_Retinaface_BBFree/weights/groupe_aligned/mobilenet0.25_2025-04-10T18:29:25.319636_Final.pth"

    # Model Digit
    trained_model = "/home/ubuntu/projects/work/Pytorch_Retinaface_BBFree/weights/groupe_aligned_BEST_MULTICLASS/mobilenet0.25_2025-04-13T02:20:19.188001_Final.pth"
    trained_model = "/home/ubuntu/projects/work/Pytorch_Retinaface_BBFree/weights/groupe_aligned/mobilenet0.25_2025-04-19T10:28:52.454463_Final.pth"
    trained_model = "/home/ubuntu/projects/work/Pytorch_Retinaface_BBFree/weights/groupe_aligned-2025_22_05_16/mobilenet0.25_epoch_0066_mAP0.97520_F1_0.99217.pth"
    trained_model = "/home/ubuntu/projects/work/Pytorch_Retinaface_BBFree/weights/groupe_aligned-2025_23_05_10/mobilenet0.25_epoch_0041_mAP0.97222_F1_0.99140.pth"
    trained_model = "/home/ubuntu/projects/work/Pytorch_Retinaface_BBFree/weights/groupe_aligned-2025_25_05_13/mobilenet0.25_2025-05-25T13:20:14.053349_Final.pth"
    trained_model = "/home/ubuntu/projects/work/Pytorch_Retinaface_BBFree/weights/groupe_aligned-2025_26_05_09/mobilenet0.25_epoch_0102_mAP0.92810_F1_0.97656.pth"
    num_classes = 11
    use_batch_normalization = True

    # # Model Year
    # trained_model = "/home/ubuntu/projects/work/Pytorch_Retinaface_BBFree/weights/groupe_multiclass/mobilenet0.25_epoch_455_mAP0.92617_F1_0.94099.pth"
    # trained_model = "/home/ubuntu/projects/work/Pytorch_Retinaface_BBFree/weights/groupe_multiclass_UNetL_SX2/mobilenet0.25_epoch_1456_mAP0.73810_F1_0.81748.pth"
    # trained_model = "/home/ubuntu/projects/work/Pytorch_Retinaface_BBFree/weights/groupe_multiclass/mobilenet0.25_epoch_0385_mAP0.94267_F1_0.96002.pth"
    # trained_model = "/home/ubuntu/projects/work/Pytorch_Retinaface_BBFree/weights/groupe_multiclass-2025_23_05_21/mobilenet0.25_2025-05-23T21:05:02.217993_Final.pth"
    # trained_model = "/home/ubuntu/projects/work/Pytorch_Retinaface_BBFree/weights/groupe_multiclass-2025_24_05_22/mobilenet0.25_2025-05-24T22:44:30.407434_Final.pth"
    # trained_model = "/home/ubuntu/projects/work/Pytorch_Retinaface_BBFree/weights/groupe_multiclass-2025_27_05_19/mobilenet0.25_epoch_0088_mAP0.93624_F1_0.95288.pth"
    # num_classes = 4
    # use_batch_normalization = True

    # test_image_path = "./5297-original.jpg"
    # trained_model = "/home/ubuntu/projects/work/Pytorch_Retinaface_BBFree/weights/groupe/mobilenet0.25_epoch_35_mAP0.92347_F1_0.94949.pth"

    test_image_path = "./1989-memorial-o.jpeg"
    test_image_path = "./5297-original.jpg"
    deploy_path = os.path.dirname(trained_model)
    check_point_model_name = os.path.basename(trained_model)
    model_onnx_path = f'{deploy_path}/{check_point_model_name}.onnx'
    mlmodel_fp32_path = f'{deploy_path}/{check_point_model_name}_fp32.mlmodel'
    mlmodel_fp16_path = f'{deploy_path}/{check_point_model_name}_fp16.mlmodel'
    mlmodel_kmeans_8bit_path = f'{deploy_path}/{check_point_model_name}_kmeans_8bit.mlmodel'
    mlmodel_ls_8bit_path = f'{deploy_path}/{check_point_model_name}_ls_8bit.mlmodel'
    model_tflite_optimized = f'{deploy_path}/{check_point_model_name}_optimized.tflite'
    model_tflite_name_mapping = f'{deploy_path}/{check_point_model_name}_model_tflite_name_mapping.json'

    cfg['pretrain'] = False

    # net and model
    net = UNetRetinaConcat(cfg=cfg, phase='test', use_batch_normalization=use_batch_normalization, num_classes=num_classes)
    net = load_model(net, trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    img_raw = cv2.imread(test_image_path)
    img_raw = cv2.resize(img_raw, (640, 640))
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
    export_config['color_scheme'] = 'BGR'
    
    if use_batch_normalization:
        rgb_mean = (0, 0, 0)
    else:
        rgb_mean = (104, 117, 123) # bgr order

    export_config['mean'] = rgb_mean

    export_model = RetinaStaticExportWrapper(net, priors, export_config, bounding_box_from_points)
    export_model.eval()

    export_config['color_scheme'] = 'RGB'
    export_model_rgb = RetinaStaticExportWrapper(net, priors, export_config, bounding_box_from_points)
    export_model_rgb.eval()

    input_numpy = img_raw_in[None, ...]
    input_torch = torch.from_numpy(input_numpy)

    predict_wrapper = export_model(input_torch)
    torch.onnx.export(export_model, input_torch, model_onnx_path, export_params=True, verbose=False,
                      input_names=['input'], output_names=['output'], opset_version=11)

    model_onnx = sess = rt.InferenceSession(model_onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name

    predict_onnx = model_onnx.run(None, {input_name: input_numpy})

    for __score, _landm, _box in zip(*(item for item in predict_onnx)):
        _score = __score.max()

        if _score < vis_thres:
            continue
        text = "{:.4f}".format(_score)
        _box = list(map(int, _box * f_xy + 0.5))
        _landm = list(map(int, _landm * f_xy + 0.5))
        cv2.rectangle(img_show, (_box[0], _box[1]), (_box[2], _box[3]), (0, 0, 255), 2)
        cx = _box[0]
        cy = _box[1] + 12
        cv2.putText(img_show, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 0), thickness=2)

        # landms
        cv2.circle(img_show, (_landm[0], _landm[1]), 1, (0, 0, 255), 4)
        cv2.circle(img_show, (_landm[2], _landm[3]), 1, (0, 255, 255), 4)
        cv2.circle(img_show, (_landm[4], _landm[5]), 1, (255, 0, 255), 4)
        cv2.circle(img_show, (_landm[6], _landm[7]), 1, (0, 255, 0), 4)
        cv2.circle(img_show, (_landm[8], _landm[9]), 1, (255, 0, 0), 4)

    input_file_path = os.path.join(result_folder, '', 'onnx_input.png')

    if bounding_box_from_points:
        out_file_path = os.path.join(result_folder, '', 'onnx_bounding_box_from_points.png')
    else:
        out_file_path = os.path.join(result_folder, '', 'onnx_original.png')

    os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
    cv2.imwrite(input_file_path, img_raw)
    cv2.imwrite(out_file_path, img_show)

    # Export and Optimization CoreML
    trace_export_model_rgb = torch.jit.trace(export_model_rgb, input_torch)

    mlmodel_fp32 = ct.convert(
        trace_export_model_rgb,
        inputs=[ct.ImageType(name="input", shape=input_numpy.shape, channel_first=False)],
        outputs=[ct.TensorType(name="score"), ct.TensorType(name="landmarks"), ct.TensorType(name="box")],
        convert_to='neuralnetwork'
    )

    mlmodel_fp32.save(mlmodel_fp32_path)

    mlmodel_fp16 = quantization_utils.quantize_weights(mlmodel_fp32, nbits=16)
    mlmodel_fp16.save(mlmodel_fp16_path)

    # mlmodel_kmeans_8bit = quantization_utils.quantize_weights(mlmodel_fp32, nbits=8, quantization_mode="kmeans")
    # mlmodel_fp16.save(mlmodel_kmeans_8bit_path)

    # mlmodel_ls_8bit = quantization_utils.quantize_weights(mlmodel_fp32, nbits=8, quantization_mode="linear_symmetric")
    # mlmodel_fp16.save(mlmodel_ls_8bit_path)

    # # Export Tensorflow Lite
    # tfl_converter_flags = {'optimizations': [tf.lite.Optimize.DEFAULT]}
    # model_tflite = ai_edge_torch.convert(export_model_rgb, (input_torch.float(), ),
    #                                      _ai_edge_converter_flags=tfl_converter_flags)

    # # model_tflite_optimized = f'{check_point_model_path}.tflite' # Wrong path ???
    # model_tflite.export(model_tflite_optimized)

    print(model_onnx_path)
    print(mlmodel_fp32_path)
    print(mlmodel_fp16_path)
    print(mlmodel_kmeans_8bit_path)
    print(mlmodel_ls_8bit_path)
    print(model_tflite_optimized)
    print(model_tflite_name_mapping)
