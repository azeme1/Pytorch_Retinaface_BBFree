import json

import torch
import numpy as np
import cv2
import ast
from skimage.transform import SimilarityTransform

from torch.utils.data import Dataset


class CustomDetectionDataset(Dataset):
    def __init__(self, dataframe, preprocess, transform=None):
        self.dataframe = dataframe
        self.preprocess = preprocess
        self.file_list = list(sorted(self.dataframe.frame_path.value_counts().keys()))
        self.transform = transform
        self.data_source = 'point'

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        frame_path = self.file_list[idx]
        rows = self.dataframe[self.dataframe.frame_path == frame_path]

        frame_data = cv2.imread(frame_path)

        annotations = np.zeros((0, 15))
        for point in rows[self.data_source]:
            annotation = np.full((1, 15), -1, dtype=np.float32)
            point = np.array(json.loads(point))
            x1, y1 = point.min(0)
            x2, y2 = point.max(0)

            # bbox
            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            pont_mean = point.mean(0, keepdims=True)

            annotation[0, 4] = point[0, 0]    # l0_x
            annotation[0, 5] = point[0, 1]    # l0_y
            annotation[0, 6] = point[1, 0]    # l1_x
            annotation[0, 7] = point[1, 1]    # l1_y
            annotation[0, 8] = pont_mean[0, 0]   # l2_x
            annotation[0, 9] = pont_mean[0, 1]   # l2_y
            annotation[0, 10] = point[2, 0]  # l3_x
            annotation[0, 11] = point[2, 1]  # l3_y
            annotation[0, 12] = point[3, 0]  # l4_x
            annotation[0, 13] = point[3, 1]  # l4_y

            if (annotation[0, 4] < 0):
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)

        target = np.array(annotations)

        if self.preprocess is not None:
            frame_data, mask_data, target = self.preprocess(frame_data, target)

        return torch.from_numpy(frame_data), mask_data, target


class GroupeDetectionDataset(CustomDetectionDataset):
    def __init__(self, dataframe, preprocess, transform=None):
        super().__init__(dataframe, preprocess, transform)
        self.data_source = 'groupe'


def points_from_box(box_item):
    x1, y1, x2, y2 = box_item
    p = np.array(((x1, y1), (x2, y1), (x2, y2), (x1, y2)), dtype=np.float32)
    return p


def get_bounding_box(points):
    x, y, w, h = cv2.boundingRect(np.array(points).astype(np.float32))
    return x, y, x + w - 1, y + h - 1


class GroupeAlignedDetectionDataset(GroupeDetectionDataset):
    def __init__(self, dataframe, preprocess, transform=None, multiclass=False, check=False, train=True):
        super().__init__(dataframe, preprocess, transform)

        self.transform = SimilarityTransform()
        self.s_dst = (640, 640)
        self.p_dst = np.array([[0.1, 0.25], [0.9, 0.25],
                               [0.9, 0.75], [0.1, 0.75]], dtype=np.float32) * np.array(self.s_dst)[None, ...]
        self.multiclass = multiclass
        self.check = check
        self.train = train

    def __getitem__(self, idx):

        frame_path = self.file_list[idx]
        rows = self.dataframe[self.dataframe.frame_path == frame_path]

        _idx = np.random.randint(len(rows))
        frame_path = rows.iloc[_idx].frame_path

        frame_data = cv2.imread(frame_path, cv2.IMREAD_COLOR_BGR)

        annotations = np.zeros((0, 15))

        if self.data_source in rows.columns:
            s_dst = inference_shape = (640, 640)

            if self.train:
                s = np.random.uniform(0.05, 0.20)
                p_dst_year_first = np.array([[0.0 + s, 0.25 + s], [1.0 - s, 0.25 + s],
                                            [1.0 - s, 0.75 - s], [0.1 + s, 0.75 - s]], dtype=np.float32) * np.array(self.s_dst)[None, ...]
            else:
                p_dst_year_first = self.p_dst

            p_src = np.array(ast.literal_eval(rows[self.data_source].iloc[_idx]))
            if len(p_src.shape) != 3:
                p_src = p_src[None, ...]

            p_src = p_src[np.random.randint(len(p_src)), ...]
            # self.transform.estimate(p_src, self.p_dst)
            self.transform.estimate(p_src[[0, 1, -2, -1], :], p_dst_year_first[[0, 1, -2, -1], :])
            M = self.transform.params
            # frame_patch = cv2.warpPerspective(frame_data, M, self.s_dst)
            frame_patch = cv2.warpPerspective(frame_data, M, self.s_dst)

            p_list_src = np.array(ast.literal_eval(rows['point'].iloc[_idx]), dtype=np.float32)
            p_list_dst = cv2.perspectiveTransform(p_list_src, M)
            label_list = ast.literal_eval(rows.label_list.iloc[_idx])
        else:
            frame_patch = frame_data
            p_list_dst = np.array(ast.literal_eval(rows['point'].iloc[_idx]), dtype=np.float32)
            label_list = ast.literal_eval(rows.label_list.iloc[_idx])

            s_dst = inference_shape = (640, 640)
            if self.train:
                if np.random.choice([True, False], p=[0.2, 0.8]):
                    s = np.random.uniform(0.20, 0.50)
                    p_dst_year_first = np.array([[0.0 + s, 0.0 + s], [1.0 - s, 0.0 + s], [0.5, 0.5],
                                                [1.0 - s, 1.0 - s], [0.0 + s, 1.0 - s]], dtype=np.float32) * np.array(s_dst)[None, ...]

                    transform = SimilarityTransform()
                    p_src = points_from_box(get_bounding_box(p_list_dst.reshape(-1, 2)))
                    p_src = np.concatenate([p_src[:2], p_src.mean(0, keepdims=True), p_src[-2:]], 0)
                    transform.estimate(p_src[[0, 1, -2, -1], :], p_dst_year_first[[0, 1, -2, -1], :])
                    M_m = transform.params
                    frame_patch = cv2.warpPerspective(frame_data, M_m, inference_shape, borderMode=cv2.BORDER_REPLICATE)
                    p_list_dst = cv2.perspectiveTransform(p_list_dst.reshape(1, -1, 2), M_m).reshape(p_list_dst.shape)
                else:
                    p_dst_year_first = self.p_dst
                    frame_patch = frame_data
            else:
                p_dst_year_first = self.p_dst
                frame_patch = frame_data

        for label, point in zip(label_list, p_list_dst[:, [0, 1, -2, -1], :]):
            annotation = np.full((1, 15), -1, dtype=np.float32)

            x1, y1 = point.min(0)
            x2, y2 = point.max(0)

            x1 = min(max(x1, 0), frame_patch.shape[1])
            x2 = min(max(x2, 0), frame_patch.shape[1])

            y1 = min(max(y1, 0), frame_patch.shape[0])
            y2 = min(max(y2, 0), frame_patch.shape[0])

            # bbox
            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            pont_mean = point.mean(0, keepdims=True)

            annotation[0, 4] = min(max(point[0, 0], 0), frame_patch.shape[1])    # l0_x
            annotation[0, 5] = min(max(point[0, 1], 0), frame_patch.shape[0])    # l0_y
            annotation[0, 6] = min(max(point[1, 0], 0), frame_patch.shape[1])    # l1_x
            annotation[0, 7] = min(max(point[1, 1], 0), frame_patch.shape[0])    # l1_y
            annotation[0, 8] = min(max(pont_mean[0, 0], 0), frame_patch.shape[1])  # l2_x
            annotation[0, 9] = min(max(pont_mean[0, 1], 0), frame_patch.shape[0])  # l2_y
            annotation[0, 10] = min(max(point[2, 0], 0), frame_patch.shape[1])   # l3_x
            annotation[0, 11] = min(max(point[2, 1], 0), frame_patch.shape[0])   # l3_y
            annotation[0, 12] = min(max(point[3, 0], 0), frame_patch.shape[1])   # l4_x
            annotation[0, 13] = min(max(point[3, 1], 0), frame_patch.shape[0])   # l4_y

            if self.multiclass:
                label = 1 + label
            else:
                label = 1

            if (annotation[0, 4] < 0):
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = label

            annotations = np.append(annotations, annotation, axis=0)

        target = np.array(annotations)

        if self.preprocess is not None:
            frame_patch, mask_data, target = self.preprocess(frame_patch, target)

        # except Exception as e:
        #     print(e)

        return torch.from_numpy(frame_patch), mask_data, target, idx


class GroupeAlignedMulticlassDetectionDataset(GroupeAlignedDetectionDataset):
    def __init__(self, dataframe, preprocess, transform=None, multiclass=False, check=False, train=True):
        super().__init__(dataframe, preprocess, transform, multiclass, check, train)
        self.p_dst = np.array([[0.0, 0.0], [1.0, 0.0],
                               [1.0, 1.0], [0.0, 1.0]], dtype=np.float32) * np.array(self.s_dst)[None, ...]
