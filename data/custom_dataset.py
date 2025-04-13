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
        self.file_list = list(self.dataframe.frame_path.value_counts().keys())
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


class GroupeAlignedDetectionDataset(GroupeDetectionDataset):
    def __init__(self, dataframe, preprocess, transform=None, multiclass=False):
        super().__init__(dataframe, preprocess, transform)

        self.transform = SimilarityTransform()
        self.s_dst = (512, 512)
        self.p_dst = np.array([[0.1, 0.25], [0.9, 0.25],
                               [0.9, 0.75], [0.1, 0.75]], dtype=np.float32) * np.array(self.s_dst)[None, ...]
        self.multiclass = multiclass

    def __getitem__(self, idx):
        frame_path = self.file_list[idx]
        rows = self.dataframe[self.dataframe.frame_path == frame_path]

        frame_data = cv2.imread(frame_path)

        annotations = np.zeros((0, 15))

        p_src = np.array(ast.literal_eval(rows[self.data_source].iloc[0]))
        self.transform.estimate(p_src, self.p_dst)
        M = self.transform.params
        frame_patch = cv2.warpPerspective(frame_data, M, self.s_dst)

        p_list_src = np.array(ast.literal_eval(rows['point'].iloc[0]), dtype=np.float32)
        p_list_dst = cv2.perspectiveTransform(p_list_src, M)
        label_list = ast.literal_eval(rows.label_list.iloc[0])

        for label, point in zip(label_list, p_list_dst):
            annotation = np.full((1, 15), -1, dtype=np.float32)

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

        return torch.from_numpy(frame_patch), mask_data, target


class GroupeAlignedMulticlassDetectionDataset(GroupeAlignedDetectionDataset):
    def __init__(self, dataframe, preprocess, transform=None, multiclass=False):
        super().__init__(dataframe, preprocess, transform)
        self.p_dst = np.array([[0.0, 0.0], [1.0, 0.0],
                               [1.0, 1.0], [0.0, 1.0]], dtype=np.float32) * np.array(self.s_dst)[None, ...]