import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retinaface Points to Box and Box to Points')
    parser.add_argument('--training_dataset', default='./data/widerface/train/label.txt',
                        help='Training dataset directory')

    args = parser.parse_args()
    training_dataset = args.training_dataset

    with open(training_dataset, 'r') as f:
        lines = f.readlines()

    print(f"Total Face Records :: {len(lines)}")

    line_list = np.array([np.array(line.strip().split(' ')).astype(np.float32)
                          for line in lines if (not line.startswith('#'))])

    print(f"Records with Landmarks :: {len(line_list)}")

    line_list_filtered = line_list[line_list[:, 5] > 0, :]

    box_list = line_list_filtered[:, [0, 1, 2, 3]].reshape(-1, 2, 2)
    hw_box_list = line_list_filtered[:, [2, 3]].reshape(-1, 2)
    landmark_list = line_list_filtered[:, [4, 5, 7, 8, 10, 11, 13, 14, 16, 17]].reshape(-1, 5, 2)

    hw_landmark_box = landmark_list.max(1) - landmark_list.min(1)

    print(f'Face Box Width/Height Reatio :: {(hw_box_list[:, 0]/hw_box_list[:, 1]).mean()}')
    print(f'Face Box/Landmarks Width and Height Reatio :: {(hw_box_list/hw_landmark_box).mean(0)}')
