import argparse
import os

import cv2
import numpy as np
import torch
from torch.autograd import Variable

import config
from calc_mean import calc_mean
from modules.model import build_ssd

cfg, temp = config.get_config()
now_path = os.getcwd()

COLOR_PRED = (0, 0, 255)
COLOR_TRUTH = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
labelmap = ['person']


def plot_pred_rect(net, img, transform):
    height, width = img.shape[:2]
    # to rgb
    img_x = img[:, :, (2, 1, 0)]
    img_x = torch.from_numpy(transform(img_x)[0]).permute(2, 0, 1)

    temp = Variable(img_x.unsqueeze(0))

    temp = Variable(temp.cuda())
    y = net(temp, 'test')
    detections = y.data

    scale = torch.Tensor([width, height, width, height])

    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.60:
            print(detections[0, i, j, 0])
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            cv2.rectangle(img,
                          (int(pt[0]), int(pt[1])),
                          (int(pt[2]), int(pt[3])),
                          COLOR_PRED, 2)
            cv2.putText(img, labelmap[i - 1], (int(pt[0]), int(pt[1])),
                        FONT, 1, COLOR_PRED, 2, cv2.LINE_AA)
            j += 1

    return img


def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels


def visible_pred(image_file, net):
    image_list = os.listdir(image_file)
    mean = calc_mean(image_file)

    for i in range(len(image_list)):
        img = cv2.imread(image_file + '/' + image_list[i])
        transform = BaseTransform(net.size, mean)

        image_plot = plot_pred_rect(net, img, transform)

        pic_file = now_path + '/visible_test/'
        cv2.imwrite(pic_file + 'test_' + str(i + 1) + '.png', image_plot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument("--model", default=now_path + '/ckpt/',
                        type=str, help='Trained state_dict file path')
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda in live demo')
    args = parser.parse_args()

    net = build_ssd()
    models_list = os.listdir(args.model)
    if models_list != []:
        model_last = args.model + models_list[len(models_list) - 1]
        add = os.listdir(model_last)[0]
    print(model_last + '/' + add)
    trained_model = torch.load(model_last + '/' + add)

    # load variables from checkpoint
    net.load_state_dict(trained_model['model_state'], strict=True)
    net.cuda()
    net.eval()

    test_img = '/home/yzh/ssd_head_detector/test_image/'
    visible_pred(test_img, net)
