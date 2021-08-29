import os

import cv2


def calc_mean(image_file):
    B_sum = 0
    G_sum = 0
    R_sum = 0
    p_sum = 0

    image_list = os.listdir(image_file)
    for i in range(len(image_list)):
        single_img = cv2.imread(image_file + '/' + image_list[i])
        h, w, c = single_img.shape

        for hi in range(h):
            for wi in range(w):
                B_sum += single_img[hi][wi][0]
                G_sum += single_img[hi][wi][1]
                R_sum += single_img[hi][wi][2]
                p_sum += 1

        B_mean = B_sum / p_sum
        G_mean = G_sum / p_sum
        R_mean = R_sum / p_sum

        mean = (B_mean, G_mean, R_mean)

    return mean


if __name__ == "__main__":
    test_img_file = '/home/yzh/ssd_head_detector_v/dataset/'
    print(calc_mean(test_img_file))
