import argparse
import os
import torch
from modules.model import build_ssd
from visible_test_pred import visible_pred

now_path = os.getcwd()
father_path = os.path.abspath(os.path.dirname(now_path) + os.path.sep + '.')


def main(image_file):
    parser = argparse.ArgumentParser(description='SSD head detector')
    parser.add_argument('--model', default=father_path + '/ckpt/',
                        type=str, help='Trained state_dict file path')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='Use cuda in live demo')
    args = parser.parse_args()

    models_list = os.listdir(args.model)

    if models_list != []:
        model_last = args.model + models_list[len(models_list) - 1]
        add = os.listdir(model_last)[0]
    print('Using: ' + model_last + '/' + add + ' now!')
    trained_model = torch.load(model_last + "/" + add)

    net = build_ssd()
    net.load_state_dict(trained_model['model_state'], strict=True)
    net.cuda()
    net.eval()

    visible_pred(image_file, net)


if __name__ == '__main__':
    image_file = "empty"
    main(image_file)


