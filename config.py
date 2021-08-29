import argparse
import os

now_path = os.getcwd()
father_path = os.path.abspath(os.path.dirname(now_path) + os.path.sep + '.')

arg_lists = []
parser = argparse.ArgumentParser(description='RAM')


def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# data params
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--pos_train_data', type=str, default=now_path + '/dataset/pos/train.txt',
                      help='Directory of the pos_train_data')
data_arg.add_argument('--pos_test_data', type=str,
                      default=now_path + '/dataset/pos/test.txt',
                      help='Directory of the pos_test_data')
data_arg.add_argument('--batch_size', type=int, default=32,
                      help='# of images in each batch of data')
data_arg.add_argument('--num_workers', type=int, default=0,
                      help='# of subprocesses to use for data loading')

# training params
train_arg = add_argument_group('Training Params')
train_arg.add_argument('--epochs', type=int, default=18300,
                       help='# of epochs to train for')
train_arg.add_argument('--init_lr', type=float, default=1e-4,  # 1.5e-4
                       help='Initial learning rate value')
train_arg.add_argument('--weight_decay', default=5e-4, type=float,
                       help='Weight decay for SGD')
train_arg.add_argument('--gamma', default=0.1, type=float,
                       help='Gamma update for SGD')
train_arg.add_argument('--visual_threshold', default=0.6, type=float,
                       help='Final confidence threshold')
train_arg.add_argument('--momentum', default=0.9, type=float,
                       help='Momentum value for optim')
train_arg.add_argument('--resume', default=True, type=bool,
                       help='Checkpoint state_dict file to resume training from')

# misc params
misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--use_gpu', type=str2bool, default=True,
                      help="Whether to run on the GPU")
misc_arg.add_argument('--train_print_freq', type=int, default=100,
                      help='How frequently to print training details')
misc_arg.add_argument('--test_print_freq', type=int, default=100000,
                      help='How frequently to print testing details')
misc_arg.add_argument('--train_tensorboard_freq', type=int, default=100,
                      help='How frequently to print training details')
misc_arg.add_argument('--ckpt_dir', type=str, default=now_path + '/ckpt/',
                      help='Directory in which to save model checkpoints')
misc_arg.add_argument('--ckpt_freq', type=int, default=100,
                      help='How frequently to save model')
misc_arg.add_argument('--log_dir', type=str, default=now_path + '/logs/',
                      help='Directory of logs')
misc_arg.add_argument('--transfer_learning', type=bool, default=True,
                      help='whether use Transfer learning')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
