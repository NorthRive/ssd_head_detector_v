import os
import time
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import config
from modules.config import head_set
from modules.layers.multibox_loss import MultiBoxLoss
from modules.model import build_ssd
from tools.logger import my_logger, current_log_dir
from tools.utils import AverageMeter


class Trainer(object):
    def __init__(self, config, train_loader, test_set):
        """
        Construct a new Trainer instance.

        Args
        ----
        - config: object containing command line arguments.
        - data_loader: data iterator
        """
        self.config = config
        # data params
        self.train_loader = train_loader
        self.num_train = len(self.train_loader.dataset)
        self.test_set = test_set
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.start_iter = 0
        self.pos_train_data = config.pos_train_data
        self.pos_test_data = config.pos_test_data

        # training params
        self.epochs = config.epochs  # the total epoch to train
        self.start_epoch = 0
        self.lr = config.init_lr
        self.weight_decay = config.weight_decay
        self.gamma = config.gamma
        self.visual_threshold = config.visual_threshold
        self.momentum = config.momentum
        self.resume = config.resume

        # misc params
        self.use_gpu = config.use_gpu
        self.ckpt_dir = config.ckpt_dir  # output dir
        self.train_print_freq = config.train_print_freq
        self.test_print_freq = config.test_print_freq
        self.train_tensorboard_freq = config.train_tensorboard_freq
        self.ckpt_freq = config.ckpt_freq
        self.train_iter = 0
        self.transfer_learning = config.transfer_learning

        if self.use_gpu and torch.cuda.device_count() > 1:
            print("We have", torch.cuda.device_count(), "GPUs! \n")

        # configure tensorboard logging
        self.writer = SummaryWriter(log_dir=current_log_dir)

        # build model(train & test)
        self.model = build_ssd()
        self.net = self.model

        if self.use_gpu:
            self.net = torch.nn.DataParallel(self.model).cuda()
            cudnn.benchmark = True

        # initialize optimizer and criterion
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum,
                                   weight_decay=self.weight_decay)
        self.criterion = MultiBoxLoss(head_set['num_classes'], 0.5, True, 0, True, 3, 0.5,
                                      False, self.use_gpu)

        my_logger.info('[*] Number of model parameters: {:,}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))

    def train(self, data_loader):
        my_logger.info("\n[*] Train on {} samples".format(self.num_train))  # no problem

        if not self.resume:
            print('Initializing weights...')
            # initialize newly added layers' weight with xavier method
            self.model.extras.apply(weights_init)
            self.model.loc.apply(weights_init)
            self.model.conf.apply(weights_init)
        else:
            self.load_checkpoint(input_file_path=self.ckpt_dir)

        if self.transfer_learning:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.conf.parameters():
                param.requires_grad = True


            self.optimizer = optim.SGD(self.model.conf.parameters(), lr=self.lr, momentum=self.momentum,
                                       weight_decay=self.weight_decay)

        self.net.train(True)

        print('Training SSD on: HEADSET')
        step_index = 0

        # for iteration in range(self.start_iter, head_set['max_iter']):
        for epoch in range(self.start_epoch, self.epochs):
            my_logger.info('\nEpoch: {}/{} - base LR: {:.6f}'.format(
                epoch + 1, self.epochs, self.lr))  # no problem

            for param_group in self.optimizer.param_groups:
                my_logger.info('Learning rate: {}'.format(param_group['lr']))

            # train for 1 epoch
            self.train_one_epoch(epoch, data_loader, step_index)
            if epoch % self.ckpt_freq == 0 and epoch != 0:
                ckpt_dir = self.ckpt_dir + datetime.now().strftime('%Y-%m-%d_%H-%M-%S-')
                epoch_dir = ckpt_dir + str(epoch)
                if not os.path.exists(epoch_dir):
                    os.makedirs(epoch_dir)
                add_file_name = 'epoch' + str(epoch)
                self.save_checkpoint(epoch_dir,
                                     {"epoch": epoch + 1,
                                      "model_state": self.model.state_dict(),
                                      "optim_state": self.optimizer.state_dict(),
                                      "iter": self.train_iter},
                                     add=add_file_name)

        # save final trained model
        if not os.path.exists(self.ckpt_dir + 'final/'):
            os.makedirs(self.ckpt_dir + 'final/')
        self.save_checkpoint(self.ckpt_dir + 'final/',
                             {"epoch": epoch + 1,
                              "model_state": self.model.state_dict(),
                              "optim_state": self.optimizer.state_dict(),
                              "iter": self.train_iter},
                             add='final')
        return

    def train_one_epoch(self, epoch, data_loader, step_index):
        """
        Train the model for 1 epoch of the training set
        """

        batch_time = AverageMeter()
        tic = time.time()

        for batch, (img, targets) in enumerate(data_loader):
            # to CUDA
            img = Variable(img.cuda())
            targets = [Variable(ann, volatile=True) for ann in targets]

            # forward
            t0 = time.time()
            out = self.net(img, 'train')

            # Loss
            self.optimizer.zero_grad()
            loss_l, loss_c = self.criterion(out, targets)
            loss = loss_l + loss_c

            # backprop
            loss.backward()
            self.optimizer.step()
            t1 = time.time()

            # if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(self.train_iter) + ' || Loss: %.4f ||' % (loss.item()), end='\n')

            if self.train_iter in head_set['lr_steps']:
                step_index += 1
                adjust_learning_rate(self.optimizer, self.gamma, step_index)

            if self.train_iter % self.train_tensorboard_freq == 0:
                self.writer.add_scalar('Loss/train', loss.item(), self.train_iter)

            # report information
            if self.train_iter % self.train_print_freq == 0 and self.train_iter != 0:
                my_logger.info('------------------------------------------------------------------------')
                toc = time.time()
                batch_time.update(toc - tic)

                my_logger.info('train loss : {} |  epoch: {} | iteration: {} / {} | duration {} mins'.format(
                    round(loss.item(), 2), epoch + 1, batch, len(data_loader), round(batch_time.val / 60.0, 2)))
                tic = time.time()

            if self.train_iter % self.test_print_freq == 0 and self.train_iter != 0:
                self.test(self.test_set)

            self.train_iter += 1

        return 0.0

    def test(self, test_set, is_load=True):
        """
        test the model for 1 epoch of the testing set
        """
        if is_load:
            models_list = os.listdir(self.ckpt_dir)
            if models_list != []:
                model_last = self.ckpt_dir + models_list[len(models_list) - 1]
                add = os.listdir(model_last)[0]
            trained_model = torch.load(model_last + '/' + add)
            self.model.load_state_dict(trained_model['model_state'], strict=True)
            self.model.cuda()
        labelmap = ['person']
        self.model.eval()

        test_transfrom = BaseTransform(300, (104, 117, 123))

        for k in range(len(test_set)):
            img = cv2.imread(test_set[k])
            if type(img) == type(cv2.imread('unexsit_path')):
                continue
            height, width, channels = img.shape


            img = img[:, :, (2, 1, 0)]
            img = torch.from_numpy(test_transfrom(img)[0]).permute(2, 0, 1)
            img = Variable(img.unsqueeze(0))
            img = Variable(img.cuda())

            # forward
            out = self.model(img, "test")
            detections = out.data

            # scale each detection back to the image
            scale = torch.Tensor([width, height, width, height])
            pred_num = 0

            for i in range(detections.size(1)):
                j = 0
                while detections[0, i, j, 0] >= 0.6:
                    score = detections[0, i, j, 0]
                    label_name = labelmap[i - 1]
                    pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                    coords = (pt[0], pt[1], pt[2], pt[3])
                    pred_num += 1

                    my_logger.info(
                        '\n pred_num:{0:d}, label:{1:s}, score:{2:.4f}, box:{3:s}'.format(pred_num, label_name,
                                                                                          score, (','.join(
                                str(c) for c in coords))))

                    j += 1

    def save_checkpoint(self, dir, state, add=None):
        """
        Save a copy of the model
        """
        if add is not None:
            filename = add + '_ckpt.pth'
        else:
            filename = 'ckpt.pth'
        ckpt_path = os.path.join(dir, filename)
        torch.save(state, ckpt_path)

        print('save file to: ', ckpt_path)

    def load_checkpoint(self, input_file_path='./ckpt/', is_strict=True):
        """
        Load the copy of a model.
        """
        models_list = os.listdir(input_file_path)
        if models_list != []:
            model_last = input_file_path + models_list[len(models_list) - 2]
            add = os.listdir(model_last)[0]

            print('load the pre-trained model: ', model_last + '/' + add)
            ckpt = torch.load(model_last + '/' + add)

            # load variables from checkpoint
            self.model.load_state_dict(ckpt['model_state'], strict=is_strict)
            self.optimizer.load_state_dict(ckpt['optim_state'])
            self.start_epoch = ckpt['epoch']
            self.train_iter = ckpt['iter']

            print(
                "[*] Loaded {} checkpoint ".format(
                    model_last + '/' + add)
            )


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    cfg, temp = config.get_config()
    lr = cfg.init_lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


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
