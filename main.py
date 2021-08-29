import numpy as np
import torch

import config
from data_loader import get_train_loader, get_transfer_data
from trainer import Trainer


def run(config):
    if config.use_gpu:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        np.random.seed(1)

        # instantiate data loaders
        train_loader = get_train_loader(
            config.pos_train_data, config.batch_size, is_shuffle=True)

        # instantiate trainer

        test_set, test_targets = get_transfer_data(phase='test')
        trainer = Trainer(config, train_loader, test_set)

        trainer.train(train_loader)
        # trainer.test(test_set, temp_targets)


if __name__ == '__main__':
    config, unparsed = config.get_config()
    run(config)
