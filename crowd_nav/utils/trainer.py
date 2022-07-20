import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable

from crowd_nav.utils.memory2 import ReplayBuffer


class Trainer(object):
    def __init__(
        self,
        model,
        memory: ReplayBuffer,
        device,
        batch_size,
        writer: SummaryWriter = None,
    ):
        """
        Train the trainable model of a policy
        """
        self.model = model
        self.device = device
        self.criterion = nn.MSELoss().to(device)
        self.memory = memory
        self.data_loader = None
        self.batch_size = batch_size
        self.optimizer = None
        self.writer = writer
        self.global_count = 0

    def set_learning_rate(self, learning_rate):
        logging.info("Current learning rate: %f", learning_rate)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def il_optimize_epoch(self, num_epochs):
        if self.optimizer is None:
            raise ValueError("Learning rate is not set!")
        if self.data_loader is None:
            # self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
            self.data_loader = self.memory
        average_epoch_loss = 0
        for epoch in range(num_epochs):
            self.global_count += 1
            epoch_loss = 0
            inputs, values = self.memory.sample_batches(batch_size=self.batch_size, sample_all=True)
            for net_input, value in zip(inputs, values):
                net_input = Variable(torch.from_numpy(np.array(net_input, dtype=np.float32))).to(self.device)
                value = Variable(torch.from_numpy(np.array(value, dtype=np.float32))).to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(net_input)
                loss = self.criterion(outputs, value)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.data.item()

            average_epoch_loss = epoch_loss / len(self.memory)
            logging.debug("Average loss in epoch %d: %.2E", epoch, average_epoch_loss)

            self.writer.add_scalar(
                "train/average_epoch_loss",
                epoch_loss / len(self.memory),
                self.global_count,
            )

        return average_epoch_loss

    def rl_optimize_batch(self, num_batches):
        self.global_count += 1
        if self.optimizer is None:
            raise ValueError("Learning rate is not set!")
        # if self.data_loader is None:
        #     self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        losses = 0
        inputs, values = self.memory.sample_batches(
            batch_size=self.batch_size, sample_all=False, num_batches=num_batches
        )
        for net_input, value in zip(inputs, values):
            net_input = Variable(torch.from_numpy(np.array(net_input, dtype=np.float32))).to(self.device)
            value = Variable(torch.from_numpy(np.array(value, dtype=np.float32))).to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(net_input)
            loss = self.criterion(outputs, value)
            loss.backward()
            self.optimizer.step()
            losses += loss.data.item()

        average_loss = losses / num_batches
        logging.debug("Average loss : %.2E", average_loss)
        self.writer.add_scalar("train/average_epoch_loss", losses / num_batches, self.global_count)

        return average_loss
