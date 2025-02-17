import torch
import numpy as np


class ModelWrapper(object):

    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_on_batch(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        if outputs.shape[1] == 1:
            targets = 2.0 * (targets > 4).float() - 1.0
            loss = torch.log(1 + torch.exp(-targets * outputs[:, 0])).mean()
            loss.backward()
            self.optimizer.step()
            correct = (torch.sign(outputs[:, 0]) == targets).sum().item()
            acc = correct / targets.size(0)
            return loss.item(), acc, correct

        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        acc = correct / targets.size(0)
        return loss.item(), acc, correct

    def eval_all(self, test_loader):
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                loss, correct = self.eval_on_batch(inputs, targets)
                total += targets.size(0)
                test_loss += loss
                test_correct += correct
            test_loss /= (batch_idx+1)
            test_acc = test_correct / total
        return test_loss, test_acc

    def eval_on_batch(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        outputs = self.model(inputs)
        if outputs.shape[1] == 1:
            targets = 2.0 * (targets > 4).float() - 1.0
            loss = torch.log(1 + torch.exp(-targets * outputs[:, 0])).mean()
            correct = (torch.sign(outputs[:, 0]) == targets).sum().item()
            return loss.item(), correct

        loss = self.criterion(outputs, targets)
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        return loss.item(), correct

    def train(self):
        return self.model.train()

    def eval(self):
        return self.model.eval()



