"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from exercise_code.data.segmentation_dataset import SegmentationData, label_img_to_rgb
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np

class SegmentationNN(pl.LightningModule):

    def __init__(self, n_class=23, hparams=None):
        super().__init__()
        self.hparams = hparams
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        # Input size: [3,240,240]
        # Output size: [23,240,240]
        
        self.model = nn.Sequential(
            nn.Conv2d(3,3,kernel_size=3,stride=1,padding=1,groups=3),
            nn.BatchNorm2d(3),
            nn.PReLU(),         
            nn.Conv2d(3,92,kernel_size=1,stride=1),
            nn.BatchNorm2d(92),
            nn.PReLU(),
            
            nn.Conv2d(92,92,kernel_size=3,stride=1,padding=1,groups=92),
            nn.BatchNorm2d(92),
            nn.PReLU(),                            
            nn.Conv2d(92,184,kernel_size=1,stride=1),
            nn.BatchNorm2d(184),
            nn.PReLU(),
            
            nn.MaxPool2d(3,stride=2),
            
            nn.Conv2d(184,184,kernel_size=3,stride=1,padding=1,groups=184),
            nn.BatchNorm2d(184),
            nn.PReLU(),                           
            nn.Conv2d(184,368,kernel_size=1,stride=1),
            nn.BatchNorm2d(368),
            nn.PReLU(),
                                            
            nn.Conv2d(368,368,kernel_size=3,stride=1,padding=1,groups=368),
            nn.BatchNorm2d(368),
            nn.PReLU(),                            
            nn.Conv2d(368,736,kernel_size=1,stride=1),
            nn.BatchNorm2d(736),
            nn.PReLU(),
            
            nn.MaxPool2d(3,stride=2),
            
            nn.ConvTranspose2d(736,736,kernel_size=4,stride=2,padding=1,groups=736),
            nn.BatchNorm2d(736),
            nn.PReLU(),         
            nn.ConvTranspose2d(736,368,kernel_size=1,stride=1),
            nn.BatchNorm2d(368),
            nn.PReLU(),
            
            nn.ConvTranspose2d(368,368,kernel_size=4,stride=2,padding=1,groups=368),
            nn.BatchNorm2d(368),
            nn.PReLU(),         
            nn.ConvTranspose2d(368,184,kernel_size=1,stride=1),
            nn.BatchNorm2d(184),
            nn.PReLU(),
            
            nn.ConvTranspose2d(184,184,kernel_size=4,stride=2,padding=1,groups=184),
            nn.BatchNorm2d(184),
            nn.PReLU(),         
            nn.ConvTranspose2d(184,92,kernel_size=1,stride=1),
            nn.BatchNorm2d(92),
            nn.PReLU(),
            
            nn.ConvTranspose2d(92,92,kernel_size=4,stride=2,padding=1,groups=92),
            nn.BatchNorm2d(92),
            nn.PReLU(),         
            nn.ConvTranspose2d(92,23, kernel_size=1,stride=1),
            nn.BatchNorm2d(23),
            nn.PReLU(),           
        )



    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        x = self.model(x)
        return x
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
    
    def general_step(self, batch, batch_idx, mode):

        inputs, targets = batch     
        outputs = self.forward(inputs)
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        loss = loss_func(outputs, targets)
        
        preds = outputs.argmax(axis=1)
        n_correct = (targets == preds).sum()
        return loss, n_correct
    
    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        total_correct = torch.stack([x[mode + '_n_correct'] for x in outputs]).sum().cpu().numpy()
        acc = total_correct / len(self.dataset[mode])
        return avg_loss, acc

    def training_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'loss': loss}
        return {'loss': loss, 'train_n_correct':n_correct, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "val")
        return {'val_loss': loss, 'val_n_correct':n_correct}
    
    def test_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_n_correct':n_correct}

    def validation_end(self, outputs):
        avg_loss, acc = self.general_end(outputs, "val")
        #print("Val-Acc={}".format(acc))
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'val_acc': acc, 'log': tensorboard_logs}

    def prepare_data(self):

        # create dataset
        data_root = "../datasets/segmentation"
        my_transform = None

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        train_data = SegmentationData(image_paths_file=f'{data_root}/segmentation_data/train.txt')
        val_data = SegmentationData(image_paths_file=f'{data_root}/segmentation_data/val.txt')
        test_data = SegmentationData(image_paths_file=f'{data_root}/segmentation_data/test.txt')

        # assign to use in dataloaders
        self.dataset = {}
        self.dataset["train"], self.dataset["val"], self.dataset["test"] = train_data, val_data, test_data

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.dataset["train"], shuffle=True, batch_size=self.hparams["batch_size"])

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.dataset["val"], batch_size=self.hparams["batch_size"])
    
    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.hparams["batch_size"])

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"], betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
        return optim

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
