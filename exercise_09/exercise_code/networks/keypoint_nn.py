"""Models for facial keypoint detection"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from exercise_code.data.facial_keypoints_dataset import FacialKeypointsDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np


class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
        """
        super(KeypointModel, self).__init__()
        self.hparams = hparams
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        ########################################################################

        # K - out_channels: number of filters in the convolutional layer
        # F - Kernel_size
        # S - the stride of the convolution
        # P - the padding
        # W - the width/height (square) of the previous layer

        self.conv1 = nn.Conv2d(1, 32, 5)
        # output size = (W-F)/S +1 = (96-5)/1 + 1 = 92
        self.pool1 = nn.MaxPool2d(2, 2)
        # 92/2 = 46  the output Tensor for one image, will have the dimensions: (32, 46, 46)

        self.conv2 = nn.Conv2d(32,64,3)
        # output size = (W-F)/S +1 = (46-3)/1 + 1 = 44
        self.pool2 = nn.MaxPool2d(2, 2)
        #44/2=22   the output Tensor for one image, will have the dimensions: (64, 22, 22)

        self.conv3 = nn.Conv2d(64,128,3)
        # output size = (W-F)/S +1 = (22-3)/1 + 1 = 20
        self.pool3 = nn.MaxPool2d(2, 2)
        #20/2=10    the output Tensor for one image, will have the dimensions: (128, 10, 10)


        # Fully-connected linear Layer
        self.fc1 = nn.Linear(128*10*10, hparams['n_hidden'])
        self.fc2 = nn.Linear(hparams['n_hidden'], 30)
        
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################


    def forward(self, x):
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints                                    #
        ########################################################################
        
        # Go through the conv layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        # Flatten the image first
        x = x.view(x.size(0), -1)
        
        # Feed to the fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x

    
    def general_step(self, batch, batch_idx, mode):
        image, keypoints = batch["image"], batch["keypoints"]
        criterion = torch.nn.MSELoss()
        predicted_keypoints = self.forward(image).view(-1,15,2)
        loss = criterion(torch.squeeze(keypoints),torch.squeeze(predicted_keypoints))
        n_correct = (keypoints == predicted_keypoints).sum()
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
        data_root = "../datasets/facial_keypoints"
      
        train_dataset = FacialKeypointsDataset(
                        train=True,
                        transform=transforms.ToTensor(),
                        root=data_root)
        
        val_dataset = FacialKeypointsDataset(
                        train=False,
                        transform=transforms.ToTensor(),
                        root=data_root)
        
        # assign to use in dataloaders
        self.dataset = {}
        self.dataset["train"], self.dataset["val"] = train_dataset, val_dataset

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.dataset["train"], shuffle=True, batch_size=self.hparams["batch_size"])

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.dataset["val"], batch_size=self.hparams["batch_size"])
    
    #@pl.data_loader
    #def test_dataloader(self):
    #    return DataLoader(self.dataset["test"], batch_size=self.hparams["batch_size"])

    def configure_optimizers(self):

        optim = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"], betas=(0.9, 0.999), eps=1e-08, weight_decay=self.hparams["reg"], amsgrad=False)

        return optim

    def getTestAcc(self, loader = None):
        self.model.eval()
        self.model = self.model.to(self.device)

        if not loader: loader = self.test_dataloader()

        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.device)
            score = self.forward(X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc    
    
    
    

class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
