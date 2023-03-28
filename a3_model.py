import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
# Whatever other imports you need
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix

# You can implement classes and helper functions here too.

class Email_Dataset(Dataset):
    # the parameter train decides if this dataset contains the training or test data
    # default is that it will be the training data
    def __init__(self, train=True):
        xy = np.loadtxt(args.featurefile, skiprows=1)
        xs = xy[np.where(xy[:, 0] == (1 if train else 0))]
        
        # read in data from the file
        self.authors = torch.from_numpy(xs[:, [1]])
        self.x_reduced = torch.from_numpy(xs[:, 2:])

        # some book-keeping
        self.n_samples = len(self.x_reduced)
        self.n_dims = len(self.x_reduced[0])
        self.authortable = self.construct_author_table()
        self.n_classes = len(self.authortable)
        self.author_names = self.get_author_names()

    def __getitem__(self, index):
        return self.x_reduced[index], self.authors[index]
    
    def __len__(self):
        return self.n_samples

    def construct_author_table(self):
        with open(args.featurefile) as file:
            table = file.readline()
            items = table.split()
            authortable = {}
            for idx in range(0,len(items),2):
                authortable[items[idx]] = int(items[idx+1])
            return authortable
    
    # given a guess from argmax(logsoftmax), return the author name as a string
    def get_author_name(self, y_pred):
        for author,y in self.authortable.items():
            if y_pred == y:
                return author
            
    def get_author_names(self):
        authors = []
        for author,y in self.authortable.items():
            authors.append(author)
        return authors

class Model(nn.Module):
    def __init__(self, n_dims,n_classes, hidden_size=None, act1=None, act2=None) -> None:
        """ is called when model is created: e.g model = Model()
            - definition of layers
        """
        super().__init__()

        # n_dims input layer size, outputs n_classes length vector as output
        # default, without using hidden layer
        if hidden_size == None:
            self.input_layer = nn.Linear(n_dims,n_classes, dtype=float)
            self.hidden = None
        # with hidden layer and two activation functions
        else:
            self.input_layer = nn.Linear(n_dims,hidden_size, dtype=float)
            if act1 == "sigmoid":
                self.act1 = nn.Sigmoid()
            elif act1 == "relu":
                self.act1 = nn.ReLU()
            self.hidden = nn.Linear(hidden_size, n_classes, dtype=float)
            if act2 == "sigmoid":
                self.act2 = nn.Sigmoid()
            elif act2 == "relu":
                self.act2 = nn.ReLU()

        # the n_classes length vector is fed into LogSoftmax to get the probabilities
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, data):
        """ is called when the model object is called: e.g. model(sample_input)
            - defines, how the input is processed with the previuosly defined layers 
        """
        if self.hidden == None:
            after_input_layer = self.input_layer(data) # run data through Perceptron layer
            probs = self.softmax(after_input_layer) # run output of that through LogSoftmax
            return probs
        else:
            after_input_layer = self.input_layer(data)
            after_act1 = self.act1(after_input_layer)
            after_hidden_layer = self.hidden(after_act1)
            after_act2 = self.act2(after_hidden_layer)
            probs = self.softmax(after_act2)
            return probs

def test_model(model, dataset):
    # true author names for test set
    y_true_no_tensor = list(dataset.get_author_name(x.item()) for x in dataset.authors)
    # model prediction with LogSoftmax probabilities for each author
    y_pred = model(dataset.x_reduced)
    # get the most likely author index
    y_pred_author = torch.argmax(y_pred, dim=1)
    # look up the corresponding author name
    y_pred_no_tensor = list(dataset.get_author_name(x.item()) for x in y_pred_author)
    # calculate and print accuracy
    count = 0
    for pred,truth in zip(y_pred_no_tensor, y_true_no_tensor):
        if pred==truth:
            count = count + 1
    print("accuracy :", count/dataset.n_samples)
    # create and print confusion matrix 
    matrix = pd.DataFrame(confusion_matrix(y_true_no_tensor, y_pred_no_tensor))
    print(matrix)
    print("confusion matrix has author numbers not names, here is how to get which name the number corresponds to: ")
    print(dataset.authortable)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    parser.add_argument('--batchsize', dest='batchsize', type=int, default="50", help='Batch size to use during training')
    parser.add_argument('--epochs', dest='epochs', type=int, default="10", help="number of epochs to train for")
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, default="0.002", help="learning rate to use for computing gradients")
    # Add options here for part 3 -- hidden layer and nonlinearity,
    # and any other options you may think you want/need.  Document
    # everything.
    parser.add_argument('--use-hidden-with-size', dest='hidden_size', type=int, help='use a hidden linear layer with a specific size')
    parser.add_argument('--act-after-input-layer', dest='act1', type=str, help='activation function to use after input layer (sigmoid or relu)')
    parser.add_argument('--act-after-hidden-layer', dest='act2', type=str, help='activation function to use after hidden layer (sigmoid or relu)')

    args = parser.parse_args()
    print(f'training a model with batchsize={args.batchsize}, epochs={args.epochs}, and learning-rate={args.learning_rate}')
    if not args.hidden_size == None:
        print(f'using hidden layer with size {args.hidden_size}')
        print(f'using activation functions {args.act1} and {args.act2}')
    print("Reading {}...".format(args.featurefile))

    # implement everything you need here
    train_dataset = Email_Dataset(train=True)
    test_dataset = Email_Dataset(train=False)

    if args.hidden_size == None:
        model = Model(train_dataset.n_dims, train_dataset.n_classes)
    else:
        model = Model(train_dataset.n_dims
                     , train_dataset.n_classes
                     , hidden_size=args.hidden_size
                     , act1=args.act1
                     , act2=args.act2)

    learning_rate = args.learning_rate
    batch_size = args.batchsize
    num_epochs = args.epochs
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    dataloader = DataLoader(train_dataset,
                        batch_size=batch_size,
                        shuffle=True)

    # number of epochs = how often the model sees the complete dataset
    for epoch in range(num_epochs):
        total_loss = 0
        # loop through batches of the dataloader
        for i, batch in enumerate(dataloader):
            model_input = batch[0]
            ground_truth = batch[1]
            # send the batch of texts to the forward function of the model
            output = model(model_input)
            # compare the output of the model to the ground truth to calculate the loss
            # the lower the loss, the closer the model's output is to the ground truth
            ground_truth = torch.Tensor([x[0] for x in ground_truth])
            loss = loss_function(output, ground_truth.long())
            # update total_loss
            total_loss += loss.item()
            # print average loss
            print(f'epoch {epoch},', f'batch {i}:', round(total_loss / (i + 1), 4), end='\r')
            
            # train the model based on the loss:
            
            # compute gradients
            loss.backward()
            # update parameters
            optimizer.step()
            # reset gradients
            optimizer.zero_grad()
        print()

# test the model
test_model(model, test_dataset)