#!/usr/bin/env python3
"""
part2.py

UNSW COMP9444 Neural Networks and Deep Learning

ONLY COMPLETE METHODS AND CLASSES MARKED "TODO".

DO NOT MODIFY IMPORTS. DO NOT ADD EXTRA FUNCTIONS.
DO NOT MODIFY EXISTING FUNCTION SIGNATURES.
DO NOT IMPORT ADDITIONAL LIBRARIES.
DOING SO MAY CAUSE YOUR CODE TO FAIL AUTOMATED TESTING.

YOU MAY MODIFY THE LINE net = NetworkLstm().to(device)
"""

import numpy as np

import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as topti
from torch.autograd import Variable

from torchtext import data
from torchtext.vocab import GloVe


# Class for creating the neural network.
class NetworkLstm(tnn.Module):
    """
    Implement an LSTM-based network that accepts batched 50-d
    vectorized inputs, with the following structure:
    LSTM(hidden dim = 100) -> Linear(64) -> ReLu-> Linear(1)
    Assume batch-first ordering.
    Output should be 1d tensor of shape [batch_size].
    """

    def __init__(self):
        super(NetworkLstm, self).__init__()
        """
        TODO:
        Create and initialise weights and biases for the layers.
        """
        self.fc1 = tnn.Linear(100,64, bias=True)
        self.lstm = tnn.LSTM(50,100, bias=True, batch_first=True)
        self.fc2 = tnn.Linear(64,1, bias=True)
        self.relu = tnn.ReLU()
        self.hidden = torch.zeros(1, 64, 100)
        # init_weights(self.lstm)
        # init_weights(self.fc1)
        # init_weights(self.fc2)

    def forward(self, input, length):

        """
        DO NOT MODIFY FUNCTION SIGNATURE
        TODO:
        Create the forward pass through the network.
        """
        h0 = torch.zeros(1, input.shape[0], 100)

        # Initialize cell state
        c0 = torch.zeros(1, input.shape[0], 100)
        # print('input shape', input.shape)
        output, (hn, cn) = self.lstm(input, (h0, c0))
        # print("h_t",h_t.shape)
        # print("length", length)
        # print("length size", length.shape)
        # print('output shape', output.shape)
        # output=output.reshape(output.shape[0],-1)
        # print(output.shape)
        output = output[:, -1, :]
        # print(output.shape)
        # print(output.shape)
        # output = self.relu(output)
        # print("output.shape", output.shape)
        # output = output[:, -1, :]
        # print("last",output.shape)
        output = self.fc1(output)
        output = self.relu(output)
        # print(output.shape)
        output = self.fc2(output)
        output=output.squeeze()
        # print('output shape', output.shape)
        return output




# Class for creating the neural network.
class NetworkCnn(tnn.Module):
    """
    Implement a Convolutional Neural Network.
    All conv layers should be of the form:
    conv1d(channels=50, kernel size=8, padding=5)

    Conv -> ReLu -> maxpool(size=4) -> Conv -> ReLu -> maxpool(size=4) ->
    Conv -> ReLu -> maxpool over time (global pooling) -> Linear(1)

    The max pool over time operation refers to taking the
    maximum val from the entire output channel. See Kim et. al. 2014:
    https://www.aclweb.org/anthology/D14-1181/
    Assume batch-first ordering.
    Output should be 1d tensor of shape [batch_size].
    """

    def __init__(self):
        super(NetworkCnn, self).__init__()
        """
        TODO:
        Create and initialise weights and biases for the layers.
        """
        self.conv1 = tnn.Conv1d(50,50,kernel_size=8,padding=5, bias=True)
        self.pool = tnn.MaxPool1d(4)
        self.pool1 = tnn.MaxPool1d(7)
        self.pool2 = tnn.AdaptiveMaxPool1d(1)
        self.fc1 = tnn.Linear(50,1, bias=True)
        self.relu = tnn.ReLU()

    def forward(self, input, length):
        """
        TODO:
        Create the forward pass through the network.
        """
        #permute the dimensions
        # output = F.conv1d(input,Variable(torch.zeros(50,input.shape[1],8)),Variable(torch.zeros(50)),padding=5)
        input = input.permute(0,2,1)# batch/dimensions/seq length
        output = self.conv1(input)
        # print(output.shape)
        output = F.relu(output)
        output = self.pool(output)
        # print("after first pool", output.shape)
        output = F.relu(self.conv1(output))
        # print("after second conv1", output.shape)
        output = self.pool(output)
        # print('after second pool', output.shape)
        # print("here now1")
        output = F.relu(self.conv1(output))
        # print("after conv1", output.shape)
        # print("after relu before global pool", output.shape)
        output = self.pool2(output) 
        # output = F.max_pool1d(output,output.shape[2]) 
        # print("before view",output.shape)
        output=output.view(output.shape[0], -1)
        # print("after view",output.shape)
        output = self.fc1(output)
        # print("after full connection", output.shape)
        output = torch.squeeze(output)
        # print(output.shape)
        # print(output)
        return output


def lossFunc():
    """
    TODO:
    Return a loss function appropriate for the above networks that
    will add a sigmoid to the output and calculate the binary
    cross-entropy.
    """
    # return tnn.BCELoss()
    return tnn.BCEWithLogitsLoss()



def measures(outputs, labels):
    """
    TODO:
    tp_batch, tn_batch, fp_batch, fn_batch = measures(outputs, labels)

    Return (in the following order): the number of true positive
    classifications, true negatives, false positives and false
    negatives from the given batch outputs and provided labels.

    outputs and labels are torch tensors.True positives are positive reviews 
    correctly identified as positive. True negatives are negative reviews correctly
    identified as negative. False positives are negative reviews incorrectly
    identified as positive. False negatives are postitive reviews incorrectly
    identified as negative.
    """
    outputs = F.sigmoid(outputs)

    tp=0
    tn=0
    fp=0
    fn=0
    print('output', outputs)
    print('labels',labels)

    for i in range(len(outputs)):
        # Change this if need be 
        if (outputs[i] >=0.5):
            outputs[i] = 1
        else:
            outputs[i] = 0


        if (outputs[i] == labels[i]):
            if(labels[i] == 1):
                tp+=1
            else:
                tn+=1
        else:
            if(labels[i] == 1):
                fn+=1
            else:
                fp+=1

    return tp,tn,fp,fn



    


def main():
    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Load the training dataset, and create a data loader to generate a batch.
    textField = data.Field(lower=True, include_lengths=True, batch_first=True)
    labelField = data.Field(sequential=False)

    from imdb_dataloader import IMDB
    train, dev = IMDB.splits(textField, labelField, train="train", validation="dev")

    textField.build_vocab(train, dev, vectors=GloVe(name="6B", dim=50))
    labelField.build_vocab(train, dev)

    trainLoader, testLoader = data.BucketIterator.splits((train, dev), shuffle=True, batch_size=64,
                                                         sort_key=lambda x: len(x.text), sort_within_batch=True)

    # Create an instance of the network in memory (potentially GPU memory). Can change to NetworkCnn during development.
    net =  NetworkCnn().to(device)

    criterion = lossFunc()
    optimiser = topti.Adam(net.parameters(), lr=0.001)  # Minimise the loss using the Adam algorithm.

    for epoch in range(10):
        print(epoch)
        running_loss = 0

        for i, batch in enumerate(trainLoader):
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # PyTorch calculates gradients by accumulating contributions to them (useful for
            # RNNs).  Hence we must manually set them to zero before calculating them.
            optimiser.zero_grad()

            # Forward pass through the network.
            output = net(inputs, length)

            loss = criterion(output, labels)

            # Calculate gradients.
            loss.backward()

            # Minimise the loss according to the gradient.
            optimiser.step()

            running_loss += loss.item()

            if i % 32 == 31:
                print("Epoch: %2d, Batch: %4d, Loss: %.3f" % (epoch + 1, i + 1, running_loss / 32))
                running_loss = 0

    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0

    # Evaluate network on the test dataset.  We aren't calculating gradients, so disable autograd to speed up
    # computations and reduce memory usage.
    with torch.no_grad():
        for batch in testLoader:
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            outputs = net(inputs, length)

            tp_batch, tn_batch, fp_batch, fn_batch = measures(outputs, labels)
            true_pos += tp_batch
            true_neg += tn_batch
            false_pos += fp_batch
            false_neg += fn_batch

    accuracy = 100 * (true_pos + true_neg) / len(dev)
    matthews = MCC(true_pos, true_neg, false_pos, false_neg)

    print("Classification accuracy: %.2f%%\n"
          "Matthews Correlation Coefficient: %.2f" % (accuracy, matthews))


# Matthews Correlation Coefficient calculation.
def MCC(tp, tn, fp, fn):
    numerator = tp * tn - fp * fn
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(numerator, denominator)


if __name__ == '__main__':
    main()
