import numpy as np
import torch
import re
import math
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as topti
from torchtext import data
from torchtext.vocab import GloVe
from imdb_dataloader import IMDB
from matplotlib import pyplot as plt
from torch.autograd import Variable

stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves",
 "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
  "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
   "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
    "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has",
     "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but",
      "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
       "about", "against", "between", "into", "through", "during", "before", "after",
        "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
         "under", "again", "further", "then", "once", "here", "there", "when", "where",
          "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
           "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

stop_words1 = ["in", "of", "from"]

class Self_Attention(tnn.Module):
    def __init__(self, query_dim):
        # assume: query_dim = key/value_dim
        super(Self_Attention, self).__init__()
        self.scale = 1. / math.sqrt(query_dim)

    def forward(self, query, key, value):
        # query == hidden: (batch_size, hidden_dim * 2)
        # key/value == gru_output: (sentence_length, batch_size, hidden_dim * 2)
        query = query.unsqueeze(1) # (batch_size, 1, hidden_dim * 2)
        # print("query", query.shape)
        key = key.transpose(1, 2) # (batch_size, hidden_dim * 2, sentence_length)
        # print("key.shape", key.shape)

        # bmm: batch matrix-matrix multiplication
        attention_weight = torch.bmm(query, key) # (batch_size, 1, sentence_length)
        # print("attention_weight", attention_weight.shape)

        attention_weight = F.softmax(attention_weight.mul_(self.scale), dim=2) # normalize sentence_length's dimension
        # print("attention_weight1", attention_weight.shape)
        #value = value.transpose(0, 1) # (batch_size, sentence_length, hidden_dim * 2)
        # print("value", value.shape)
        attention_output = torch.bmm(attention_weight, value) # (batch_size, 1, hidden_dim * 2)
        # print("attention_output", attention_output)
        attention_output = attention_output.squeeze(1) # (batch_size, hidden_dim * 2)
        # print("attention_output1", attention_output)
        return attention_output, attention_weight.squeeze(1)

# TO DO: Build a model using transformer layers (eg. ELMO, BERT) and aim for a 90% accuracy 
#Look into how to implement the transformer layers.

# Class for creating the neural network.
class Network(tnn.Module):
    def __init__(self):
        super(Network, self).__init__()
        """
        TODO:
        Create and initialise weights and biases for the layers.
        """
        self.gru = tnn.GRU(50, 200, num_layers=3, bidirectional=True, dropout=0.4, batch_first=True)
        self.dense = tnn.Linear(2 * 200, 1)
        self.dropout = tnn.Dropout(0.4)
        self.attention = Self_Attention(2 * 200)

        # self.fc1 = tnn.Linear(64,20, bias=True)
        # self.lstm = tnn.LSTM(50,32, bias=True, batch_first=True, bidirectional=True)
        # self.fc2 = tnn.Linear(20,1, bias=True)
        # self.relu = tnn.ReLU()
        # self.hidden = torch.zeros(1, 64, 100)
        # self.dropout1 = tnn.Dropout(p=0.05)
        # self.dropout2 = tnn.Dropout(p=0.3)
        # self.pool = tnn.MaxPool1d(2)
        # self.conv1 = tnn.Conv1d(50,50,kernel_size=3,padding=5, bias=True)
        # self.fc1 = tnn.Linear(200,64, bias=True)
        # self.lstm = tnn.LSTM(50,100, bias=True, batch_first=True, bidirectional=True, dropout=0.2)
        # self.gru =tnn.GRU(50,100, bias=True, batch_first=True, bidirectional=True, dropout=0.2)
        # self.fc2 = tnn.Linear(64,1, bias=True)
        # self.relu = tnn.ReLU()
        # self.hidden = torch.zeros(1, 64, 100)
        # self.dropout1 = tnn.Dropout(p=0.2)
        # self.dropout2 = tnn.Dropout(p=0.3)



    def forward(self, input, length):
        """
        DO NOT MODIFY FUNCTION SIGNATURE
        Create the forward pass through the network.
        """
      # output = F.conv1d(input,Variable(torch.zeros(50,input.shape[1],8)),Variable(torch.zeros(50)),padding=5)
        h0 = Variable(torch.zeros(6, input.shape[0], 200)).cuda()
        output = self.dropout(input)
        output, hn = self.gru(output,h0)
        # print(hn.shape)
        # print(output.shape)
        hidden1 = hn[-2,:,:]
        # print("hidden", hidden1.shape)
        hidden2 = hn[-1,:,:]
        # print("hidden2", hidden2.shape)
        cat = torch.cat((hidden1, hidden2), dim=1)
        # print("cat", cat.shape)
        hidden3 = self.dropout(cat)
        # print("hidden3", hidden3.shape)
        # hidden: (batch_size, hidden_dim * 2)
        rescaled_hidden, attention_weight = self.attention(query=hidden3, key=output, value=output)

        output = self.dense(rescaled_hidden)
        output=output.squeeze()
        # c0 = Variable(torch.zeros(2, input.shape[0], 100)).cuda()
        # output = self.dropout1(input)
        # output = output.permute(0,2,1)
        # output = self.conv1(output)
        # output = F.relu(output)
        # output = self.pool(output)

        # output = output.permute(0,2,1)
        # print('input shape', input.shape)
        # print(input.shape)
        # output, (hn, cn) = self.lstm(input, (h0, c0))
        # print("h_t",h_t.shape)
        # print("length", length)
        # print("length size", length.shape)
        # print('output shape', output.shape)
        # output=output.reshape(output.shape[0],-1)
        # print(output.shape)
        
        # output = self.pool(output)
        # output = output[:, -1, :]
        # print(output.shape)
        # print(output.shape)
        # output = self.relu(output)
        # print("output.shape", output.shape)
        # output = output[:, -1, :]
        # print("last",output.shape)
        # output = self.fc1(output)
        # output = self.relu(output)
        # print(output.shape)
        # output = self.dropout1(output)
        # print("after first drouput",output.shape)
        # output = self.fc2(output)
        # output = self.dropout2(output)
        # print("after second dropout", output.shape)
        # print(output.shape)
        # output = self.relu(output)
        # print("output.shape", output.shape)
        # output = output[:, -1, :]
        # print("last",output.shape)
        # output = self.fc1(output)
        # output = self.relu(output)
        # print(output.shape)
        # output = self.fc2(output)
        # output=output.squeeze()
        # print('output shape', output.shape)
        return output


class PreProcessing():
    def pre(x):
        noDigits = []
        """Called after tokenization"""
        filtered_sentence = []

  
        #TO DO: Remove html letters, multiple spaces, punctuation and numbers
        
        for w in x: 
          if w not in stop_words1: 
            w = re.sub('[^A-Za-z0-9]+', ' ', w)
            # print(w)
            filtered_sentence.append(w)

        return filtered_sentence
        
    def post(batch, vocab):
        """Called after numericalization but prior to vectorization"""
        return batch

    text_field = data.Field(lower=True, include_lengths=True, batch_first=True, preprocessing=pre, postprocessing=post)


def lossFunc():
    """
    Define a loss function appropriate for the above networks that will
    add a sigmoid to the output and calculate the binary cross-entropy.
    """
    return tnn.BCEWithLogitsLoss()

def main():
    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Load the training dataset, and create a data loader to generate a batch.
    textField = PreProcessing.text_field
    # textField = data.Field(lower=True, include_lengths=True, batch_first=True)
    labelField = data.Field(sequential=False)

    train, dev = IMDB.splits(textField, labelField, train="train", validation="dev")
    print(train.size)

    textField.build_vocab(train, dev, vectors=GloVe(name="6B", dim=50))
    labelField.build_vocab(train, dev)

    trainLoader, testLoader = data.BucketIterator.splits((train, dev), shuffle=True, batch_size=64,
                                                         sort_key=lambda x: len(x.text), sort_within_batch=True)

    net = Network().to(device)
    criterion =lossFunc()
    optimiser = topti.Adam(net.parameters(), lr=0.001)  # Minimise the loss using the Adam algorithm.

    loss_values = []
    for epoch in range(20):
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

        loss_values.append(running_loss)
        
    num_correct = 0

    
    # Save mode
    torch.save(net.state_dict(), "./model.pth")
    print("Saved model")

    # Evaluate network on the test dataset.  We aren't calculating gradients, so disable autograd to speed up
    # computations and reduce memory usage.
    with torch.no_grad():
        for batch in testLoader:
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # Get predictions
            outputs = torch.sigmoid(net(inputs, length))
            predicted = torch.round(outputs)

            num_correct += torch.sum(labels == predicted).item()

    accuracy = 100 * num_correct / len(dev)

    print(f"Classification accuracy: {accuracy}")
    plt.plot(loss_values)

if __name__ == '__main__':
    main()
