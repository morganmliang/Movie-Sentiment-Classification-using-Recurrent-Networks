import numpy as np
import torch
import re
import string
import time
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




#################################################################################################
#TRAIN ON ALL DATASET BEFORE SUBMITTING


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
hidden_value = 100
Co = 300
epoch_num = 20
gru_layers = 3
class Network(tnn.Module):
    def __init__(self):
        super(Network, self).__init__()
        """
        TODO:
        Create and initialise weights and biases for the layers.
        """
        
        self.gru = tnn.GRU(50, hidden_value, num_layers=gru_layers, bidirectional=True, dropout=0.5, batch_first=True)
        self.dense = tnn.Linear(2 * hidden_value, 1)
        self.dropout = tnn.Dropout(0.2)
        self.attention = Self_Attention(2 * 200)

        
        self.conv1 = tnn.Conv1d(2 * hidden_value,Co, kernel_size=3, stride=1)
        self.conv2 = tnn.Conv1d(2 * hidden_value,Co, kernel_size=4, stride=1)
        self.conv3 = tnn.Conv1d(2 * hidden_value,Co, kernel_size=5, stride=1)

        self.fc1 = tnn.Linear(3*Co,1)
        # self.fc2 = tnn.Linear(64,1)





        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
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
        
        if(torch.cuda.is_available()):
          h0 = Variable(torch.zeros(6, input.shape[0], hidden_value)).cuda()
          # print("using h0 with gpu")
        else:
          h0 = Variable(torch.zeros(6, input.shape[0], hidden_value))
        #   print("using h0 with cpu")
        output = self.dropout(input)
        output, hn = self.gru(output,h0)
        # print(hn.shape)
        # print("output shape",output.shape)
        # hidden1 = hn[-2,:,:]
        # print("hidden", hidden1.shape)
        # hidden2 = hn[-1,:,:]
        # print("hidden2", hidden2.shape)
        # cat = torch.cat((hidden1, hidden2), dim=1)
        # print("cat", cat.shape)
        # hidden3 = self.dropout(cat)
        output = output.permute(0,2,1)
        output1 = self.conv1(output)
        # print("output1 after conv", output1.shape)
        output1 = F.relu(self.conv1(output))
        # print("output1 before pooling", output1.shape)
        output1 = F.max_pool1d(output1,output1.shape[2])
        # print("output1 after pooling", output1.shape)
        output2 = F.relu(self.conv2(output))
        # print("output2", output2.shape)
        output2 = F.max_pool1d(output2, output2.shape[2])
        # print("output2 after pooling", output2.shape)
        output3 = F.relu(self.conv3(output))
        # print("output3", output3.shape)
        output3 = F.max_pool1d(output3,output3.shape[2])
        # print("output3 after pooling", output3.shape)

        output = torch.cat((output1, output2, output3), 1)
        # print("output concat", output.shape)

        output = torch.squeeze(output)
        output = self.dropout(output)

        output = self.fc1(output)
        # output = self.fc2(output)
        output = torch.squeeze(output)
        # print("final",output.shape)

        # x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        # x = F.max_pool1d(x, x.size(2)).squeeze(2)
        # print("hidden3", hidden3.shape)
        # hidden: (batch_size, hidden_dim * 2)
        # rescaled_hidden, attention_weight = self.attention(query=hidden3, key=output, value=output)

        # output = self.dense(rescaled_hidden)
        # output=output.squeeze()
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
          
          w = re.sub("//><br|><br|br>|<br", "", w)
          w=w.strip(string.punctuation)

          if w not in stop_words1: 
            w = re.sub('[^A-Za-z0-9]+', ' ', w)
            # print(w)
            filtered_sentence.append(w)

        tokens = [word for word in filtered_sentence if len(word) > 2]
        # print(tokens)


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

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    predicted = torch.round(torch.sigmoid(preds))
    correct = (predicted == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

def train():
    # print("what is up")
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        inputs, length, labels = vocab[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

        optimizer.zero_grad()

        labels -= 1
        
        predictions = model(inputs,length)
        
        loss = criterion(predictions, labels)
        
        acc = binary_accuracy(predictions, labels)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)



def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            inputs, length, labels = vocab[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # Get predictions
            outputs = torch.sigmoid(net(inputs, length))
            predictions = torch.round(outputs)

            
            loss = criterion(predictions, labels)
            
            acc = binary_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def main():
    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Load the training dataset, and create a data loader to generate a batch.
    textField = PreProcessing.text_field
    # textField = data.Field(lower=True, include_lengths=True, batch_first=True)
    labelField = data.Field(sequential=False)

    train, dev = IMDB.splits(textField, labelField, train="fulltrain", validation="fulldev")
    print(train.size)

    textField.build_vocab(train, dev, vectors=GloVe(name="6B", dim=50))
    labelField.build_vocab(train, dev)

    trainLoader, testLoader = data.BucketIterator.splits((train, dev), shuffle=True, batch_size=64,
                                                         sort_key=lambda x: len(x.text), sort_within_batch=True)

    net = Network().to(device)
    criterion =lossFunc().to(device)

    optimiser = topti.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)  # Minimise the loss using the Adam algorithm.
    #REPLACE THIS WITH optimiser = topti.Adadelta(net.parameters(), lr=0.001, weight_decay=1e-5)

    loss_values = []
    best_valid_loss = float('inf')
    for epoch in range(epoch_num):
        print(epoch)
        running_loss = 0
        start_time = time.time()

        # vocab = textField.vocab
        epoch_loss = 0
        epoch_acc = 0
        
        net.train()
        
        for i, batch in enumerate(trainLoader):
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            optimiser.zero_grad()

   
            
            predictions = net(inputs,length)
            
            loss = criterion(predictions, labels)
            
            acc = binary_accuracy(predictions, labels)
            
            loss.backward()
            
            optimiser.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
        train_loss = epoch_loss / len(trainLoader)
        train_acc = epoch_acc / len(trainLoader)

        epoch_loss = 0
        epoch_acc = 0
        
        # net.eval()
        
        # with torch.no_grad():
        
        #     for batch in testLoader:

        #         inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
        #             device), batch.label.type(torch.FloatTensor).to(device)

        #         labels -= 1


        #         # Get predictions
        #         outputs = torch.sigmoid(net(inputs, length))
        #         predictions = torch.round(outputs)

                
        #         loss = criterion(predictions, labels)
                
        #         acc = binary_accuracy(predictions, labels)

        #         epoch_loss += loss.item()
        #         epoch_acc += acc.item()
            
        # valid_loss = epoch_loss / len(testLoader)
        # valid_acc = epoch_acc / len(testLoader)


            # valid_loss, valid_acc = evaluate( testLoader, criterion)

        # for i, batch in enumerate(trainLoader):
        #     # Get a batch and potentially send it to GPU memory.
        #     inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
        #         device), batch.label.type(torch.FloatTensor).to(device)

        #     labels -= 1

        #     # PyTorch calculates gradients by accumulating contributions to them (useful for
        #     # RNNs).  Hence we must manually set them to zero before calculating them.
        #     optimiser.zero_grad()

        #     # Forward pass through the network.
        #     output = net(inputs, length)

        #     loss = criterion(output, labels)

        #     # Calculate gradients.
        #     loss.backward()

        #     # Minimise the loss according to the gradient.
        #     optimiser.step()

        #     running_loss += loss.item()


        #     if i % 32 == 31:
        #         print("Epoch: %2d, Batch: %4d, Loss: %.3f" % (epoch + 1, i + 1, running_loss / 32))
                
        #         running_loss = 0

        # loss_values.append(running_loss)
        
        # num_correct = 0
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)



        # if valid_loss < best_valid_loss:
        
          # best_valid_loss = valid_loss
        
        # torch.save(net.state_dict(), './model'+str(epoch)+'.pth')
    
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        # print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    torch.save(net.state_dict(), './model.pth')

    
    # Evaluate network on the test dataset.  We aren't calculating gradients, so disable autograd to speed up
    # computations and reduce memory usage.
    # with torch.no_grad():
    #     for batch in testLoader:
    #         # Get a batch and potentially send it to GPU memory.
    #         inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
    #             device), batch.label.type(torch.FloatTensor).to(device)

    #         labels -= 1

    #         # Get predictions
    #         outputs = torch.sigmoid(net(inputs, length))
    #         predicted = torch.round(outputs)

    #         num_correct += torch.sum(labels == predicted).item()

    # accuracy = 100 * num_correct / len(dev)

    # print(f"Classification accuracy: {accuracy}")
    # plt.plot(loss_values)

if __name__ == '__main__':
    main()


