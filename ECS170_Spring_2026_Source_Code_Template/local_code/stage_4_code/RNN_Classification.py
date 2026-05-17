import torch
import numpy as np
import torch.nn as nn
import torchvision
from sympy.codegen.ast import none
from torchvision import transforms
import torch.optim as optim
import torch.nn.functional as f
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

class RNN_Classification(nn.Module):
    dataset = None
    testing_set = None

    max_sentence_size = 250
    hidden_size = 64
    output_size = 2

    # the number of unique words
    vocab_size = 5
    # size of the embedding vector
    embedding_size = 50

    word_bank = []
    embedding_bank = []

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.load_glove_embedding()
        self.embedding.weight.data.copy_(self.embedding_bank)
        #for i in range(self.vocab_size):
        #   print(self.embedding.weight[i])
        self.rnn = nn.RNN(self.embedding_size, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.output_size)


    def forward(self,x):
        x = self.embedding(x)
        output,_ = self.rnn(x)
        output = self.fc(output[:, -1, :])
        return output


    def training_process(self):
        print("Starting training process...")
        self.train()

        max_epochs = 1
        learning_rate = 0.001

        loss_function = nn.CrossEntropyLoss()

        optimizer = optim.Adam(self.rnn.parameters(), lr=learning_rate)

        text, labels = self.make_tensor_data(self.dataset)
        batch_size = 30

        print(text)

        adjusted_dataset = TensorDataset(text, labels)
        dataloader = DataLoader(adjusted_dataset, batch_size=batch_size, shuffle=True)

        loss_list = []
        for epoch in range(max_epochs):
            print("Epoch: {}".format(epoch))
            print("*" * 10)
            count = 0
            cumulative_loss = 0
            for i,batch in enumerate(dataloader):
                txt, lbl = batch
                prediction = self.forward(txt)

                #print(torch.argmax(prediction,dim=1))

                optimizer.zero_grad()

                loss = loss_function(prediction, lbl)

                loss.backward()
                optimizer.step()

                cumulative_loss += loss.item()
                count += 1
            print(cumulative_loss/count)
            print("\n")

        print("training finished...")


    def testing_process(self):
        print("Starting Testing process...")
        self.eval()
        batch_size = 30
        text, label = self.make_tensor_data(self.testing_set)
        adjusted_dataset = TensorDataset(text, label)

        dataloader = DataLoader(adjusted_dataset, batch_size=batch_size)
        total = 0
        correct = 0
        for data in dataloader:
            txt, lbl = data
            predictions = torch.argmax(self.forward(txt),dim=1)
            for i,guess in enumerate(predictions):
                total += 1
                if guess == lbl[i]:
                    correct += 1

        print("Accuracy: {}".format(correct/total))


    def start(self):
        self.training_process()
        self.testing_process()


    def make_tensor_data(self,dataset):
        text_embedding_numbers_list = []
        labels = []

        for sentence in dataset["text"]:
            if len(sentence) >= self.max_sentence_size:
                text_embedding_numbers = []
                sentence = sentence[:self.max_sentence_size]
                for word in sentence:
                    text_embedding_numbers.append(self.find_word_number(word))

                text_embedding_numbers_list.append(text_embedding_numbers)

            else:
                text_embedding_numbers = []
                entries_to_fill = self.max_sentence_size - len(sentence)
                for word in sentence:
                    text_embedding_numbers.append(self.find_word_number(word))
                for i in range(entries_to_fill):
                    text_embedding_numbers.append(0)

                text_embedding_numbers_list.append(text_embedding_numbers)

        for label in dataset["label"]:
            labels.append(label)

        labels = torch.tensor(labels, dtype=torch.long)
        text_embedding_numbers_list = torch.tensor(text_embedding_numbers_list, dtype=torch.long)

        return text_embedding_numbers_list, labels

    def find_word_number(self,word):
        if word not in self.word_bank:
            return self.word_bank["<PAD>"]
        else:
            return self.word_bank[word]


    def load_glove_embedding(self):
        print("Adjusting embeddings and creating word bank...")
        vocab_word = []
        embedding_number = []

        list_of_embeddings = []

        # Add the padding vector to our list of words
        vocab_word.append("<PAD>")
        embedding_number.append(0)
        x = []
        for i in range(self.embedding_size):
            x.append(0)
        list_of_embeddings.append(x)

        words_added = 1

        with open("data/stage_4_data/gloVe.txt", 'r') as file:
            print("Loading gloVE embeddings and copying...")
            for line in file:
                embedding_values = []
                for j,entry in enumerate(line.split()):
                    if j == 0:
                        vocab_word.append(entry)
                    else:
                        embedding_values.append(float(entry))
                list_of_embeddings.append(embedding_values)
                embedding_number.append(words_added)
                words_added += 1

                if words_added >= self.vocab_size:
                    break

        self.embedding_bank = torch.tensor(list_of_embeddings, dtype=torch.float)
        print("Embeddings updated...")
        self.word_bank = dict(zip(vocab_word, embedding_number))
        print("Word bank updated...")
        print(self.word_bank)
        #print(pretrained_embedding_data)