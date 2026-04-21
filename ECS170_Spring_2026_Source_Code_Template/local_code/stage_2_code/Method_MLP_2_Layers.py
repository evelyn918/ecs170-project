'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


class Method_MLP_2_Layers(method, nn.Module):
    data = None
    accuracy_evaluator = None
    # it defines the max rounds to train the model
    max_epoch = None
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    #learning_rate = 1e-1

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription, epoch):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.max_epoch = epoch
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # Feature that adds the number of layers
        self.fc_layer_1 = nn.Linear(784, 392)
        # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        self.activation_func_1 = nn.ReLU()
        self.fc_layer_2 = nn.Linear(392, 10)
        # check here for nn.Softmax doc: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        self.activation_func_2 = nn.Softmax(dim=1)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        # hidden layer embeddings
        h = self.activation_func_1(self.fc_layer_1(x))

        # outout layer result
        # n (denotes the input instance number): 0th dimension; 2 (denotes the class number): 1st dimension
        # we do softmax along dim=1 to get the normalized classification probability distributions for each instance

        y_pred = self.activation_func_2(self.fc_layer_2(h))
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        self.prepare_save_file()
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # SGD returns a very low accuracy
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # save loss values
        loss_val = []
        # for training accuracy investigation purpose
        self.accuracy_evaluator = Evaluate_Metrics('training evaluator', '')

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            y_pred = self.forward(torch.FloatTensor(np.array(X)))
            # convert y to torch.tensor as well
            y_true = torch.LongTensor(np.array(y))
            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)

            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()
            # add to loss_val
            loss_val.append(train_loss.item())

            if epoch%10 == 0:
                self.accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                self.accuracy_evaluator.evaluate()
                print(' '*9,'Epoch', epoch,'\n',
                        '_' * 25,'\n',
                        'Accuracy:', self.accuracy_evaluator.accuracy,'\n',
                        'Precision:', self.accuracy_evaluator.precision, '\n',
                        'Recall:', self.accuracy_evaluator.recall, '\n',
                        'F1:', self.accuracy_evaluator.f1, '\n',
                        'Loss:', train_loss.item())
            self.save_readable_data(self.accuracy_evaluator,self.method_description, epoch, train_loss.item())
        # convergence curve plot
        plt.figure()
        plt.plot(range(self.max_epoch),loss_val)
        plt.xlabel('Epoch')
        plt.ylabel('Loss Values')
        plt.title('Learning Convergence Plot')
        plt.show()

    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}

    def save_readable_data(self, accuracy_evaluator, model_type, epoch, loss):
        path = "result/stage_2_result/readable_data/" + model_type + ".txt"
        with open(path, 'a', encoding='utf-8') as file:
            file.write((' ' * 9) + 'Epoch '+ str(epoch) + '\n'+
                       ('_' * 25) + '\n' +
                      'Accuracy:'+ str(accuracy_evaluator.accuracy) + '\n' +
                      'Precision:' + str(accuracy_evaluator.precision) + '\n' +
                      'Recall:' + str(accuracy_evaluator.recall) + '\n' +
                      'F1:' + str(accuracy_evaluator.f1) + '\n' +
                      'Loss:' + str(loss) + '\n\n')

    def save_readable_data_overall(self, model_type):
        path = "result/stage_2_result/readable_data/" + model_type + ".txt"
        with open(path, 'a', encoding='utf-8') as file:
            file.write('************ Final Performance ************'+ '\n')
            file.write('MLP Accuracy: ' + str(self.accuracy_evaluator.accuracy) + '\n')
            file.write('Precision: ' + str(self.accuracy_evaluator.precision)+ '\n')
            file.write('Recall: ' + str(self.accuracy_evaluator.recall)+ '\n')
            file.write('f1: ' + str(self.accuracy_evaluator.f1)+ '\n')
            file.write('************ Finish ************'+ '\n')

    def prepare_save_file(self):
        path = "result/stage_2_result/readable_data/" + self.method_description + ".txt"
        with open(path, 'w', encoding='utf-8') as file:
            file.write("Data for model with {}\n".format(self.method_description))
            file.write("*" * 50 + "\n\n\n")