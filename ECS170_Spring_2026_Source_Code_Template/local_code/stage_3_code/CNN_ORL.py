import torch
import numpy as np
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.optim as optim
import torch.nn.functional as f
from torch.utils.data import DataLoader, TensorDataset

class CNN_ORL(nn.Module):
    data = None
    transform_function = None
    test_transform = None

    def __init__(self):
        #self.transform_function = transforms.ToTensor()
        # This function gives the images a bit more variation when creating the image matrix

        self.transform_function = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((92, 92)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((92, 92)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
        # Batch normalization decreases variation to help with learning
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32,64,5, padding=2)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 11 * 11, 450)
        self.fc2 = nn.Linear(450,250)
        self.fc3 = nn.Linear(250,40)


    def forward(self,x):
        x = self.pool(self.bn1(f.relu(self.conv1(x))))
        x = self.pool(self.bn2(f.relu(self.conv2(x))))
        x = self.pool(self.bn3(f.relu(self.conv3(x))))
        x = torch.flatten(x,1)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x


    def training_process(self):
        # .train() ensures our model is in training mode
        self.train()
        max_epoch = 15
        learning_rate = 5e-4
        #learning_rate = 0.001
        print("*******Starting Training*******\n")
        loss_function = nn.CrossEntropyLoss()
        #optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        optimizer = optim.Adam(self.parameters(), learning_rate)

        batch_size = 15

        for epoch in range(max_epoch + 1):
            t_data, l_data = self.create_tensor_array('train')
            t_data = torch.stack(t_data)
            l_data = torch.tensor(l_data)
            adjusted_dataset = TensorDataset(t_data, l_data)
            dataloader = DataLoader(adjusted_dataset, batch_size, shuffle=True)

            print("epoch: {}\n***************\n".format(epoch))
            for i, batch in enumerate(dataloader):
                matrix,label = batch

                optimizer.zero_grad()

                prediction = self.forward(matrix)

                loss = loss_function(prediction, label)
                loss.backward()

                optimizer.step()
                # Print the current loss for each 1000 mini batches
                if i % 1000 == 0 :
                    print("current loss: {}\n".format(loss.item()))
            #self.testing_process()
        print("Training finished")


    def testing_process(self):
        # .eval() takes out model out of learning mode so neurons should not be affected by testing data
        self.eval()

        print("*******Starting Testing*******\n")
        t_data, l_data = self.create_tensor_array('test')
        t_data = torch.stack(t_data)
        l_data = torch.tensor(l_data)
        adjusted_dataset = TensorDataset(t_data, l_data)

        correct = 0
        total = 0
        with torch.no_grad():
            for i, instance in enumerate(adjusted_dataset):
                total += 1
                image,label = instance

                # Forward expects a batch dimension so we need to add one here to each image matrix
                image = image.unsqueeze(0)

                _,prediction = torch.max(self.forward(image),1)
                if i % 1000 == 0 and 0:
                    print("Image {} \n Predicted label: {} \n Actual Label: {}\n".format(i,prediction.item(),label))

                if prediction.item() == label:
                    correct += 1

        print("***************\n")
        print("Testing finished")
        print("Score: {}/{}\n".format(correct,total))
        print("accuracy: {}\n".format(correct/total))
        print("***************\n")


    def create_tensor_array(self,t_type):
        tensor_array = []
        label_array = []

        # Training data needs to use the transform function that adds variation to the data
        if t_type == 'test':
            for matrix in self.data[t_type]:
                tensor_array.append(self.test_transform(matrix['image']).float())
                label_array.append(torch.tensor(matrix['label'] - 1).long())
        else:
            for matrix in self.data[t_type]:
                tensor_array.append(self.transform_function(matrix['image']).float())
                label_array.append(torch.tensor(matrix['label'] - 1).long())

        return tensor_array,label_array


    def start(self, dataset):
        print("running...")
        self.data = dataset
        self.training_process()
        self.testing_process()

