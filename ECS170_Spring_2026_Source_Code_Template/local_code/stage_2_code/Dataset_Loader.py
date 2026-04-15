'''
Concrete IO class for a specific dataset
'''
from sympy.codegen.ast import none

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_training_file = None
    dataset_testing_file = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading training data...')
        X = []
        y = []
        f = open(self.dataset_source_folder_path + self.dataset_training_file, 'r')
        for line in f:
            line = line.strip('\n')
            elements = [int(i) for i in line.split(',')]
            X.append(elements[1:])
            y.append(elements[0])
        f.close()
        return {'X': X, 'y': y}

    def load_test_data(self):
        print('loading testing data...')
        X_tr = []
        y_tr = []
        f = open(self.dataset_source_folder_path + self.dataset_testing_file, 'r')
        for line in f:
            line = line.strip('\n')
            elements = [int(i) for i in line.split(',')]
            X_tr.append(elements[1:])
            y_tr.append(elements[0])
        f.close()
        return {'X': X_tr, 'y': y_tr}