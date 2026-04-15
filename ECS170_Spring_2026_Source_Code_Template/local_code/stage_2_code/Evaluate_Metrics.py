'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


class Evaluate_Metrics(evaluate):
    average = 'macro'
    data = None
    accuracy = None
    precision = None
    recall = None
    f1 = None

    def evaluate(self):
        print('evaluating performance...\n')
        self.accuracy = accuracy_score(self.data['true_y'], self.data['pred_y'])
        self.precision = precision_score(self.data['true_y'], self.data['pred_y'], average= self.average, zero_division=0)
        self.recall = recall_score(self.data['true_y'], self.data['pred_y'], average= self.average)
        self.f1 = f1_score(self.data['true_y'], self.data['pred_y'], average= self.average)
        return
        