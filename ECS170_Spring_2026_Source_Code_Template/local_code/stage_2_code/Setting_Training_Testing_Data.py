'''
Concrete SettingModule class for a specific experimental SettingModule
'''
from local_code.base_class.evaluate import evaluate
from local_code.base_class.method import method
# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting
import numpy as np

class Setting_Training_Testing_Data(setting):
    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_training_data = self.dataset.load()
        loaded_testing_data = self.dataset.load_test_data()


        score_list = []

        print('************ Test: ************')
        #X_train, X_test = np.array(loaded_data['X']), np.array(loaded_data['X'])[test_index]
        X_train = np.array(loaded_training_data['X'])
        X_test = np.array(loaded_testing_data['X'])
        y_train = np.array(loaded_training_data['y'])
        y_test = np.array(loaded_testing_data['y'])

        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result
        self.evaluate.evaluate()

        self.method.accuracy_evaluator.data = learned_result
        self.method.accuracy_evaluator.evaluate()

        self.method.save_readable_data_overall(self.method.method_description)
        score_list.append(self.evaluate.accuracy)

        return np.mean(score_list), np.std(score_list)

        