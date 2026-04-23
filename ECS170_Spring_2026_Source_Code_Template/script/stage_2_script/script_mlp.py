import os
import sys

# get the directory of the current script
file_path = os.path.dirname(__file__)
# Change the working directory two files out (ECS170_Spring_2026_Source_Code_Template)
working_directory = os.path.join(file_path,'../../')
os.chdir(working_directory)
# Change the system path two files out (ECS170_Spring_2026_Source_Code_Template)
sys.path.append(str(working_directory))
sys.path.insert(0, str(working_directory))
#print(os.getcwd())

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Method_MLP_2_Layers import Method_MLP_2_Layers
from local_code.stage_2_code.Method_MLP_8_Layers import Method_MLP_8_Layers
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_Training_Testing_Data import Setting_Training_Testing_Data
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
def set_up_model(data_object, method, result_object, evaluate_object):
    setting = Setting_Training_Testing_Data('Deep Learning', '')
    setting.prepare(data_object, method, result_object, evaluate_object)
    setting.print_setup_summary()
    mean_score, std_score = setting.load_run_save_evaluate()
    precision = setting.evaluate.precision
    recall = setting.evaluate.recall
    f1 = setting.evaluate.f1
    print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))
    print('f1: ' + str(f1))
    print('************ Finish ************')

if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization section ---------------

    mode = input("Enter a mode -- (1) for 8 Layer mode | (2) for 4 Layer mode | (3) for 2 Layer mode] | (4) for all: ")
    while not mode.isdigit() or int(mode) > 4:
            mode = input("Enter a number 1-4: ")

    epoch = input("Enter the number of epochs: ")
    while not mode.isdigit():
        epoch = input("Enter the number of epochs: ")
    epoch = int(epoch)

    data_obj = Dataset_Loader('Part 2 Dataset', '')
    data_obj.dataset_source_folder_path = 'data/stage_2_data/'
    # if using path root: = 'data/stage_2_data/'
    data_obj.dataset_testing_file = "../../data/stage_2_data/test.csv"
    data_obj.dataset_training_file = "../../data/stage_2_data/train.csv"

    method_obj = Method_MLP('multi-layer perceptron', '4_Layers', epoch)
    method_obj_2 = Method_MLP_2_Layers('multi-layer perceptron', '2_Layers', epoch)
    method_obj_3 = Method_MLP_8_Layers('multi-layer perceptron', '8_Layers', epoch)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = 'result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Training_Testing_Data('Deep Learning', '')
    #setting_obj = Setting_Tra
    # in_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Metrics('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    match int(mode):
        case 1:
            print("********* Running in mode 1 ********")
            set_up_model(data_obj, method_obj_3, result_obj, evaluate_obj)
        case 2:
            print("********* Running in mode 2 ********")
            set_up_model(data_obj, method_obj, result_obj, evaluate_obj)
        case 3:
            print("********* Running in mode 3 ********")
            set_up_model(data_obj, method_obj_2, result_obj, evaluate_obj)

        case _:
            print("********* Running all modes ********")
            print("********* Running in mode 1 ********")
            set_up_model(data_obj, method_obj_3, result_obj, evaluate_obj)
            print("********* Running in mode 2 ********")
            set_up_model(data_obj, method_obj, result_obj, evaluate_obj)
            print("********* Running in mode 3 ********")
            set_up_model(data_obj, method_obj_2, result_obj, evaluate_obj)
    # ------------------------------------------------------

