from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_Training_Testing_Data import Setting_Training_Testing_Data
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('Part 2 Dataset', '')
    data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    # if using path root: = 'data/stage_2_data/'
    data_obj.dataset_testing_file = "test.csv"
    data_obj.dataset_training_file = "train.csv"

    method_obj = Method_MLP('multi-layer perceptron', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Training_Testing_Data('Deep Learning', '')
    #setting_obj = Setting_Tra
    # in_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Metrics('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')

    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    Precision = setting_obj.evaluate.precision
    Recall = setting_obj.evaluate.recall
    f1 = setting_obj.evaluate.f1

    print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('Precision: ' + str(Precision))
    print('Recall: ' + str(Recall))
    print('f1: ' + str(f1))
    print('************ Finish ************')
    # ------------------------------------------------------
