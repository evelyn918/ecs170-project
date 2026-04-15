from local_code.stage_1_code.Result_Loader import Result_Loader
import os

from script.stage_1_script.script_decision_tree import script_path


if 1:
    result_obj = Result_Loader('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_1_result/SVM_'
    result_obj.result_destination_file_name = 'prediction_result'

    for fold_count in [1, 2, 3, None]:
        result_obj.fold_count = fold_count
        result_obj.load()
        print('Fold:', fold_count, ', Result:', result_obj.data)