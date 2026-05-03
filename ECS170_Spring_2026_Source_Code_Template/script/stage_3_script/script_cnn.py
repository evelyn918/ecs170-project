import os
import sys
import numpy as np
import torch

from local_code.stage_3_code.CNN_RGB import CNN_RGB
from local_code.stage_3_code.CNN_Grey import CNN_Grey
from local_code.stage_3_code.CNN_ORL import CNN_ORL
from local_code.stage_3_code.Simple_Setting import Simple_Setting


# get the directory of the current script
file_path = os.path.dirname(__file__)
# Change the working directory two files out (ECS170_Spring_2026_Source_Code_Template)
working_directory = os.path.join(file_path,'../../')
os.chdir(working_directory)
# Change the system path two files out (ECS170_Spring_2026_Source_Code_Template)
sys.path.append(str(working_directory))
sys.path.insert(0, str(working_directory))
#print(os.getcwd())


if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------
    m_number = input(
        "Enter the model you would like to use | (1) for RGB/CIFAR (2) for greyscale/MNIST (3) for ORL faces: ")

    colored_data_path = "data/stage_3_data/CIFAR"
    grey_data_path = "data/stage_3_data/MNIST"
    orl_data_path = "data/stage_3_data/ORL"
    match int(m_number):
        case 1:
            model = CNN_RGB()
            setting = Simple_Setting(colored_data_path, model)
            setting.train()
        case 2:
            model2 = CNN_Grey()
            setting2 = Simple_Setting(grey_data_path, model2)
            setting2.train()
        case 3:
            model3 = CNN_ORL()
            setting3 = Simple_Setting(orl_data_path, model3)
            setting3.train()