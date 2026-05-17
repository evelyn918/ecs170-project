import os
import sys
from asyncio.windows_events import NULL

import numpy as np
import torch

from local_code.stage_4_code.Simple_Setting import Simple_Setting
from local_code.stage_4_code.RNN_Classification import RNN_Classification


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
    f = "data/stage_4_data/text_classification"

    model = RNN_Classification()
    setting = Simple_Setting(f, model)
    setting.train()