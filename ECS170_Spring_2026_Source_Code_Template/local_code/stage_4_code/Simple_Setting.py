import os
from local_code.stage_4_code.tokenizer import *

class Simple_Setting():
    model = None

    def __init__(self,datapath,mod):
        self.model = mod
        self.model.dataset = create_dataset(datapath + "/train")
        self.model.testing_set = create_dataset(datapath + "/test")


    def train(self):
        self.model.start()



