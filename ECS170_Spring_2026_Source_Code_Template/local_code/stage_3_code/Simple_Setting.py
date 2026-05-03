import pickle


class Simple_Setting():
    model = None
    dataset = None

    def __init__(self,datapath,mod):
        self.model = mod
        f = open(datapath, 'rb')
        self.dataset = pickle.load(f)
        f.close()

    def train(self):
        self.model.start(self.dataset)
        pass



