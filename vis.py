import reader
import torch
import numpy as np
from torch.autograd import Variable
import random 
import pandas as pd
import sys


class BoundaryVisualizer():
    
    def __init__(self, dataset, model, raw_data):
        """
        model: str path to model checkpoint or already loaded model.
        raw_data: str path to ptb_iterator or ptb_iterator itself.
        dataset: only text8 or coco
        """
        assert dataset == "text8" or dataset == "coco"
        self.dataset = dataset

        try:
            self.model = torch.load(model)
        except AttributeError:
            self.model = model
        except IOError:
            print("Could not find model at {}".format(model))
            raise
        except:
            print("Unexpected error")
            raise
        
        raw_reader = reader.text8_raw_data if self.dataset == "text8" else reader.coco_raw_data

        try:
            print("1")
            self.raw_data = raw_reader(raw_data)
        except TypeError: 
            print("2")
            self.raw_data = raw_data
        except IOError:
            print("Could not find data at {}".format(raw_data))
            raise
        except:
            print("Unexpected error")
            raise
            
        tr, val, te, w2i, i2w = self.raw_data
        self.inps = list(reader.ptb_iterator(val, 1, self.model.num_steps))
        self.i2w = i2w

    def prepare_inp(self, inp, batch_size=1):
        weight = next(self.model.parameters()).data
        Ht = [Variable(weight.new(1, self.model.hidden_size).zero_()) for x in range(self.model.num_layers)]
        C = [Variable(weight.new(1, self.model.hidden_size).zero_()) for x in range(self.model.num_layers)]
        Z = [Variable(weight.new(1).zero_()) for x in range(self.model.num_layers - 1)]
        hidden = [Ht, C, Z]
        emb = self.model.word_embeddings(inp)
        return emb, hidden
    

    def bounds(self, inp, txt):
        inp = Variable(torch.from_numpy(inp.astype(np.int64)).transpose(0, 1).contiguous()).cuda()
        emb, h0 = self.prepare_inp(inp)
        z,g = self.model.lstm(emb, h0, pred_boundaries=True)

    def vis_bounds(self, z, txt):
        out = ""
        for i in range(len(z)):
            out = ""
            z1 = map(int, list(z[i]))
            for c, b in zip(txt, z1):
                if c == "\n":
                    c = " "
                if b == 1:
                    out += c + "|"
                else:
                    out += c

    def run_vis(self, n=10):
        for i in range(n):
            inp, _ = self.inps[random.randint(0, len(self.inps) - 1)]
            txt = [self.i2w[x] for x in inp[0]]
            self.bounds(inp, txt)
    
    def eval_bounds(self, n=100):
        for i in range(n):
            

if __name__ == "__main__":
    #run_vis()
    print("Im not gonna do shit")
