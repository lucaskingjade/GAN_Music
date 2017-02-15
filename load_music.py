__author__ = 'qiwang'
import json
import numpy as np
from sklearn.utils import shuffle

class BC4s(object):
    def __init__(self,
                 ratio_train,
                 ratio_val,
                 patch_len,
                 patch_step,
                 pitch_scale=False,
                 is_one_hot_code = False):

        args = locals().copy()
        del args['self']
        self.__dict__.update(args)
        (self.X_train, self.y_train),\
        (self.X_val, self.y_val),\
        (self.X_test, self.y_test)\
        =self.load()

    def load(self):
        data_path = './data/bc4s.json'
        with open(data_path, 'rb') as f:
            data_json = json.load(f)

            if not self.is_one_hot_code:
                keys_dic = np.sort(data_json.keys())
                X = []
                y = []
                for key in keys_dic:
                    M = np.array(data_json[key])
                    assert M.shape[0] == 4
                    Mlen = M.shape[1]
                    istart = 0
                    istop = istart + self.patch_len
                    while istop <= Mlen:
                        X.append(M[:, istart:istop].tolist())
                        y.append(key)
                        istart = istart + self.patch_step
                        istop = istart + self.patch_len
                    if istop > Mlen:
                        M_fin = np.zeros((M.shape[0], self.patch_len), dtype=M.dtype)
                        M_fin[:, :Mlen - istart] = M[:, istart:]
                        # set the values of last NaN columns in M_fin to last columns of M.
                        M_fin[:, Mlen - istart:] = M[:, Mlen - istart - 1, None]
                        X.append(M_fin)
                        y.append(key)
                assert len(X) == len(y)
                X = np.cast['float32'](X)
                if self.pitch_scale:
                    X = X / 108.0
                    # add another normalization method?
                y = np.array(y)
            else:
                raise ValueError("one_hot_code did not implemented!")
            # shuffle X and y and seperate X into Training Set,Validation Set and Test Set
            m,r,c = X.shape
            X = X.reshape(m,r*c)
            shuffle(X, y, random_state=2)
            X = X.reshape(m,r,c)
            num_training_set = int(X.shape[0] * self.ratio_train)
            num_val_set = int(X.shape[0] * self.ratio_val)

            X_train = X[:num_training_set,:,:]
            y_train = y[:num_training_set]
            X_val = X[num_training_set:num_training_set + num_val_set,:,:]
            y_val = y[num_training_set:num_training_set + num_val_set]
            X_test = X[num_training_set + num_val_set:,:,:]
            y_test = y[num_training_set + num_val_set:]
            return (X_train, y_train), (X_val, y_val), (X_test, y_test)


#(x_train, y_train), (x_val, y_val), (x_test, y_test) = load(ratio_train=0.7, ratio_val=0.1,\
#                                                            patch_len=32, patch_step=4)
