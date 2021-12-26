

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from keras.layers.core import Activation
from keras.layers import LeakyReLU,BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from scipy.stats import pearsonr
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_file', type=str, default='./seq/WGAN-GP_generated_cRBSs.fas', 
                        help='fluorescence intensity on 0 and 20 g/l glucarate will be predicted by four regression models')
 
    args = parser.parse_args()
    return args


class PREDICT():  
    
    def __init__(self,file_input):  
        self.file = file_input
        self.model_weight = './weight/weight_MSE_N_20_1201_cnn5.h5'
        self.CNN_train_num = 6000
        self.shuffle_flag = 2
        
    
    def CNN_model(self,cRBS_length):
        model = Sequential()
        model.add(
                Conv2D(100, (6, 1),                
                padding='same',
                input_shape=(50, 1, 4))
                )
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 1)))
        model.add(Conv2D(200, (5, 1),padding='same'))
        model.add(LeakyReLU(alpha=0.1
                            ))
        model.add(Conv2D(200, (5, 1),padding='same'))
        model.add(LeakyReLU(alpha=0.1
                            ))
        model.add(MaxPooling2D(pool_size=(2, 1)))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.1
                            ))
        model.add(Dropout(0.4))
        model.add(Dense(1))
        return model


    def seq2onehot(self,seq):     
        module = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        i = 0
        cRBS_onehot = []
        while i < len(seq):
           tmp = []
           for item in seq[i]:
                if item == 't' or item == 'T':
                    tmp.append(module[0])
                elif item == 'c' or item == 'C':
                    tmp.append(module[1])
                elif item == 'g' or item == 'G':
                    tmp.append(module[2])
                elif item == 'a' or item == 'A':
                    tmp.append(module[3])
                else:
                    tmp.append([0,0,0,0])
           cRBS_onehot.append(tmp)
           i = i + 1
        data = np.zeros((len(seq),50,1,4))
        data = np.float32(data)
        i = 0
        while i < len(seq):
            j = 0
            while j < len(seq[0]):
                data[i,j,0,:] = cRBS_onehot[i][j]
                j = j + 1
            i = i + 1
        return data
    
    
    def CNN_predict(self,seq_onehot):
        model = self.CNN_model(len(seq_onehot[0]))
        model.load_weights(self.model_weight)
        batch_flu = model.predict(seq_onehot,verbose=1)
        return batch_flu  
        

    
    def open_fa(self,file):
        record = []
        f = open(file,'r')
        for item in f:
            if '>' not in item:
                record.append(item[0:-1])
        f.close()
        return record
        
    
    def random_perm(self,seq,flu,shuffle_flag):
        indices = np.arange(seq.shape[0])
        np.random.seed(shuffle_flag)
        np.random.shuffle(indices)
        seq = seq[indices]
        flu = flu[indices]
        return seq,flu
        

    def predict(self):
        seq = self.open_fa(self.file)
        seq_onehot = self.seq2onehot(seq)
        

        flu_CNN = self.CNN_predict(seq_onehot)
        
        return seq,flu_CNN
        

if __name__ == '__main__':
    
    args = parse_args()
    input_file = args.input_file
    
    predictor = PREDICT(input_file)
    
    seq,flu_CNN = predictor.predict()
    
    f = open('Accuracy_seq_Flu_MSE_N_20_1210.txt','w')
    i = 0
    while i < len(seq):
        f.write(seq[i] + '   ' + str(np.round(flu_CNN[i],5)) + '\n')
        i = i + 1
    f.close()
    