import pandas as pd
import numpy as np
import tensorflow as tf
import math
from keras.models import load_model

model_path = "./model/seq_encoder_model/"
data_path = "./data/"
raw_data_path = data_path+"raw_data/"
seq_dict = {'T': 0.25, 'C': 0.5, 'A': 0.75, 'G': 1}
n_classes = 5
#load encoder model
encoder_model = load_model(model_path+"seq_encoder_model.h5")
decoder_model = load_model(model_path+"seq_decoder_model.h5")

#prepare set
seq_value_list = []
seq_label_list = []
for i in range(n_classes):
    cls = i+1
    seq_data = pd.read_csv(raw_data_path+'{cls}.txt'.format(cls=cls), header=None, names=['sequence'])
    seq_class_value_list = []
    for inx, cum in seq_data.iterrows():
        seq = cum.sequence
        seq_value_sublist = []
        for j in seq:
            seq_value_sublist.append(seq_dict[j])
        seq_class_value_list.append(seq_value_sublist)
    seq_value_list.append(np.array(seq_class_value_list))
    seq_label_list.append(np.ones(len(seq_data))*i)
pre_x_train = np.concatenate(seq_value_list)
pre_y_train = np.concatenate(seq_label_list)
pre_x_train = encoder_model.predict(pre_x_train)

#################split set####################
from sklearn.model_selection import train_test_split
SEED = 42
x_train, x_val, y_train, y_val = train_test_split(pre_x_train, pre_y_train,
                                                  random_state=SEED,
                                                  test_size=0.2,
                                                  stratify=pre_y_train)
print(x_train.shape, x_val.shape)

# %% --------------------------------------- Save as .npy --------------------------------------------------------------
# Save
np.save(data_path+"x_train.npy", x_train)
np.save(data_path+"y_train.npy", y_train)
np.save(data_path+"x_val.npy", x_val)
np.save(data_path+"y_val.npy", y_val)