import pandas as pd
import numpy as np
import tensorflow as tf
import math
from keras.models import Model, load_model
from keras.layers import Dense, Reshape, Input, Flatten

##################load data and transform to value################
data_path = "./data/raw_data/cRBS_short.csv"
model_path="./model/seq_encoder_model/"
seq_dict = { 'T' : 0.25, 'C' : 0.5, 'A' : 0.75, 'G' : 1}  #dict for transform
value_data = []      #value data
encoder_dim=(64, 64, 1) #target dims
seq_data = pd.read_csv(data_path, header=None, names=['sequence'])
for inx,cum in seq_data.iterrows():
  sub_value_data = []
  sequence = cum.sequence
  for i in sequence:
    sub_value_data.append(seq_dict[i])
  value_data.append(sub_value_data)
x_train = tf.convert_to_tensor(np.array(value_data))

###################define and training autoencoder model#######################
input_seq = Input(shape=(10,))
#seq_encoder
seq_encoder = Dense(128, activation="relu", name="encoder_dense_1")(input_seq)
seq_encoder = Dense(512, activation="relu", name="encoder_dense_2")(seq_encoder)
seq_encoder = Dense(64*64, activation="relu", name="encoder_dense_3")(seq_encoder)
seq_encoder_output = Reshape(encoder_dim)(seq_encoder)
#seq_decoder
seq_decoder = Flatten()(seq_encoder_output)
seq_decoder = Dense(512, activation="relu", name="decoder_dense_1")(seq_decoder)
seq_decoder = Dense(128, activation="relu", name="decoder_dense_2")(seq_decoder)
output_seq = Dense(10, activation="tanh", name="decoder_dense_3")(seq_decoder)
#seq_autoencoder
autoencoder_model = Model(inputs=input_seq,outputs=output_seq)
autoencoder_model.compile(optimizer='adam', loss='mse')
autoencoder_model.fit(x_train, x_train, epochs=100, batch_size=64, shuffle=True)
autoencoder_model.summary()
#save autoencoder model
autoencoder_model.save(model_path+'seq_autoencoder_model.h5')

##################load weights for encoder model and decoder model

#encode model
encoder_input = Input(shape=(10,), name="encoder_input")
encoder = Dense(128, activation="relu", name="encoder_dense_1")(encoder_input)
encoder = Dense(512, activation="relu", name="encoder_dense_2")(encoder)
encoder = Dense(64*64, activation="relu", name="encoder_dense_3")(encoder)
decoder_output = Reshape(encoder_dim)(encoder)
encoder_model=Model(inputs=encoder_input, outputs=decoder_output)
encoder_model.load_weights(model_path+"seq_autoencoder_model.h5", by_name=True)
encoder_model.summary()

#decoder_model
decoder_input = Input(shape=(64, 64, 1), name="decoder_input")
decoder = Flatten()(decoder_input)
seq_decoder = Dense(512,activation="relu", name="decoder_dense_1")(decoder)
seq_decoder = Dense(128,activation="relu", name="decoder_dense_2")(seq_decoder)
decoder_output = Dense(10,activation="tanh", name="decoder_dense_3")(seq_decoder)
decoder_model = Model(inputs=decoder_input, outputs=decoder_output)
decoder_model.load_weights(model_path+"seq_autoencoder_model.h5", by_name=True)
decoder_model.summary()

#save encode/decode model
encoder_model.save(model_path+'seq_encoder_model.h5')
decoder_model.save(model_path+'seq_decoder_model.h5')