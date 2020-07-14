#!/usr/bin/env python
# coding: utf-8


# Activate TF2 behavior:
from tensorflow.python import tf2
if not tf2.enabled():
  import tensorflow.compat.v2 as tf
  tf.enable_v2_behavior()
  assert tf2.enabled()

import numpy as np

# Set seeds
np.random.seed(10)

from tensorflow.keras.layers import Input, Dense, LSTM, Lambda, Dropout, Flatten, Reshape, Conv2DTranspose
from tensorflow.keras.layers import Conv2D, UpSampling2D, MaxPooling2D

from tensorflow.keras import optimizers, models, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.regularizers import l1
from tensorflow.keras.utils import plot_model
from keras.losses import mse, binary_crossentropy

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import matplotlib.pyplot as plt
import time
mode = 'train'


time1 = time.time()

num_epochs = 100
batch_size = 256 #256
learning_rate = 1e-4 #1e-4
decay_rate = 0.01

latent_dim = 512  
epsilon_mean = 0.1
epsilon_std = 1e-4 #1e-4




swe_data = np.log10(np.load("Zeldotest.npy"))
swe_data = swe_data.reshape(1000,64*64)

preproc = Pipeline([('stdscaler', StandardScaler())])

swe_train = preproc.fit_transform(swe_data[:900,:])
swe_valid = preproc.transform(swe_data[900:,:])
# swe_train = swe_data[:900,:]
# swe_valid = swe_data[900:,:]
swe_train = swe_train.reshape(900,64,64,1)
swe_valid = swe_valid.reshape(100,64,64,1)

# Shuffle - to preserve the order of the initial dataset
swe_train_data = np.copy(swe_train)
swe_valid_data = np.copy(swe_valid)

np.random.shuffle(swe_train_data)
np.random.shuffle(swe_valid_data)


print(swe_valid.shape)

  
def model_def():
    
    def coeff_determination(y_pred, y_true): #Order of function inputs is important here        
        SS_res =  K.sum(K.square( y_true-y_pred )) 
        SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
        return ( 1 - SS_res/(SS_tot + K.epsilon()) )


    def sampling(args):
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.
         Arguments
            args (tensor): mean and log of variance of Q(z|X)
         Returns
            z (tensor): sampled latent vector
        """


        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim), mean=epsilon_mean, stddev=epsilon_std)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    ## Encoder
    encoder_inputs = Input(shape=(64,64,1),name='Field')
    # Encode   
    # x = Conv2D(256,kernel_size=(5,5),activation='relu',padding='same')(encoder_inputs)
    x = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same')(encoder_inputs)
    x = Conv2D(512,kernel_size=(2,2),activation='relu',padding='same')(x)
    x = Conv2D(512,kernel_size=(1,1),activation='relu',padding='same')(x)

    x = MaxPooling2D(pool_size=(2, 2),padding='same')(x)

    x = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2),padding='same')(x)



    x = Flatten()(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    # instantiate encoder model
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(512)(latent_inputs)
    x = Reshape((16, 16, 2))(x)  ## removed for latent space 
    # x = Dense(2048)(latent_inputs)
    # x = Reshape((32, 32, 2))(x)
       


    x = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x) ## removed for latent space 

#     x = Conv2D(256,kernel_size=(5,5),activation='relu',padding='same')(x)
    x = Conv2D(512,kernel_size=(1,1),activation='relu',padding='same')(x)
    x = Conv2D(512,kernel_size=(2,2),activation='relu',padding='same')(x)
    x = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same')(x)
    # x = Conv2D(256,kernel_size=(5,5),activation='relu',padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)

    decoded = Conv2D(1,kernel_size=(3,3),activation=None,padding='same')(x)
    decoder = Model(inputs=latent_inputs,outputs=decoded)
    decoder.summary()
    # instantiate VAE model
    ae_outputs = decoder(encoder(encoder_inputs))
    model = Model(inputs=encoder_inputs,outputs=ae_outputs,name='VAE')

    reconstruction_loss = mse(K.flatten(encoder_inputs), K.flatten(ae_outputs))

    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    model.add_loss(vae_loss)
    model.compile(optimizer='adam')
    K.set_value(model.optimizer.lr, learning_rate)
    K.set_value(model.optimizer.decay, decay_rate)
    model.summary()


    return model, decoder, encoder




model,decoder,encoder = model_def()




weights_filepath = 'best_weights_vae.h5'
if mode == 'train':
    checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min',save_weights_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
    callbacks_list = [checkpoint,earlystopping]

    train_history = model.fit(swe_train_data, epochs=num_epochs, batch_size=batch_size, validation_split=0.1)
    model.save_weights('vae_cnn')
    print('Training complete')

if mode == 'train':
     fig1 = plt.figure()
     plt.plot(train_history.history['loss'],'r')
     plt.plot(train_history.history['val_loss'])
plt.savefig('VAE_hist.png')


generator = model.predict(swe_valid[0:10])

print(generator.shape)


indx = 6

f, a = plt.subplots(1, 3, figsize = (16,5))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=None)

a[0].imshow(generator[indx,:,:,0])

a[1].imshow(swe_valid[indx,:,:,0])

a[2].imshow(generator[indx,:,:,0] - swe_valid[indx,:,:,0])
plt.savefig('VAE_gen.png')


generator_train = model.predict(swe_train[0:10])


for indx in range(8):

    f, a = plt.subplots(1, 2, figsize = (11,5))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=None)

    a[0].imshow(generator[indx,:,:,0])
    a[1].imshow(swe_valid[indx,:,:,0])

    plt.savefig('VAE_gen'+str(indx)+'.png')
    plt.clf()


time2 = time.time()

print( str(time2-time1)+'seconds')
print('Code completion' )

