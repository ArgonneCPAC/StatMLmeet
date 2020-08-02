#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Activate TF2 behavior:
from tensorflow.python import tf2
if not tf2.enabled():
  import tensorflow.compat.v2 as tf
  tf.enable_v2_behavior()
  assert tf2.enabled()

import numpy as np
# import tensorflow as tf

# Set seeds
np.random.seed(10)
# tf.random.set_seed(10)

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
time1 = time.time()

mode = 'train'

num_epochs = 1000
batch_size = 64 #256
learning_rate = 1e-5 #1e-4
decay_rate = 0.0 #0.0

latent_dim = 128 # 512  
epsilon_mean = 0.1 # 0.1
epsilon_std = 1e-4 # 1e-4 #1e-4

input_dim = 128


def shuffle(X, y):
    shuffleOrder = np.arange(X.shape[0])
    np.random.shuffle(shuffleOrder)
    X = X[shuffleOrder]
    y = y[shuffleOrder]
    return X, y, shuffleOrder

np.random.seed(2323)


snapshot_range =  np.array([ 4, 5, 6, 7, 8, 9]) #np.arange(10)
names=['0.001.npy','0.001544452104946379.npy','0.0023853323044733007.npy','0.0036840314986403863.npy','0.005689810202763908.npy','0.0087876393444041.npy','0.013572088082974531.npy','0.02096144000826768.npy','0.03237394014347626.npy','0.049999999999999996.npy']

num_sim_per_snap = 1000 # 10000
test_train_split = 0.9
num_train = np.int(test_train_split*num_sim_per_snap*np.size(snapshot_range) )#100000
num_test = np.int( (1.0 - test_train_split)*num_sim_per_snap*np.size(snapshot_range))

print('num train, num test', str(num_train), str(num_test))

if input_dim == 128: 

    for i in snapshot_range: 
        print("now we're loading the 10 different numpy files.")
        print(str("../data/"+ names[i])+" has a shape of: ")
        
        read_data = np.load(str("../data/"+ names[i]))
        
        print(read_data.shape)
        
        read_data = np.log10(read_data + 1)# + 0.0001   ### setting an offset so that log10 doesn't give -inf.

        
        if i == snapshot_range[0] :
            temp_data=read_data[0:num_sim_per_snap ,:,:]
            temp_idx = i*np.ones(shape=(num_sim_per_snap))
        
        else:
            temp_data = np.concatenate((temp_data, read_data[0:num_sim_per_snap,:,:]))   #each timestep we take 1000, and append all together.
            #print(read_data[0:999,:,:])
            temp_idx = np.concatenate((temp_idx, i*np.ones(shape=(num_sim_per_snap)) ))


        

    # output = (output - output.min() )/(output.max() - output.min())
    print('==== temp index', temp_idx.min(), temp_idx.max())
    print('==== temp index', temp_idx.shape)

    print("Data loading accomplished. The final dataset has a shape of:")
    print(temp_data.shape)
    #plt.imshow(temp_data[44])   ### restore this if you'd like to check the dataset plots.

    # swe_data = np.log10(temp_data)
    # np.random.shuffle(temp_data)

    temp_data_shuffle, temp_idx_shuffle, shuffle_order = shuffle(temp_data, temp_idx)

    swe_data = temp_data_shuffle

    print('==== temp index shuffle', temp_idx_shuffle.min(), temp_idx_shuffle.max())
    print('==== temp index shuffle', temp_idx_shuffle.shape)


    swe_data = swe_data.reshape(temp_data.shape[0],input_dim*input_dim)

    preproc = Pipeline([('stdscaler', StandardScaler())])

    # preproc = Pipeline([('stdscaler', MinMaxScaler())])

    print(np.max(swe_data))
    print(np.min(swe_data))
    print('over')

    swe_train = preproc.fit_transform(swe_data[:num_train,:])
    swe_valid = preproc.transform(swe_data[num_train:num_train+num_test,:])

    temp_idx_shuffle_data = temp_idx_shuffle[:num_train]
    temp_idx_shuffle_valid = temp_idx_shuffle[num_train:num_train+num_test]

    swe_train = swe_train.reshape(num_train, input_dim, input_dim,1)
    swe_valid = swe_valid.reshape(num_test, input_dim, input_dim,1)


# Shuffle - to preserve the order of the initial dataset
swe_train_data = np.copy(swe_train)
swe_valid_data = np.copy(swe_valid)

# np.random.shuffle(swe_train_data)
# np.random.shuffle(swe_valid_data)

print('==== temp index shuffle data', temp_idx_shuffle_data.min(), temp_idx_shuffle_data.max())
print('==== temp index shuffle valid', temp_idx_shuffle_valid.min(), temp_idx_shuffle_valid.max())

print(swe_valid.shape)


plt.imshow(swe_valid[20, :, :, 0])


swe_valid_data.max()


# def model_def():
    
#     def coeff_determination(y_pred, y_true): #Order of function inputs is important here        
#         SS_res =  K.sum(K.square( y_true-y_pred )) 
#         SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
#         return ( 1 - SS_res/(SS_tot + K.epsilon()) )

#     # reparameterization trick
#     # instead of sampling from Q(z|X), sample eps = N(0,I)
#     # then z = z_mean + sqrt(var)*eps
#     def sampling(args):
#         """Reparameterization trick by sampling fr an isotropic unit Gaussian.
#          Arguments
#             args (tensor): mean and log of variance of Q(z|X)
#          Returns
#             z (tensor): sampled latent vector
#         """


#         z_mean, z_log_var = args
#         batch = K.shape(z_mean)[0]
#         dim = K.int_shape(z_mean)[1]
#         # by default, random_normal has mean=0 and std=1.0
#         epsilon = K.random_normal(shape=(batch, dim), mean=epsilon_mean, stddev=epsilon_std)
#         return z_mean + K.exp(0.5 * z_log_var) * epsilon

#     ## Encoder
#     encoder_inputs = Input(shape=(64,64,1),name='Field')
#     # Encode   
#     x = Conv2D(512,kernel_size=(5,5),activation='relu',padding='same')(encoder_inputs)
#     x = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same')(encoder_inputs)
#     x = Conv2D(256,kernel_size=(2,2),activation='relu',padding='same')(encoder_inputs)
#     # x = Conv2D(128,kernel_size=(1,1),activation='relu',padding='same')(encoder_inputs)

#     x = MaxPooling2D(pool_size=(2, 2),padding='same')(x)

#     x = Conv2D(128,kernel_size=(2,2),activation='relu',padding='same')(x)
#     x = MaxPooling2D(pool_size=(2, 2),padding='same')(x)

# #     x = Conv2D(64,kernel_size=(3,3),activation='relu',padding='same')(x)
# #     x = MaxPooling2D(pool_size=(2, 2),padding='same')(x)

# #     x = Conv2D(15,kernel_size=(3,3),activation='relu',padding='same')(enc_l4)
# #     enc_l5 = MaxPooling2D(pool_size=(2, 2),padding='same')(x)

# #     x = Conv2D(10,kernel_size=(3,3),activation=None,padding='same')(enc_l5)
# #     encoded = MaxPooling2D(pool_size=(2, 2),padding='same')(x)

#     x = Flatten()(x)
#     z_mean = Dense(latent_dim, name='z_mean')(x)
#     z_log_var = Dense(latent_dim, name='z_log_var')(x)

#     # use reparameterization trick to push the sampling out as input
#     # note that "output_shape" isn't necessary with the TensorFlow backend
#     z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
#     # instantiate encoder model
#     encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
#     encoder.summary()

#     # build decoder model
#     latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
#     x = Dense(512)(latent_inputs)
#     x = Reshape((16, 16, 2))(x)
       
# #     x = Conv2D(2,kernel_size=(3,3),activation=None,padding='same')(x)
# #     dec_l1 = UpSampling2D(size=(2, 2))(x)

# #     x = Conv2D(15,kernel_size=(3,3),activation='relu',padding='same')(dec_l1)
# #     dec_l2 = UpSampling2D(size=(2, 2))(x)

# #     x = Conv2D(64,kernel_size=(3,3),activation='relu',padding='same')(x)
# #     x = UpSampling2D(size=(2, 2))(x)

#     x = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same')(x)
#     x = UpSampling2D(size=(2, 2))(x)

# #     x = Conv2D(256,kernel_size=(5,5),activation='relu',padding='same')(x)
#     # x = Conv2D(256,kernel_size=(1,1),activation='relu',padding='same')(x)
#     x = Conv2D(256,kernel_size=(2,2),activation='relu',padding='same')(x)
#     x = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same')(x)
#     x = Conv2D(512,kernel_size=(5,5),activation='relu',padding='same')(x)
#     x = UpSampling2D(size=(2, 2))(x)

#     decoded = Conv2D(1,kernel_size=(2,2),activation=None,padding='same')(x)
#     decoder = Model(inputs=latent_inputs,outputs=decoded)
#     decoder.summary()
#     # instantiate VAE model
#     ae_outputs = decoder(encoder(encoder_inputs))
#     model = Model(inputs=encoder_inputs,outputs=ae_outputs,name='VAE')

#     # Losses and optimization
#     my_adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate, amsgrad=False)
    
    
    
#     # Compute VAE loss
#     def my_vae_loss(y_true, y_pred):
#         reconstruction_loss = mse(K.flatten(y_true), K.flatten(y_pred))

#         kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
#         kl_loss = K.sum(kl_loss, axis=-1)
#         kl_loss *= -0.5
#         vae_loss = K.mean(reconstruction_loss + kl_loss)
#         return vae_loss

#     model.compile(optimizer=my_adam, loss = my_vae_loss, metrics=[coeff_determination])

#     model.summary()

#     return model, decoder, encoder


# In[ ]:


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
    encoder_inputs = Input(shape=(input_dim, input_dim,1),name='Field')
    # Encode   
    # x = Conv2D(256,kernel_size=(5,5),activation='relu',padding='same')(encoder_inputs)
    x = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same')(encoder_inputs)
    x = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same')(x)
    x = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same')(x)

    # x = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same')(x)
    # x = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same')(x)
    # x = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same')(x)

    # x = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same')(x)
    # x = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same')(x)
    # x = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same')(x)


    x = MaxPooling2D(pool_size=(2, 2),padding='same')(x)

    x = Conv2D(64,kernel_size=(3,3),activation='relu',padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2),padding='same')(x)

    x = Conv2D(64,kernel_size=(3,3),activation='relu',padding='same')(x)
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
    
    x = Conv2D(64,kernel_size=(3,3),activation='relu',padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x) ## removed for latent space 

#     x = Conv2D(256,kernel_size=(5,5),activation='relu',padding='same')(x)
    x = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same')(x)
    x = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same')(x)
    x = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same')(x)

    # x = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same')(x)
    # x = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same')(x)
    # x = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same')(x)

    # x = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same')(x)
    # x = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same')(x)
    # x = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same')(x)


    x = UpSampling2D(size=(2, 2))(x)

    x = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same')(x)


    # x = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same')(x)
    # x = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same')(x)
    # x = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same')(x)


    x = UpSampling2D(size=(2, 2))(x)


    decoded = Conv2D(1,kernel_size=(3,3),activation=None,padding='same')(x)
    decoder = Model(inputs=latent_inputs,outputs=decoded)
    decoder.summary()
    # instantiate VAE model
    ae_outputs = decoder(encoder(encoder_inputs))
    model = Model(inputs=encoder_inputs,outputs=ae_outputs,name='VAE')

#     # Losses and optimization
#     my_adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate, amsgrad=False)
    
#     # Compute VAE loss
# #     def my_vae_loss(y_true, y_pred):
#     def my_vae_loss(encoder_inputs, ae_outputs):

#         reconstruction_loss = mse(K.flatten(ae_outputs), K.flatten(ae_outputs))

#         kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
#         kl_loss = K.sum(kl_loss, axis=-1)
#         kl_loss *= -0.5
#         vae_loss = K.mean(reconstruction_loss + kl_loss)
#         return vae_loss
#     model.compile(optimizer=my_adam, loss = my_vae_loss, metrics=[coeff_determination])


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


weights_filepath = 'allz_best_weights_vae_1.h5'
if mode == 'train':
    checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min',save_weights_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
    callbacks_list = [checkpoint,earlystopping]
#     train_history = model.fit(x=swe_train_data, y=swe_train_data, epochs=num_epochs, batch_size=batch_size, callbacks=callbacks_list, validation_split=0.1)

    train_history = model.fit(swe_train_data, epochs=num_epochs, batch_size=batch_size, validation_split=0.1)
    model.save_weights('allz_vae_cnn_1')
    print('Training complete')
        # model.load_weights(weights_filepath)

model.load_weights('allz_vae_cnn_1')


# train_history = model.fit(x=swe_train_data, y=swe_train_data, epochs=num_epochs, batch_size=batch_size, validation_split=0.1)


if mode == 'train':
    fig1 = plt.figure()
    plt.plot(train_history.history['loss'],'r')
    plt.plot(train_history.history['val_loss'])
    plt.savefig('allzVAE_hist_1.png')


generator = model.predict(swe_valid[0:10])


print(generator.shape)


indx = 6

f, a = plt.subplots(1, 3, figsize = (16,5))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=None)

a[0].imshow(generator[indx,:,:,0], vmin = np.min(swe_data), vmax = np.max(swe_data))

a[1].imshow(swe_valid[indx,:,:,0], vmin = np.min(swe_data), vmax = np.max(swe_data))

a[2].imshow(generator[indx,:,:,0] - swe_valid[indx,:,:,0])
plt.savefig('allzVAE_gen_1.png')


generator_train = model.predict(swe_train[0:10])


for indx in range(8):

    f, a = plt.subplots(1, 2, figsize = (11,5))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=None)

    a[0].imshow(generator[indx,:,:,0] , vmin = np.min(swe_data), vmax = np.max(swe_data))
    a[1].imshow(swe_valid[indx,:,:,0] , vmin = np.min(swe_data), vmax = np.max(swe_data))

    plt.savefig('allzVAE_gen_1'+str(indx)+'.png')
    plt.clf()

time2 = time.time()

print( str(time2-time1)+' seconds')
print('Code completion' )
#=======

encoded_train = encoder.predict(swe_train)[0]

print(temp_idx_shuffle_data.shape)
print(swe_train.shape)
print(np.shape(encoded_train))


import umap
np.random.seed(5)
reducer = umap.UMAP()
embedding = reducer.fit_transform(encoded_train)
# f, a = plt.subplots(2, 2, figsize = (7, 6))


f, a = plt.subplots(1, 1, figsize = (9, 6))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=None)
# for para_idx in range(5):
sc = a.scatter(embedding[:, 0], embedding[:, 1], c= temp_idx_shuffle_data, s = 20, alpha = 0.5)
plt.colorbar(sc)

plt.savefig('allz_scatter_encoded_umap_1.png')

print(temp_idx_shuffle_data.max(), temp_idx_shuffle_data.min())