 #!/usr/bin/env python
# coding: utf-8

# In[2]:


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

mode = 'train'
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)


# In[1]:


#!pip install tensorflow-gpu


# In[ ]:


#import tensorflow as tf
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# In[ ]:


#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
  # Restrict TensorFlow to only use the first GPU
#  try:
#    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
#  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
#    print(e)


# In[9]:


#tf.debugging.set_log_device_placement(True)

#try:
  # Specify an invalid GPU device
#  with tf.device('/device:GPU:2'):
#    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
#    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
#    c = tf.matmul(a, b)
#except RuntimeError as e:
#  print(e)


# In[3]:


num_epochs = 100 #100 #5000
batch_size = 64
learning_rate = 1e-4 #1e-4
decay_rate = 1e-3

latent_dim = 512 #2*2*2
epsilon_mean = 0.1
epsilon_std = 1e-4  #http://localhost:8937/notebooks/VAE-copy4.ipynb#


# In[4]:


swe_data = np.log10(np.load("Zeldotest.np.npy"))
# output = (output - output.min() )/(output.max() - output.min())

# swe_train = output[0:800,:,:]
# swe_valid = output[800:1000,:,:]

swe_data = swe_data.reshape(1000,64*64)
# swe_data = (swe_data - np.min(swe_data))/(swe_data.max() - swe_data.min())
# swe_valid = swe_valid.reshape(200,64,64,1)


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


# In[5]:


print(swe_valid.shape)


# In[6]:


plt.imshow(swe_valid[20, :, :, 0])


# In[7]:


swe_valid_data.max()


# In[8]:


def model_def():
    
    def coeff_determination(y_pred, y_true): #Order of function inputs is important here        
        SS_res =  K.sum(K.square( y_true-y_pred )) 
        SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
        return ( 1 - SS_res/(SS_tot + K.epsilon()) )

    # reparameterization trick
    # instead of sampling from Q(z|X), sample eps = N(0,I)
    # then z = z_mean + sqrt(var)*eps
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
#     x = Conv2D(256,kernel_size=(5,5),activation='relu',padding='same')(encoder_inputs)
    x = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same')(encoder_inputs)
    x = Conv2D(128,kernel_size=(2,2),activation='relu',padding='same')(encoder_inputs)
    x = Conv2D(128,kernel_size=(1,1),activation='relu',padding='same')(encoder_inputs)

    x = MaxPooling2D(pool_size=(2, 2),padding='same')(x)

    x = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2),padding='same')(x)

#     x = Conv2D(64,kernel_size=(3,3),activation='relu',padding='same')(x)
#     x = MaxPooling2D(pool_size=(2, 2),padding='same')(x)

#     x = Conv2D(15,kernel_size=(3,3),activation='relu',padding='same')(enc_l4)
#     enc_l5 = MaxPooling2D(pool_size=(2, 2),padding='same')(x)

#     x = Conv2D(10,kernel_size=(3,3),activation=None,padding='same')(enc_l5)
#     encoded = MaxPooling2D(pool_size=(2, 2),padding='same')(x)

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
    x = Reshape((16, 16, 2))(x)
       
#     x = Conv2D(2,kernel_size=(3,3),activation=None,padding='same')(x)
#     dec_l1 = UpSampling2D(size=(2, 2))(x)

#     x = Conv2D(15,kernel_size=(3,3),activation='relu',padding='same')(dec_l1)
#     dec_l2 = UpSampling2D(size=(2, 2))(x)

#     x = Conv2D(64,kernel_size=(3,3),activation='relu',padding='same')(x)
#     x = UpSampling2D(size=(2, 2))(x)

    x = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)

#     x = Conv2D(256,kernel_size=(5,5),activation='relu',padding='same')(x)
    x = Conv2D(256,kernel_size=(1,1),activation='relu',padding='same')(x)
    x = Conv2D(256,kernel_size=(2,2),activation='relu',padding='same')(x)
    x = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)

    decoded = Conv2D(1,kernel_size=(3,3),activation=None,padding='same')(x)
    decoder = Model(inputs=latent_inputs,outputs=decoded)
    decoder.summary()
    # instantiate VAE model
    ae_outputs = decoder(encoder(encoder_inputs))
    model = Model(inputs=encoder_inputs,outputs=ae_outputs,name='VAE')

    # Losses and optimization
    my_adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate, amsgrad=False)
    
    
    
    # Compute VAE loss
    def my_vae_loss(y_true, y_pred):
        reconstruction_loss = mse(K.flatten(y_true), K.flatten(y_pred))

        kl_loss = 1.0 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        return vae_loss

    model.compile(optimizer=my_adam, loss = my_vae_loss, metrics=[coeff_determination])

    model.summary()

    return model, decoder, encoder


# In[9]:


model,decoder,encoder = model_def()


# In[ ]:





# In[ ]:


weights_filepath = 'best_weights_vae.h5'
if mode == 'train':
    checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min',save_weights_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
    callbacks_list = [checkpoint,earlystopping]
    train_history = model.fit(x=swe_train_data, y=swe_train_data, epochs=num_epochs, batch_size=batch_size, callbacks=callbacks_list, validation_split=0.1)

    print('Training complete')
        # model.load_weights(weights_filepath)


# In[1]:


# train_history = model.fit(x=swe_train_data, y=swe_train_data, epochs=num_epochs, batch_size=batch_size, validation_split=0.1)


# In[2]:


if mode == 'train':
     fig1 = plt.figure()
     plt.plot(train_history.history['loss'],'r')
     plt.plot(train_history.history['val_loss'])
        
plt.savefig('VAE_wide_bottleneck.png')


# In[3]:


generator = model.predict(swe_valid[0:10])


# In[4]:


print(generator.shape)


# In[5]:


indx = 1

f, a = plt.subplots(1, 3, figsize = (16,5))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=None)

a[0].imshow(generator[indx,:,:,0])

a[1].imshow(swe_valid[indx,:,:,0])

a[2].imshow(generator[indx,:,:,0] - swe_valid[indx,:,:,0])


# In[ ]:


generator_train = model.predict(swe_train[0:10])


# In[ ]:


indx = 3
plt.imshow(generator_train[indx,:,:,0])


# In[ ]:





# In[ ]:




