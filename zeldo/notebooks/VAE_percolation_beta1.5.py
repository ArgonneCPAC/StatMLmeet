#!/usr/bin/env python
# coding: utf-8
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

import tensorflow as tf

# In[19]:



mode = 'train'

#mode = 'test'
#folder = './Downloads/Phoenix_Downloads/connect/'

#folder = './Downloads/Phoenix_Downloads/newconnect/'
#folder_vae = './Downloads/Phoenix_Downloads/connect/'

#folder = './Downloads/Phoenix_Downloads/0921-2/'
#folder_vae = './Downloads/Phoenix_Downloads/0921-2/'

folder = './1029/'

import umap
import time
time1 = time.time()


#folder = './connect/'
print("testing path folder")
plt.figure()
plt.savefig(folder + 'testfigure.png')
plt.clf()
print("test is over.")


def shuffle(X, y):
    shuffleOrder = np.arange(X.shape[0])
    np.random.shuffle(shuffleOrder)
    X = X[shuffleOrder]
    y = y[shuffleOrder]
    return X, y, shuffleOrder

# In[ ]:


num_epochs = 400
batch_size = 32 #256
learning_rate = 1e-4 #1e-4
decay_rate = 0.001

latent_dim = 512 ##32 ##512  
epsilon_mean = 0.1
epsilon_std = 1e-4 #1e-4

beta =  1.5
alpha = 0.1
gamma = 0.1

### Gamma is not used yet


###### normalization factor for power spectra loss #########
renorm_factor = tf.constant(alpha * 1/(128*128*128))


# In[9]:


# In[ ]:

#### This is the original read data line
#read_data = np.load("./0.0500.npy")


data_offset=1.0

names=['0.0010.npy','0.0015.npy','0.0024.npy','0.0037.npy','0.0057.npy','0.0088.npy','0.0136.npy','0.0210.npy','0.0324.npy','0.0500.npy']

for i in range(10):
    print("now we're loading the 10 different numpy files.")
    #print(str("/data/a/cpac/dongx/"+ names[i])+" has a shape of: ")
    print(str("/homes/dongx/"+ names[i])+" has a shape of: ")
    
    take_data = np.load(str("/homes/dongx/"+ names[i]))
    
    print(take_data.shape)
    
    #    take_data = take_data + dataoffset   ### setting an offset so that log10 doesn't give -inf.
    
    if i == 0 :
        temp_data=take_data[0:1000,:,:]
        temp_indx=np.full(1000,i)
    else:
        temp_data = np.concatenate((temp_data, take_data[0:1000,:,:]))   #each timestep we take 1000, and append all together.
        temp_indx = np.concatenate((temp_indx, np.full(1000,i)))
    print(temp_data.shape)
    print(temp_indx.shape)
#print(read_data[0:999,:,:])


# output = (output - output.min() )/(output.max() - output.min())

print("Data loading accomplished. The final dataset has a shape of:")
print(temp_data.shape)
#plt.imshow(temp_data[44])   ### restore this if you'd like to check the dataset plots.

temp_data = temp_data + data_offset
swe_data = np.log10(temp_data)


#read_data = read_data + 1.0001

#swe_data = np.log10(read_data)
# output = (output - output.min() )/(output.max() - output.min())

# swe_train = output[0:800,:,:]
# swe_valid = output[800:1000,:,:]

swe_data = swe_data.reshape(10000,128*128)
# swe_data = (swe_data - np.min(swe_data))/(swe_data.max() - swe_data.min())
# swe_valid = swe_valid.reshape(200,64,64,1)


preproc = Pipeline([('stdscaler', StandardScaler())])

print(np.max(swe_data))
print(np.min(swe_data))
print('over')


#np.random.shuffle(swe_data)


swe_data_shuffle, indx_shuffle, shuffle_order  = shuffle(swe_data, temp_indx)


swe_train = preproc.fit_transform(swe_data_shuffle[:9000,:])
train_indx = indx_shuffle[:9000]


swe_valid = preproc.transform(swe_data_shuffle[9000:,:])
valid_indx = indx_shuffle[9000:]
#swe_train = swe_data[:9000,:]
#swe_valid = swe_data[9000:,:]




swe_train = swe_train.reshape(9000,128,128,1)
swe_valid = swe_valid.reshape(1000,128,128,1)

#plt.figure()
plt.imshow(swe_train[1,:,:,0])
plt.savefig('training.png')


# Shuffle - to preserve the order of the initial dataset
swe_train_data = np.copy(swe_train)
swe_valid_data = np.copy(swe_valid)

#np.random.shuffle(swe_train_data)
#np.random.shuffle(swe_valid_data)


# In[ ]:


print(swe_valid.shape)


# In[ ]:


plt.imshow(swe_valid[20, :, :, 0])


# In[ ]:


swe_valid_data.max()



#################

dim = 128
bin_dim = 200
freqs = np.fft.fftfreq(dim)
binning = (np.amax(freqs)-np.amin(freqs))*np.sqrt(2)/bin_dim

pow_spec = np.zeros(bin_dim)
cnt_spec = np.zeros(bin_dim)
k_spec = np.zeros(bin_dim)
k_matrix = np.zeros((dim,dim)) 
index_matrix = np.zeros((dim,dim)) 



#for i in range(0,dim):
#    for j in range(0,dim):
#        k_val = 


# In[5]:


#import sys


# In[11]:


#np.set_printoptions(threshold=sys.maxsize)


# In[13]:


def generate_p_k_matrices(L):
    
    dim = 128
    bin_dim = 30
    #bin_dim = 200
    freqs = np.fft.fftfreq(dim)
    binning = (np.amax(freqs)-np.amin(freqs))*np.sqrt(2)/bin_dim

    pow_spec = np.zeros(bin_dim)
    cnt_spec = np.zeros(bin_dim)
    k_spec = np.zeros(bin_dim)
    k_matrix = np.zeros((dim,dim)) 
    index_matrix = np.zeros((dim,dim)) 
    
    
    freqs = np.fft.fftfreq(dim)
    ##### SETTING BINNING PARAMETERS ##########
    
    binning = (np.amax(freqs)-np.amin(freqs))*np.sqrt(2)/bin_dim
    print("power spectrum binning="+str(binning))


    pow_spec = np.zeros(bin_dim)
    cnt_spec = np.zeros(bin_dim)
    k_spec = np.zeros(bin_dim)


    freq_ax=np.arange(bin_dim)*binning*np.pi*2
    print(freq_ax)


    for i in range(0,dim):
        for j in range(0,dim):
              k_val = np.sqrt((freqs[i]-freqs[0])**2+(freqs[j]-freqs[0])**2) #### k_val is the magnitude of |k|
              k_matrix[i][j]=k_val  
            
              if int(k_val/binning)> (bin_dim-1) :  
                    print("k_value="+ str(distance) + ",error!")
                    
              else:
                    index_matrix[i][j]= int(k_val/binning)
                    #print("index="+ str(int(k_val/binning)) + ".")
                    
    #print(index_matrix)
    #print("above is index matrix")
    
    print(np.max(index_matrix))
    pow_dim = np.max(index_matrix)
    count_array = np.zeros(int(np.max(index_matrix))+1)   ##### THINK ABOUT +1 index later!!! ######
    Mult_matrix = np.zeros((int(np.max(index_matrix))+1,128,128))
    Cnt_mult = np.zeros((128,128,(int(np.max(index_matrix))+1))) 
    pow_mult = np.zeros((128,128))
    #Fourier_matrix = fft*np.conj(fft)
    B=np.zeros((128,128))
    for i in range(int(np.max(index_matrix))+1):
        A= np.where(index_matrix==i,1,0)

        Mult_matrix[i,:,:] = A
        Cnt_mult[:,:,i]=(1/np.sum(A))*A
        count_array[i]=np.sum(A)
        pow_mult += (1/np.sum(A))*A
        k_spec[i]+=i*binning*2*np.pi          
        #pow_spec[i]+=component/(L*L*L*L*np.sum(A))
        #cnt_spec[i]+=np.sum(A)
        #print(A)
        #component = np.sum(np.multiply(A,Fourier_matrix))
    return index_matrix, Mult_matrix, count_array, k_spec, Cnt_mult, pow_mult, pow_dim   
    
    


# In[14]:


A,B,C,D,E,F,pow_dim=generate_p_k_matrices(128)
#print(np.amax(A))

Cnt_mult_matrix = tf.convert_to_tensor(E)


#print(Cnt_mult_matrix2[:,:,0]-Cnt_mult_matrix[0,:,:])

############ Tile function test ##############
#print(A.shape)
A = tf.reshape(A,[128,128,1])
tile_test = tf.tile(A,[1,1,16])


def map(fn, arrays, dtype=tf.float64):
    # assumes all arrays have same leading dim
    indices = tf.range(tf.shape(arrays[0])[0])
    out = tf.map_fn(lambda ii: fn(*[array[ii] for array in arrays]), indices, dtype=dtype)
    return out


def pow_spec_test(tensor, tensor2):
    renorm_factor = tf.constant((128*128), dtype = tf.float64)
    
    
    L=128
    complex_input = tf.dtypes.cast(tensor, tf.complex128)
    complex_output = tf.dtypes.cast(tensor2, tf.complex128)
 
    #print(complex_input)
    #print("complex_input is above")
    
    avg_input = tf.reduce_mean(complex_input)
    avg_output = tf.reduce_mean(complex_output)
 
    #print(avg_input)
    #print("avg_input is above")

    ovden_input = complex_input - avg_input * tf.ones([128,128,1],tf.complex128)
    ovden_output = complex_output - avg_output * tf.ones([128,128,1],tf.complex128)

    #print(ovden_input)
    #print("ovden_input is above")

    
    fourier_input=tf.signal.fft2d(ovden_input)#/(L*L*L*L)
    fourier_output=tf.signal.fft2d(ovden_output)#/(L*L*L*L)

    #print(fourier_input)
    #print("fourier_input is above") 


    mult_input = tf.math.multiply(tf.math.conj(fourier_input), fourier_input)  #/(L*L*L*L)
    mult_output = tf.math.multiply(tf.math.conj(fourier_output), fourier_output) #/(L*L*L*L)
    
    mult_input = tf.dtypes.cast(mult_input,tf.float64)
    mult_output = tf.dtypes.cast(mult_output,tf.float64)
    
    #print(mult_input)
    #print("mult_input is above")
    
    A = tf.reshape(mult_input,[128,128,1])
    B = tf.reshape(mult_output,[128,128,1])
#    print(A)
#    print("A is above")
    
    
    mult_tile_input = tf.tile(A,[1,1,16])
    mult_tile_output = tf.tile(B,[1,1,16])
    
    #print(mult_tile_input[:,:,0]-mult_tile_input[:,:,1])
    #print("difference is above")
    
    #print(mult_tile_input.shape)
    #print("mult_tile_input  shape is above")
    
    #tile_param = tf.constant([128,15], tf.int32)
    #tile_input = tf.tile(mult_input,[1,1,1,16])

    #print(Cnt_mult_matrix)
    #print("Cnt_mult_matrix, before mult is above")
        
    
    
    
    pow_spec_input = tf.math.multiply(Cnt_mult_matrix,mult_tile_input)
    pow_spec_output = tf.math.multiply(Cnt_mult_matrix,mult_tile_output)
    
    #print(pow_spec_input)
    #print("pow_spec_input before sum is above")
    
    
    pow_spec_input = tf.math.reduce_sum(pow_spec_input,[0,1])
    pow_spec_output = tf.math.reduce_sum(pow_spec_output,[0,1])
    
    #print(pow_spec_input.shape)
    #print("pow_spec_input is above")
    #print(pow_spec_input)
    #print("pow_spec_input is above")
    #print(pow_spec_output)
    #print("pow_spec_output is above")    
    diff_in_out = tf.math.divide(mse(K.flatten(pow_spec_input), K.flatten(pow_spec_output)),renorm_factor)
    
    
    #print(tile_input)
    #print("tile_input is above")    

    #print(Cnt_mult_matrix)
    #print("Cnt_mult_matrix is above")    
    
#    Cnt_mult_matrix
    
    
    #fourier_loss = mse(K.flatten(mult_input), K.flatten(mult_output))
    #vae_loss = tf.dtypes.cast(fourier_loss,tf.float32) + vae_loss
    
    return  diff_in_out #pow_spec_input,
    
    
    


# In[ ]:





# In[ ]:





# In[21]:





# In[ ]:
def tf_repeat(tensor, repeats):
    """
    Args:

    input: A Tensor. 1-D or higher.
    repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input

    Returns:
    
    A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
    """
    with tf.variable_scope("repeat"):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)
        repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    return repeated_tesnor


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
    encoder_inputs = Input(shape=(128,128,1),name='Field')
    # Encode   
    # x = Conv2D(256,kernel_size=(5,5),activation='relu',padding='same')(encoder_inputs)
    x = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same')(encoder_inputs)
    x = Conv2D(128,kernel_size=(2,2),activation='relu',padding='same')(x)
    x = Conv2D(128,kernel_size=(1,1),activation='relu',padding='same')(x)

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
    vae_loss = K.mean(reconstruction_loss + beta * kl_loss)
    
    #hist_edges = tf.linspace( 0, 2, 200, axis=0)
    
    ########################################## Adding metric PDF ##########################
    value_input = K.flatten(encoder_inputs)
    
    value_range = [0.0, 2.0]
    
    hist_input = tf.histogram_fixed_width(value_input, value_range, nbins=200)
    
    value_output = K.flatten(ae_outputs)
    
    hist_output = tf.histogram_fixed_width(value_output, value_range, nbins=200)
    
    hist_loss = 0.01* tf.dtypes.cast(mse(hist_input, hist_output), tf.float32)

    #factor of 0.01 added   

#    vae_loss = hist_loss + vae_loss
    ####################################### Adding metric Power Spectra #####################
    tensor1 = tf.reshape(encoder_inputs,[batch_size,128,128,1])
    tensor2 = tf.reshape(ae_outputs,[batch_size,128,128,1])
    #diff_pow_spec = pow_spec_test(tensor1,tensor2)
    #pow_spec = tf.map_fn(fn=lambda t1,t2: pow_spec_test(t1, t2), elems=(encoder_inputs,ae_outputs))
    f = lambda t1,t2: pow_spec_test(t1,t2) #+ b0
   
    batch_y = map(f, [tensor1, tensor2])


    diff_pow_spec = tf.math.reduce_sum(batch_y)
    pow_spec_loss = tf.dtypes.cast(diff_pow_spec,tf.float32)*renorm_factor
    #print("pow_spec_loss shape is below!!!!")
    #print(pow_spec_loss.shape)
    vae_loss = vae_loss + pow_spec_loss

    
    ##########################################################
    model.add_loss(vae_loss)
    #model.add_loss(pow_spec_loss)
    model.compile(optimizer='adam')
    K.set_value(model.optimizer.lr, learning_rate)
    K.set_value(model.optimizer.decay, decay_rate)
    model.summary()


    return model, decoder, encoder


# In[ ]:


model,decoder,encoder = model_def()


# In[ ]:


weights_filepath = folder +'best_weights_vae.h5'
if mode == 'train':
    checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min',save_weights_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
    callbacks_list = [checkpoint,earlystopping]
#     train_history = model.fit(x=swe_train_data, y=swe_train_data, epochs=num_epochs, batch_size=batch_size, callbacks=callbacks_list, validation_split=0.1)

    train_history = model.fit(swe_train_data, epochs=num_epochs, batch_size=batch_size, callbacks=callbacks_list, validation_split=0.1)
    model.save_weights(folder + 'vae_cnn')
    print('Training complete')
        # model.load_weights(weights_filepath)

if mode == 'test':
    model.load_weights(folder+'vae_cnn')


# In[ ]:


# train_history = model.fit(x=swe_train_data, y=swe_train_data, epochs=num_epochs, batch_size=batch_size, validation_split=0.1)


# In[ ]:



if mode == 'train':
     fig1 = plt.figure()
     plt.plot(train_history.history['loss'],'r')
     plt.plot(train_history.history['val_loss'])
plt.savefig( folder + 'VAE_hist.png')


# In[ ]:




###### INSERTED!! CAN DELETE ########


# In[6]:


generator = model.predict(swe_train[0:50])

swe_valid = swe_train

# In[ ]:


print(generator.shape)


# In[ ]:


indx = 6

f, a = plt.subplots(1, 3, figsize = (16,5))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=None)

a[0].imshow(generator[indx,:,:,0])

a[1].imshow(swe_train[indx,:,:,0])

a[2].imshow(generator[indx,:,:,0] - swe_train[indx,:,:,0])
plt.savefig(folder + 'VAE_gen.png')

np.save(folder+'VAE_gen_',generator[indx,:,:,0])

# In[ ]:


generator_train = model.predict(swe_train[0:10])


# In[ ]:


for indx in range(50):

    f, a = plt.subplots(1, 2, figsize = (11,5))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=None)

    a[0].imshow(generator[indx,:,:,0])
    a[1].imshow(swe_train[indx,:,:,0])

    plt.savefig(folder+'VAE_gen_connect_'+str(indx)+'.png')
    plt.clf()
    np.save(folder+'VAE_gen_connect_'+str(indx), generator[indx,:,:,0])
    np.save(folder+'VAE_gen_'+str(indx), swe_train[indx,:,:,0])


# In[19]:




#encoded_valid = encoder.predict(swe_train)[0]
 
encoded_valid = encoder.predict(swe_train)[0]   
print(encoded_valid.shape)
 
    
batch_num = 50
encoded_valid = encoded_valid[0:batch_num*10,:]   
umap_type= 'valid'
distmin = 0.5
neighbor = 200

if umap_type == 'valid':
                                   reducer = umap.UMAP(n_neighbors=neighbor, min_dist=distmin)
                                   #reducer = umap.UMAP()
                                   embedding = reducer.fit_transform(encoded_valid)
                                   f, a = plt.subplots(1, 1, figsize = (9, 6))
                                   plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=None)
                                                                      # for para_idx in range(5):

                                   sc = a.scatter(embedding[:, 0], embedding[:, 1], c=temp_indx[0:batch_num*10], s = 20, alpha = 0.5)
                                   print(embedding.shape)
                                   for batch in range(batch_num):
                                       #sc = a.scatter(embedding[batch:batch+10, 0], embedding[batch:batch+10, 1], c=temp_indx[batch:batch+10], s = 20, alpha = 0.5) #
                                        print(batch*10)
                                        print(batch*10+9)
                                        
                                        plt.plot(embedding[batch*10:batch*10+10, 0], embedding[batch*10:batch*10+10, 1],label= str(batch) )
                                   cbar = plt.colorbar(sc)
                                   cbar.set_label('Time-step', rotation=270)
                                   plt.ylabel('Z2')
                                   plt.xlabel('Z1')
                                   plt.legend()
                                   plt.title('Latent Space Projection-validation data')
                                   plt.savefig(folder+'allz_scatter_encoded_umap_valid-connecting data.png')
                                   
                                   
                                   
encoded_train = encoder.predict(swe_train[0:batch_num*10])[0]
reducer = umap.UMAP(n_neighbors=neighbor, min_dist=distmin)
                                   
embedding = reducer.fit_transform(encoded_train)
f, a = plt.subplots(1, 1, figsize = (9, 6))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=None)
                                   # for para_idx in range(5):
sc = a.scatter(embedding[:, 0], embedding[:, 1], c= temp_indx[0:batch_num*10], s = 20, alpha = 0.5) #
cbar = plt.colorbar(sc)
cbar.set_label('Time-step', rotation=270)
plt.ylabel('Z2')
plt.xlabel('Z1')
plt.title('Latent Space Projection-training data')
plt.savefig(folder+'allz_scatter_encoded_umap_train-connecting dots-beta.png')

