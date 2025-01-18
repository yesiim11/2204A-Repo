import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
from keras.callbacks import EarlyStopping
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras import regularizers

from random import sample
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os





#GPU
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
#GPU




#GEN DEFINE
def get_filepaths(dir:str):
    files=[]
    for filename in os.listdir(dir):
        files.append(os.path.join(dir, filename))
    return files


def generator(filepaths:list, batch_size:int):

    while 1:
        samples = sample(filepaths, batch_size)
        return_x_arr = []
        return_y_arr = []
        for smpl in samples:
            arrs = np.load(smpl)
            return_x_arr.append(np.expand_dims(arrs[0], axis = -1))
            return_y_arr.append(np.expand_dims(arrs[1] - arrs[0], axis = -1))
        
        return_x_arr = np.array(return_x_arr)
        return_y_arr = np.array(return_y_arr)
    
        yield return_x_arr, return_y_arr
#GEN DEFINE
    



###MODEL###

#Evrişim bloğu
def conv_block(x,num_filters):
    x=L.Conv2D(num_filters,3, padding="same", activation = L.PReLU(), kernel_regularizer=regularizers.l2(1e-7))(x)
    x=L.Conv2D(num_filters,3, padding="same", activation = L.PReLU(), kernel_regularizer=regularizers.l2(1e-7))(x)

    return x


#Kodlayıcı Blok
def encoder_block(x, num_filters):
    X=conv_block(x, num_filters)
    p=L.MaxPool2D((2,2))(X)
    return X, p


#Dikkat Kapısı
def attention_gate(g, s, num_filters):
    Wg=L.Conv2D(num_filters, 1, padding="same", activation = L.PReLU(), kernel_regularizer=regularizers.l2(1e-7))(s)
    Wg=L.BatchNormalization()(Wg)

    Ws=L.Conv2D(num_filters, 1, padding="same", activation = L.PReLU(), kernel_regularizer=regularizers.l2(1e-7))(s)
    Ws= L.BatchNormalization()(Ws)

    out=L.Conv2D(num_filters, 1, padding="same")(Wg + Ws)
    out=L.Activation("tanh")(out)

    return out*s


#Kod Çözücü Blok
def decoder_block(x,s,num_filters):
    x=L.UpSampling2D(interpolation="bilinear")(x)
    s= attention_gate(x,s,num_filters)
    x = L.Concatenate()([x,s])
    x=conv_block(x,num_filters)
    return x



def attention_unet(input_shape):
    
    inputs=L.Input(input_shape)
    

    s1, p1 = encoder_block(inputs,32)
    s2, p2 = encoder_block(p1, 64)
    s3, p3= encoder_block(p2,128)
    s4, p4= encoder_block(p3,256)
    s5, p5= encoder_block(p4,512)
    s6, p6= encoder_block(p5,1024)

    
    b1=conv_block(p6,2048)

    d1=decoder_block(b1,s6, 1024)
    d2=decoder_block(d1,s5, 512)
    d3=decoder_block(d2,s4, 256)
    d4=decoder_block(d3,s3, 128)
    d5=decoder_block(d4,s2, 64)
    d6=decoder_block(d5,s1, 32)
    
    outputs=L.Conv2D(1,1,padding="same", activation="tanh")(d6)

    model=Model(inputs, outputs, name="Attention-Unet")
    return model


input_shape=(512,512,1)
model= attention_unet(input_shape)
model.summary()
#MODEL




#GEN
tr_filepaths = get_filepaths("E:\\YAD\\YAD_İşlenmiş_Düzgün\\")
val_filepaths = get_filepaths("E:\\YAD\\YAD_İşlenmiş_Düzgün\\Val\\")

tr_gen = generator(tr_filepaths, 4)
val_gen = generator(val_filepaths,4)
#GEN


#KAYIP FONKSİYONU
def loss(y_true, y_pred):
    error = tf.abs(y_true - y_pred)
    weights = tf.abs(y_true)
    weighted_loss = weights*error
    return tf.reduce_mean(weighted_loss)
#KAYIP FONKSİYONU

print("*")

#EĞİTİM
adam = tf.keras.optimizers.Adam(learning_rate = 0.001)
es = EarlyStopping(mode="min", monitor="val_loss", patience=7, restore_best_weights=True)

model.compile(optimizer=adam, loss = loss, metrics=[loss, "mse", "mae"])

history = model.fit(tr_gen, epochs = 30, steps_per_epoch = 100, verbose="auto", batch_size = 4, 
                    callbacks = [es], validation_data = val_gen, validation_steps = 32)

history_df = pd.DataFrame(history.history)
history_df[["val_mae", "mae"]].plot()
print(history_df)


model.save("C:\\Users\\kullanıcı\\Desktop\\YAD_04A\\Model5.keras")
model = tf.keras.models.load_model("C:\\Users\\kullanıcı\\Desktop\\YAD_04A\\Model.keras", custom_objects=(loss))

aaa = next(tr_gen)
pred = model.predict(aaa[0][0].reshape(1,512,512,1))

res = aaa[0][0] + pred[0]

print(aaa[0].shape)
print(pred.shape)


plt.imshow(aaa[0][0] + aaa[1][0], vmin = 0.0, vmax = 0.5, cmap = "gray")
plt.title("y")
plt.imshow(aaa[0][0], vmin = 0.0, vmax = 0.5, cmap = "gray")
plt.title("x")
plt.imshow(res, vmin = 0, vmax = 0.5, cmap = "gray")
plt.title("pred")





plt.imshow(aaa[1][0]*3, vmin = -0.24, vmax = 0.9, cmap = "gray")
plt.imshow(res, vmin = 0.0, vmax = 0.1, cmap = "gray")
plt.imshow(res, vmin = 0.0, vmax = 0.1, cmap = "gray")

plt.imshow(pred[0]*13.5*3, vmin = -0.25,vmax =0.63, cmap = "gray")
plt.imshow(aaa[1][0]*3, vmin = -0.25, vmax = 0.63, cmap = "gray")

print(np.min(pred[0]))

print(len(tr_filepaths))

model.save("C:\\Users\\kullanıcı\\Desktop\\YAD_04A\\Model6.keras")
