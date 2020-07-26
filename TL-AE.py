# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 17:11:25 2019

@author: Horace
"""

import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Dropout, Activation
import matplotlib.pyplot as plt
from CSVDataModule.LabelDataTable import LabelDataTable
import datetime
import csv

np.random.seed(1358)
LabelDataModule = LabelDataTable()

# In[]

def show_train_history_val(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train history')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

# In[] 讀建模檔案

MachineID = "SN0003"
want = "d3"
day = "0828"

intput_data = './ModelTrain/Z_SN0003_d3_0829_u-615.34_std-950.8.csv'

train_ForgingLabelData = LabelDataModule.load_csv_data(intput_data, direction='r')
test_ForgingLabelData = LabelDataModule.load_csv_data(intput_data, direction='r')
train_ForgingLabelData_col = LabelDataModule.load_csv_data(intput_data, direction='c')
test_ForgingLabelData_col = LabelDataModule.load_csv_data(intput_data, direction='c')

train_ForgingLabelDataKeys = LabelDataModule.get_dict_keys(train_ForgingLabelData)
test_ForgingLabelDataKeys = LabelDataModule.get_dict_keys(test_ForgingLabelData)

TimeTag_train = train_ForgingLabelData_col['TimeTag']
y_train1 = train_ForgingLabelData_col['HardwareShift']
y_train2 = train_ForgingLabelData_col['SoftWareShift']

for i in range(len(train_ForgingLabelDataKeys)):
    if train_ForgingLabelDataKeys[i] is None:
        raise Exception('key array have None value.')
for i in range(len(test_ForgingLabelDataKeys)):
    if test_ForgingLabelDataKeys[i] is None:
        raise Exception('key array have None value.')

x_train_data = []
x_test_data = []
for i in range(int(train_ForgingLabelDataKeys[-1]) + 1):
    x_train_data.append(train_ForgingLabelData[str(i)][3:])
for i in range(int(test_ForgingLabelDataKeys[-1]) + 1):
    x_test_data.append(test_ForgingLabelData[str(i)][3:])

x_train = np.array(x_train_data).astype('float32')
x_test = np.array(x_test_data).astype('float32')
print(x_train.shape)

# In[] 遷移：讀新資料

MachineID1 = "SN0003"
want1 = "d3"
day1 = "0829"

intput_data = './refresh_train/Z_SN0003_d3_0829(1min).csv'

ForgingLabelData_Ytrain = LabelDataModule.load_csv_data(intput_data, direction='r')
ForgingLabelData_Ytest = LabelDataModule.load_csv_data(intput_data, direction='r')
ForgingLabelData_Ytrain_col = LabelDataModule.load_csv_data(intput_data, direction='c')
ForgingLabelData_Ytest_col = LabelDataModule.load_csv_data(intput_data, direction='c')

ForgingLabelDataKeys_Ytrain = LabelDataModule.get_dict_keys(ForgingLabelData_Ytrain)
ForgingLabelDataKeys_Ytest = LabelDataModule.get_dict_keys(ForgingLabelData_Ytest)

TimeTag_train_Y = ForgingLabelData_Ytrain_col['TimeTag']
train1_Y = ForgingLabelData_Ytrain_col['HardwareShift']
train2_Y = ForgingLabelData_Ytrain_col['SoftWareShift']

for i in range(len(ForgingLabelDataKeys_Ytrain)):
    if ForgingLabelDataKeys_Ytrain[i] is None:
        raise Exception('key array have None value.')
for i in range(len(ForgingLabelDataKeys_Ytest)):
    if ForgingLabelDataKeys_Ytest[i] is None:
        raise Exception('key array have None value.')

Y_train_data = []
Y_test_data = []
for i in range(int(ForgingLabelDataKeys_Ytrain[-1]) + 1):
    Y_train_data.append(ForgingLabelData_Ytrain[str(i)][3:])
for i in range(int(ForgingLabelDataKeys_Ytest[-1]) + 1):
    Y_test_data.append(ForgingLabelData_Ytest[str(i)][3:])

Y_train = np.array(Y_train_data).astype('float32')
Y_test = np.array(Y_test_data).astype('float32')
print(Y_train.shape)

# In[] AE

encoding_dim = 1 # 維度設定
d1,d2,d3,d4 = 700,500,200,32


# In[] AE結構
#
input_raw = Input(shape=(1000,))

# encoded layers
encoded1 = Dense(d1, activation='relu', kernel_initializer='RandomNormal')(input_raw)
encoded2 = Dense(d2, activation='relu', kernel_initializer='RandomNormal')(encoded1)
encoded3 = Dense(d3, activation='relu', kernel_initializer='RandomNormal')(encoded2)
encoded4 = Dense(d4, activation='relu', kernel_initializer='RandomNormal')(encoded3)
encoder_output = Dense(encoding_dim)(encoded4)

#
## decoder layers
decoded4 = Dense(d4, activation='relu', kernel_initializer='RandomNormal')(encoder_output)
decoded3 = Dense(d3, activation='relu', kernel_initializer='RandomNormal')(decoded4)
decoded2 = Dense(d2, activation='relu', kernel_initializer='RandomNormal')(decoded3)
decoded1 = Dense(d1, activation='relu', kernel_initializer='RandomNormal')(decoded2)
#
decoded = Dense(1000, activation='sigmoid')(decoded1)  #relu,sigmoid


#
# In[] 建模

autoencoder = Model(input=input_raw, output=decoded)
encoder = Model(input=input_raw, output=encoder_output)
autoencoder.compile(optimizer='Adam', loss='mae', metrics=['accuracy']) #RMSprop,Adam
print(autoencoder.summary())
autoencoder
StartTime = datetime.datetime.now()

# training
epoch= 200
size = int(len(x_train)/10)

m_train_history = autoencoder.fit(x_train, x_train, nb_epoch=epoch, batch_size= size ,shuffle=True,
                                  validation_data=(x_test, x_test), verbose=2)

autoencoder.save("./Results/%s_%s_%s(AE).h5"%(MachineID,want,day))
encoder.save("./Results/%s_%s_%s(en).h5"%(MachineID,want,day))

EndTime = datetime.datetime.now()
print(StartTime, '; ', EndTime)
print(EndTime - StartTime)

show_train_history_val(m_train_history, 'acc', 'val_acc')
show_train_history_val(m_train_history, 'loss', 'val_loss')

train_result = autoencoder.evaluate(x_train, x_train, batch_size=size)
test_result = autoencoder.evaluate(x_test, x_test, batch_size=size)
train_predict = autoencoder.predict(x_train, batch_size=size)
test_predict = autoencoder.predict(x_test, batch_size=size)
#print(train_predict[0, :])
#print('Train Acc:', train_result[1])
#print('Test Acc:', test_result[1])
encoded_data = encoder.predict(x_train)

# In[] 凍結部分模型參數
for layer in autoencoder.layers:
    layer.trainable = False # 先凍結所有層

# model.trainable = False 
autoencoder.layers[4].trainable = True
autoencoder.layers[5].trainable = True
autoencoder.layers[6].trainable = True

print('可訓練層:')
for x in autoencoder.trainable_weights:
    print(x.name)

print('\n凍結層:')
for x in autoencoder.non_trainable_weights:
    print(x.name)

# In[] 遷移：訓練資料

size = int(len(Y_train)/10)
StartTime = datetime.datetime.now()
m_train_history1 = autoencoder.fit(Y_train, Y_train, nb_epoch=epoch, batch_size= size ,shuffle=True,
                                  validation_data=(Y_test, Y_test), verbose=2)

autoencoder.save("./Results/%s_%s_%s(AE).h5"%(MachineID1,want1,day1))
encoder.save("./Results/%s_%s_%s(en).h5"%(MachineID1,want1,day1))

EndTime = datetime.datetime.now()
print(EndTime - StartTime)

show_train_history_val(m_train_history1, 'acc', 'val_acc')
show_train_history_val(m_train_history1, 'loss', 'val_loss')

train1_result_Y = autoencoder.evaluate(Y_train, Y_train, batch_size=size)
test1_result_Y = autoencoder.evaluate(Y_test, Y_test, batch_size=size)
train1_predict_Y = autoencoder.predict(Y_train, batch_size=size)
test1_predict_Y = autoencoder.predict(Y_test, batch_size=size)

encoded_data1 = encoder.predict(Y_train)

# In[] 存檔

#Decoded Data
with open("./Results/sigmoid+Adam_%s_%s(AE)%s.csv"%(MachineID1,want1,day1), 'w', newline='') as csv_file:
    m_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    m_writer.writerow(["TimeTag", "HardwareShift", "SoftWareShift", "EncodedData"])
    for i in range(len(train1_Y)):
        tempWriteString = []
        tempWriteString.append(TimeTag_train_Y[i])
        tempWriteString.append(train1_Y[i]) 
        tempWriteString.append(train2_Y[i])
        for index in range(len(train1_predict_Y[i, :])):
            tempWriteString.append(str(train1_predict_Y[i, index]))
        m_writer.writerow(tempWriteString)

# Encoded Data 1D

with open("./Results/sigmoid+Adam_%s_%s(1D)%s.csv"%(MachineID1,want1,day1), 'w', newline='') as csv_file:
    m_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    m_writer.writerow(["TimeTag", "HardwareShift", "SoftWareShift", "loc1"])
    for i in range(len(encoded_data1)):
        m_writer.writerow([TimeTag_train_Y[i], train1_Y[i], train2_Y[i], encoded_data1[i, 0]])
#    tempWriteString.append(Label[i])
        
with open("./Results/sigmoid+Adam_%s_%s(AE)%s.csv"%(MachineID,want,day), 'w', newline='') as csv_file:
    m_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    m_writer.writerow(["TimeTag", "HardwareShift", "SoftWareShift", "EncodedData"])
    for i in range(len(y_train1)):
        tempWriteString = []
        tempWriteString.append(TimeTag_train[i])
        tempWriteString.append(y_train1[i]) 
        tempWriteString.append(y_train2[i])
        for index in range(len(train_predict[i, :])):
            tempWriteString.append(str(train_predict[i, index]))
        m_writer.writerow(tempWriteString)

# Encoded Data 1D

with open("./Results/sigmoid+Adam_%s_%s(1D)%s.csv"%(MachineID,want,day), 'w', newline='') as csv_file:
    m_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    m_writer.writerow(["TimeTag", "HardwareShift", "SoftWareShift", "loc1"])
    for i in range(len(encoded_data)):
        m_writer.writerow([TimeTag_train[i], y_train1[i], y_train2[i], encoded_data[i, 0]])
