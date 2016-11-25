##############################
#Text-Generation using LSTM's
#By Soham Banerjee
##############################

#Import the libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils


data = open("HumanAction.txt").read()
data = data.lower()

#Important Parameters
total_chars = len(data)
set_of_chars = set(data)
no_uniq_chars = len(set_of_chars)
print("Total Chars: %d" % total_chars)
print("Total Unique: %d" % no_uniq_chars)

#Create the hashmaps for coversion from character to number and vice versa
c_to_i = dict((c,i) for i,c in enumerate(set_of_chars))
i_to_c = dict(enumerate(set_of_chars))

#Pre-Processing of Data
length = 100  #Length of the sequence to be generated
dataX = []
dataY = []
#Create the sequences to be used as input
for i in range(0, total_chars - length, 3):
	inp = data[i:i + length]
	oup = data[i + length]
	dataX.append([c_to_i[c] for c in inp])
	dataY.append(c_to_i[oup])

dataX = np.array(dataX,dtype= 'int32')
num_seq = len(dataX) 
#Apply One-Hot Encoding to the labels
trainY = np_utils.to_categorical(dataY).astype('bool')  
trainX = np.zeros((num_seq,length,no_uniq_chars), dtype='bool')
for i, seq in enumerate(dataX):
    for j, c in enumerate(seq):
        trainX[i,j,c] = 1
print(trainX.shape)
print(trainY.shape)

#Define the model
#Two layers of LSTM with 512 units followed by the fully connected softmax layer
model = Sequential()
model.add(LSTM(256,input_shape=(trainX.shape[1],trainX.shape[2]), consume_less = 'gpu'))
model.add(Dropout(0.2))
#model.add(LSTM(512, consume_less = 'gpu'))
#model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1], activation='softmax'))

#Define the training parameters
epochs = 1
optimizer = 'RMSprop'
batch_size = 128

#Compile the model
model.compile(loss = 'categorical_crossentropy',optimizer = optimizer)

#Mega-Epochs
for k in range(3):
    #Train the model
    model.fit(trainX, trainY, nb_epoch = epochs, batch_size = batch_size)

    #Save the model for reproducibility
    name = "saved_model" + str(k) + ".h5"
    model.save(name)
    
    #Output the predicted sequences into text files
    for i in range(5):
        idx = np.random.randint(0, len(dataX)-1)  #Pick out a random index to be used as input for prediction
        seq = dataX[idx]
        file_name = "Output" + str(k) + str(i) + ".txt"
        text_file = open(file_name, 'w')
        text_file.write("%s" % "Input-\n")
        text_file.write("%s" % ''.join([i_to_c[c] for c in seq]))
        text_file.write("%s" % "\nOutput-\n")
        for j in range(1000):
            inp = np.zeros((length,no_uniq_chars),dtype = 'bool')
            for n,c in enumerate(seq):
                inp[n,c] = 1
            pred = model.predict(inp)   #Generate the characted prediction
            int_pred = np.argmax(pred)
            text_file.write("%s" % i_to_c[int_pred]) 
            seq = np.append(seq,int_pred)
            seq = seq[1:]
        text_file.close()
