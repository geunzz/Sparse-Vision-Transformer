import random
import pickle
import torch
import numpy as np
from torch import nn
import torch.optim as optim
import pytorch_model_summary
from ptflops import get_model_complexity_info
from structured_prune import *
from Sparse_ViT import *

EPOCHS = 100
BATCH_SIZE = 64
ARCFACE = True
EMB_FEATURE = 512
NUM_CLASSES = 10575

# #model check : Check the number of parameters
model_check = Sparse_ViT(batch_size=1, in_features=EMB_FEATURE, num_classes=NUM_CLASSES)
model_check.load_state_dict(torch.load('PATH/TO/THE/MODEL_WEIGHTS.pt'))

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model_check.to('cpu')
print(device)

macs, params = get_model_complexity_info(model_check, (3,112,112), as_strings=False, print_per_layer_stat=True)
print('Flops:  ', macs/2)
print('Params: ',params)
print(pytorch_model_summary.summary(model_check.cuda(), torch.zeros(1,3,80,110).to(device)))

# model for train
model = Sparse_ViT(batch_size=BATCH_SIZE, in_features=EMB_FEATURE, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load('PATH/TO/THE/MODEL_WEIGHTS.pt'))

model.train()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)
loss_cal = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

device = 'cpu'
model.to(device)
print(device)

#data import
with open('PATH/TO/THE/TRAIN_DATA.pickle','rb') as f:
    train_data = pickle.load(f)

with open('PATH/TO/THE/VAL_DATA.pickle','rb') as f:
    val_data = pickle.load(f)

random.shuffle(train_data)
random.shuffle(val_data)

data_array = np.array(train_data, dtype=object)[:,0] #image
label = np.array(train_data, dtype=object)[:,1] #class

val_array = np.array(val_data, dtype=object)[:,0] #image
val_label = np.array(val_data, dtype=object)[:,1] #class

loss_string = 100000
loss_string_min = 100000

for epoch in range(EPOCHS):
    #train
    for step in range(int(len(data_array)/BATCH_SIZE)):
        indices = range(step*BATCH_SIZE,step*BATCH_SIZE+BATCH_SIZE)
        batch_image = []
        batch_label = []
        for index in indices:
            sample = data_array[index]
            sample = np.transpose(sample)
            batch_image.append(sample)
            batch_label.append(label[index]) #class

        input = torch.tensor(np.array(batch_image)).float()
        if ARCFACE == True:
            out = model.arcface_forward(input.to(device), torch.LongTensor(batch_label).to(device))
        else:
            out = model.softmax_forward(input.to(device))

        #loss calculation
        loss = loss_cal(out, torch.LongTensor(batch_label).to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('steps:',step,'/',int(len(data_array)/BATCH_SIZE),'epoch:', epoch,'/',EPOCHS,'loss :',round(float(loss.detach()),4))
    
    #validation
    loss_list = []
    for step in range(int(len(val_array)/BATCH_SIZE)):
        indices = range(step*BATCH_SIZE,step*BATCH_SIZE+BATCH_SIZE)
        batch_image = []
        batch_label = []
        for index in indices:
            sample = val_array[index]
            sample = np.transpose(sample)
            batch_image.append(sample)
            batch_label.append(val_label[index]) #class

        input = torch.tensor(np.array(batch_image)).float()
        if ARCFACE == True:
            out, _ = model.forward_train(input.to(device), torch.LongTensor(batch_label).to(device))
        else:
            out = model.forward_train(input.to(device))
        loss = loss_cal(out, torch.LongTensor(batch_label).to(device))
        loss_list.append(round(float(loss.clone().detach()),4))

    if loss_string < loss_string_min:
        loss_string_min = loss_string
    loss_string = round(np.mean(loss_list),4)


    if loss_string < loss_string_min:
        torch.save(model, 'lfw/LFW_ViT_S_try1_'+str(epoch)+'_'+str(loss_string)+'.pt')

    random.shuffle(train_data)

    data_array = np.array(train_data, dtype=object)[:,0] #image
    label = np.array(train_data, dtype=object)[:,1] #class

torch.save(model, 'lfw/LFW_ViT_S_try1_'+str(epoch)+'_'+str(loss_string)+'.pt')






