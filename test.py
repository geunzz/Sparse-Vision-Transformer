import pickle
import random
import torch
import numpy as np
from Sparse_ViT import Sparse_ViT

BATCH_SIZE = 16

#model property input
NUM_CLASSES = 10575
EMB_FEATURE = 512

IDENTIFICATION = True #for identification test, if False then verification test
CLOSED_SET = True # Closed-set test
OPEN_SET = not CLOSED_SET #Open-set test

model = Sparse_ViT(batch_size=BATCH_SIZE, num_classes=NUM_CLASSES, in_features=EMB_FEATURE)
model.load_state_dict(torch.load('PATH/TO/THE/MODEL_WEIGHTS.pt'))

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model.to(device)
model.eval()

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def metric(x1, x2):
    return np.linalg.norm(x1-x2)

def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0 
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return best_acc, best_th

if IDENTIFICATION:
    if CLOSED_SET:
        with open('PATH/TO/THE/TEST_DATASET.pickle', 'rb') as f:
            test_data = pickle.load(f)

        random.shuffle(test_data)
        correct = 0
        total = 0.000001

        data_array = np.array(test_data, dtype=object)[:,0] #image
        label = np.array(test_data, dtype=object)[:,1] #class

        feature_gallery = []

        for step in range(int(len(data_array)/BATCH_SIZE)):
            indices = range(step*BATCH_SIZE,step*BATCH_SIZE+BATCH_SIZE)
            batch_gallery = []
            target_gallery = []
            for index in indices:
                sample = data_array[index]
                sample = np.transpose(sample)
                batch_gallery.append(sample)
                target_gallery.append(label[index]) #class
            input = torch.tensor(np.array(batch_gallery)).clone().detach().float()

            _, out = model.arcface_forward(input.to(device), torch.LongTensor(target_gallery).to(device))

            pred_labels = torch.argmax(torch.softmax(out, -1),-1)
            for pred_label, target_label in zip(pred_labels, target_gallery):
                if pred_label == target_label:
                    correct = correct + 1
                    total = total + 1
                else:
                    total = total + 1

        print('======================final=======================')
        print('total accuracy:', round(correct/total,4))
        
    elif OPEN_SET:
        with open('PATH/TO/THE/TEST_DATASET.pickle', 'rb') as f:
            test_data = pickle.load(f)

        random.shuffle(test_data)
        correct = 0
        total = 0.000001

        data_array = np.array(test_data, dtype=object)[:,0] #image
        label = np.array(test_data, dtype=object)[:,1] #class

        feature_gallery = []

        print('generating gallery..')
        #prediction feature gallery
        for step in range(int(len(data_array)/BATCH_SIZE)):
            indices = range(step*BATCH_SIZE,step*BATCH_SIZE+BATCH_SIZE)
            batch_gallery = []
            target_gallery = []
            for index in indices:
                sample = data_array[index]
                sample = np.transpose(sample)
                batch_gallery.append(sample)
                target_gallery.append(label[index]) #class
            input = torch.tensor(np.array(batch_gallery)).clone().detach().float()

            out = model.forward(input.to(device))
            features = out #class
            for i in range(len(features)):
                feature_gallery.append([features[i].detach().cpu().numpy(), target_gallery[i]])

        with open('feature.pickle', 'wb') as f:
            pickle.dump(feature_gallery, f)

        #trimming train data
        rep_array = []
        for i in range(len(set(np.array(feature_gallery, dtype=object)[:,1]))):
            indices = np.where(np.array(feature_gallery, dtype=object)[:,1]==i)
            summation = 0
            if len(indices) != 0:
                for index in indices[-1]:
                    summation = feature_gallery[index][0] + summation
                representation = summation/len(indices[-1])
            else:
                break

            rep_array.append([representation, i])

        #class prediction 
        pred_features = np.array(feature_gallery)[:,0]
        target_labels = np.array(feature_gallery)[:,1]

        for (pred_feature, target_label) in zip(pred_features, target_labels):
            similarity_old = 0
            for (class_num, rep_feature) in zip(range(len(rep_array)), np.array(rep_array)[:,0]):
                similarity = cosin_metric(rep_feature, pred_feature)
                if similarity > similarity_old:
                    min_similarity = similarity_old = similarity
                    pred_class = class_num
            
            if target_label == pred_class:
                correct = correct + 1
                total = total + 1
            else:
                total = total + 1

        print('======================final=======================')
        print('total accuracy:', round(correct/total,4))

#open set verification
else:
    with open('PATH/TO/THE/TEST_DATASET.pickle', 'rb') as f:
        test_data = pickle.load(f)

    correct = 0
    total = 0.000001

    data_array1 = np.array(test_data, dtype=object)[:,0] #image
    label1 = np.array(test_data, dtype=object)[:,1] #class

    data_array2 = np.array(test_data, dtype=object)[:,2] #image
    label2 = np.array(test_data, dtype=object)[:,3] #class

    ans_list = []

    for step in range(int(len(data_array1)/BATCH_SIZE)):
        indices = range(step*BATCH_SIZE,step*BATCH_SIZE+BATCH_SIZE)
        batch_gallery1 = []
        target_gallery1 = []
        batch_gallery2 = []
        target_gallery2 = []
        for index in indices:
            sample1 = data_array1[index]
            sample1 = np.transpose(sample1)
            batch_gallery1.append(sample1)
            target_gallery1.append(label1[index])

            sample2 = data_array2[index]
            sample2 = np.transpose(sample2)
            batch_gallery2.append(sample2)
            target_gallery2.append(label2[index])
            
        input1 = torch.tensor(np.array(batch_gallery1)).clone().detach().float()
        input2 = torch.tensor(np.array(batch_gallery2)).clone().detach().float()

        out1 = model.forward(input1.to(device))
        out2 = model.forward(input2.to(device))

        for (feature1, feature2, lab1, lab2) in zip(out1, out2, target_gallery1, target_gallery2):
            similarity = cosin_metric(feature1.cpu().detach().numpy(), feature2.cpu().detach().numpy())
            ans_list.append([similarity, lab1 == lab2])

    best_acc, best_th = cal_accuracy(np.array(ans_list)[:,0], np.array(ans_list)[:,1])

    print('======================final=======================')
    print('total accuracy:', best_acc, 'best threshold:', best_th)