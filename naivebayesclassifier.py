__author__ = 'Xiaofei Zhang'

import csv
import sys

train = sys.argv[1]
test = sys.argv[2]

yelp2 = []
training_set = []
test_set = []

# import training-set data
# with open('train-set.csv', 'rb') as file1:
with open(train, 'rb') as file1:
    data1 = csv.reader(file1)
    training_set = list(data1)

# import test-set data
# with open('test-set.csv', 'rb') as file2:
with open(test, 'rb') as file2:
    data2 = csv.reader(file2)
    test_set = list(data2)


# convert training_set to dictionary
train_dict = {}
for i in range(len(training_set[0])):
    train_dict[training_set[0][i]] = []

# len(training_set) = 10000
for i in range(1, len(training_set)):
    for j in range(len(training_set[i])):
        train_dict[training_set[0][j]].append(training_set[i][j])


# convert testing_set to dictionary
test_dict = {}
for i in range(len(test_set[0])):
    test_dict[training_set[0][i]] = []

# len(training_set) = 10000
for i in range(1, len(test_set)):
    for j in range(len(test_set[i])):
        test_dict[test_set[0][j]].append(test_set[i][j])


n_yes = 0
n_no = 0

for j in range(len(train_dict['goodForGroups'])):
    if train_dict['goodForGroups'][j] == '1':
        n_yes += 1
    elif train_dict['goodForGroups'][j] == '0':
        n_no += 1


prYes = 1.0 * n_yes / (n_yes + n_no)
prNo = 1.0 - prYes


CPD_dict = {}
CPD_key = []
list1 = ['goodForGroups', 'latitude', 'longitude', 'reviewCount', 'checkins']
for x in train_dict.keys():
    if x not in list1:
        CPD_key.append(x)

# add CPD keys to the CPD dictionary
# initialize the dictionary
# dictionary of dictionary of list with count of [yes, no]
for k in range(len(CPD_key)):
    CPD_dict[CPD_key[k]] = {}

# calculate each CPD
for x in CPD_dict.keys():
    count_yes = n_yes
    count_no = n_no
    # different level in each factors(key), with no repetition
    for level in set(train_dict[x]):
        CPD_dict[x][level] = [0, 0]
        # count the frequency of each level given YES / NO
        for i in range(len(train_dict[x])):
            if train_dict['goodForGroups'][i] == '1' and train_dict[x][i] == level:
                CPD_dict[x][level][0] += 1
            elif train_dict['goodForGroups'][i] == '0' and train_dict[x][i] == level:
                CPD_dict[x][level][1] += 1
    # smoothing
    for level in set(train_dict[x]):
        if CPD_dict[x][level][0] == 0:
            # adjust denominator
            count_yes += len(set(train_dict[x]))
            # adjust numerator
            for sublevel in set(train_dict[x]):
                CPD_dict[x][sublevel][0] += 1
        if CPD_dict[x][level][1] == 0:
            count_no += 1*len(set(train_dict[x]))
            for sublevel in set(train_dict[x]):
                CPD_dict[x][sublevel][1] += 1

    for level in set(train_dict[x]):
        CPD_dict[x][level][0] = 1.0*CPD_dict[x][level][0]/count_yes
        CPD_dict[x][level][1] = 1.0*CPD_dict[x][level][1]/count_no


pred_test = []
prob = []
for i in range(len(test_dict[test_dict.keys()[0]])):
    prob_Y = prYes
    prob_N = prNo
    for key in CPD_dict.keys():
        # print "test_dict[key][i]:", test_dict[key][i]
        if test_dict[key][i] in set(train_dict[key]):
            prob_Y = prob_Y * CPD_dict[key][test_dict[key][i]][0]
            prob_N = prob_N * CPD_dict[key][test_dict[key][i]][1]
        # print prob_Y, prob_N
    if prob_Y >= prob_N:
        pred_test.append('1')
    elif prob_Y < prob_N:
        pred_test.append('0')
    prob.append((prob_Y)/((prob_Y+prob_N))) # p_i

# zero-one loss
zero_one_loss = 0
for i in range(len(pred_test)):
    if pred_test[i] != test_dict['goodForGroups'][i]:
        zero_one_loss += 1
zero_one_loss = 1.0*zero_one_loss/len(pred_test)

# Squared loss
squared_loss = 0
for i in range(len(pred_test)):
    if pred_test[i] == test_dict['goodForGroups'][i]:
        if pred_test[i] == '1':
            squared_loss += (1-prob[i])**2
        elif pred_test[i] == '0':
            squared_loss += prob[i]**2
    elif pred_test[i] != test_dict['goodForGroups'][i]:
        if pred_test[i] == '1':
            squared_loss += prob[i]**2
        elif pred_test[i] == '0':
            squared_loss += (1-prob[i])**2
squared_loss = 1.0 * squared_loss/len(prob)

print 'ZERO-ONE LOSS =', zero_one_loss
print 'SQUARED LOSS =', squared_loss

