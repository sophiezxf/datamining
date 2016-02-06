from numpy import *
import itertools
import csv
import sys

raw_file = sys.argv[1]
minSup = float(sys.argv[2])
minConf = float(sys.argv[3])

support_dic = {}

# generate C_1
def createC1(dataSet):
    C1 = set([])
    for item in dataSet:
        C1 = C1.union(set(item))
    return [frozenset([i]) for i in C1]

# the function to generate L_k
def getLk(dataset, Ck, minSup):
    global support_dic
    Lk = {}
    # Counting the frequency of C_k
    for item in dataset:
        for Ci in Ck:
            if Ci.issubset(item):
                if not Ci in Lk:
                    Lk[Ci] = 1
                else:
                    Lk[Ci] += 1
    #generate L_k
    Lk_return = []
    for Li in Lk:
        support_Li = Lk[Li] / float(len(dataset))
        if support_Li >= minSup:
            Lk_return.append(Li)
            support_dic[Li] = support_Li
    return Lk_return

# generate C_k+1
def genLk1(Lk):
    Ck1 = []
    for i in range(len(Lk) - 1):
        for j in range(i + 1, len(Lk)):
            if sorted(list(Lk[i]))[0:-1] == sorted(list(Lk[j]))[0:-1]:
                Ck1.append(Lk[i] | Lk[j])
    return Ck1

# Association Rules function
def genRule(freqSet, minConf):
    res = []
    for i in range(1, len(freqSet)):
        count = 0
        for Item in freqSet[i]:
            for element in itertools.combinations(list(Item), 1):
                if support_dic[Item] / float(support_dic[Item - frozenset(element)]) >= minConf:
                    count = count + 1
        res.append(count)
    return res

f = open(raw_file, 'rb')
reader = csv.reader(f)
header = reader.next()
train_set = list(reader)
clear_data = []
for i in range(len(train_set)):
    temp = []
    for j in range(len(train_set[i])):
        if train_set[i][j] != '':
            temp.append(header[j]+":"+train_set[i][j])
    clear_data.append(temp)


# main function
if __name__ == '__main__':
    dataset = clear_data
    result_list = []
    # calculating frequent itemsets
    Ck = createC1(dataset)
    while True:
        Lk = getLk(dataset, Ck, minSup)
        if not Lk:
            break
        result_list.append(Lk)
        Ck = genLk1(Lk)
        if not Ck:
            break
    # output frequent items and its frequency
    for i in range(1,len(result_list)):
        print "FREQUENT-ITEMS",i+1,len(result_list[i])

    # Association Rules
    res = genRule(result_list, minConf)
    for i in range(len(res)):
        print "ASSOCIATION-RULES", i+2, res[i]


