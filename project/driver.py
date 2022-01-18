# driver to control the ML instances

from re import X
from models.model import GenericModel as g_model
from models.bernoulli_model import BernoulliModel as b_model
from booleanize import Booleanize as bool

import numpy as np 
import csv
import random
import time

# open the data file
labels = [0,1,2,3,4,5,6,7,8,9]
aggregate_data = []
data_file = open('project/mnist/train.csv', newline='')
reader = csv.reader(data_file)

# 42,000 images now in a list stores as row vectors
reader.__next__()
start_time = time.time()
for row in reader:
    # convert to ints
    array_ints = list( map(int, row) )

    array_point = np.array(array_ints)
    aggregate_data.append(array_point)
print("read data time: ", time.time() - start_time)

# booleanize the data
boolean_param = 64
start_time = time.time()
bool.booleanize(aggregate_data, boolean_param)
print("booleanize time: ", time.time() - start_time)
print('boolean param: ', boolean_param)

#print(aggregate_data[3])
print('length of aggregate data: ', len(aggregate_data))
#print(np.add(aggregate_data[3], aggregate_data[3]))

# shuffle the order of the list
random.shuffle(aggregate_data)
#print(aggregate_data[0:1])

# 0-8399, 8400-
# testing
# reader.__next__()
# data_point = reader.__next__()
# array_data_point = np.asarray(data_point)
# # show
# print(array_data_point)
# # get position
# print(array_data_point[0])

# array_data_point
# # print(reader.__next__()[0])

# partition the data into 5 sectors for 5-fold cross validation
partition_index = int( len(aggregate_data) / 5 )
print('pdex: ', partition_index) 
s = 0
fold = []
for i in range(5): #0-4
    tr = aggregate_data.copy()
    n = s + partition_index # was -1
    te = tr[s:n]
    del tr[s:s + partition_index]

    fold.append( (tr,te) )

    s += partition_index

#print(type(fold[0][0][0])) #np.ndarray
#print(len(fold[0][0][0])) #785
# fold is now a list of couples (tr, te) where tr and te are lists of 33600 and 8400 arrays respectively
# each of len 785 where array[0] is the label and all other is img data.

# create instance of model

results = open('image_results.txt', "w")
test_accuracies = []
train_accuracies = []
trial_num = 0

for laplace_k in range(5):
    results.write(f'Laplace Smoothing k = {laplace_k}\n')
    for tr_te_partition in fold:
        b_model1 = b_model(tr_te_partition, labels, 785)

        # train the current partition with the current k
        start_time = time.time()
        b_model1.train(laplace_k)
        results.write(f'train time: {time.time() - start_time}\n')

        # classify on training set and test set and evaluate accuracy
        start_time = time.time()
        coupled_accuracies = b_model1.classify()
        results.write(f'classify aggregate time: {time.time() - start_time}\n')

        train_accuracies.append(coupled_accuracies[0])
        test_accuracies.append(coupled_accuracies[1])
        
        b_model1.stats()

    # writing
    results.write('\n')
    results.write(f'avg training accuracy for Laplace Smoothing k = {laplace_k}: {sum(train_accuracies) / len(train_accuracies)}\n')
    results.write(f'avg test accuracy for Laplace Smoothing k = {laplace_k}: {sum(test_accuracies) / len(test_accuracies)}\n')
    results.write('\n')
    results.write('--------------------------New Trial Session--------------------------\n')
    
    print('train accuracies for laplace k = ', laplace_k, ': ', train_accuracies)
    print('test accuracies for laplace k = ', laplace_k, ': ', test_accuracies)
    
    print('avg train accuracy: ', sum(train_accuracies) / len(train_accuracies))
    print('avg test accuracy: ', sum(test_accuracies) / len(test_accuracies))

    # reset
    test_accuracies = []
    train_accuracies = []
    
results.close()


# b_model1 = b_model(fold[0], labels, 785 )

# start_time = time.time()
# b_model1.train(0)
# print("train time: ", time.time() - start_time)

# start_time = time.time()
# accur = b_model1.classify()
# print("classify time: ", time.time() - start_time)

# b_model1.stats()


