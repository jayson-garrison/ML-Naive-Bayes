# driver to control the ML instances

import numpy as np 
import csv
import random

# open the data file
aggregate_data = []
data_file = open('mnist/train.csv', newline='')
reader = csv.reader(data_file)

# 42,000 images now in a list stores as row vectors
reader.__next__()
for row in reader:
    array_point = np.asarray(row)
    aggregate_data.append(array_point)

print(aggregate_data[0])
print(len(aggregate_data))

# shuffle the order of the list
random.shuffle(aggregate_data)
#print(aggregate_data[0:1])

# partition the data into 5 sectors for 5-fold cross validation
partition_index = int( len(aggregate_data) / 5 )
print(partition_index)
s = 0
fold = []
for i in range(5): #0-4
    tr = aggregate_data.copy()
    n = s + partition_index - 1
    te = tr[s:n]
    del tr[s:s + partition_index]

    fold.append( (tr,te) )

    s += partition_index
    print(i)

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
