import numpy as np 

import tensorflow as tf

import tensorflow_datasets as tfds

import random

# (ds_train, ds_test), ds_info = tfds.load(
#     'mnist',
#     split=['train', 'test'],
#     shuffle_files=True,
#     as_supervised=True,
#     with_info=True,
# )

mylist = ['1','2','3','4']
print(mylist)
#random.shuffle(mylist)
#print(mylist)
print(mylist[0:1])
x = 1
z = 3
tl = mylist[x:z]
del mylist[x:z]

print(tl)

print(mylist)

keys = [1,2,3,4]

# mydict = dict.fromkeys(keys, np.array([1,2]))
mydict = { 1: 2, 2: 4}

mydict[1] = 5 + mydict[1]

print(mydict)

for pos in range(10):
    print(pos)

mydict = {'george': 16, 'amber': 19}

print(max(mydict.values()))

print(list(mydict.keys())[list(mydict.values()).index(16)])  # Prints george


