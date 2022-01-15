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


