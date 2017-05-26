'''
Created on 26 May 2017

@author: xuepeng
'''
from network import Network
import mnist_loader

training_data, validation_data, test_data =  mnist_loader.load_data_wrapper()

sizes = [784,30,10]
net = Network(sizes)
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
# 
# a = [1,0]
# o = net.feedforward(a)
# print o

