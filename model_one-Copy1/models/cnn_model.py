from .conv_module import ConvModule
from .fc_network import my_flow
from utils.metrics import calculate_accuracy
import pickle
import os
import numpy as np


class ConvMyFlow(my_flow):

    def __init__(self,conv_config,hidden_layers,input_data,label_data,
                 categories_size,activation_function_label_list,
                 train_numder=100,lr=0.001,lamda=0.01):

        self.conv_config=conv_config

        self.conv_module=ConvModule(conv_config,input_data)

        conv_flat=self.conv_module.forward()

        self.train_label=label_data

        super().__init__(hidden_layers,conv_flat,label_data,
                         categories_size,
                         activation_function_label_list,
                         train_numder,lr,lamda)

    def train(self,X_test,Y_test):

        self.loss_list=[]
        self.train_acc_list=[]
        self.test_acc_list=[]

        for epoch in range(self.train_numder):

            conv_flat=self.conv_module.forward()

            self.input_data=conv_flat

            self.forward()

            loss=self.loss()

            train_acc=self.accuracy()

            test_conv=self.conv_module.test_forward(X_test)

            test_pred=self.test_forward(test_conv)

            test_acc=calculate_accuracy(test_pred,Y_test)

            self.backward()

            self.adam_update()

            self.conv_module.adam_update(self.lr,self.t)

            print(epoch,loss,train_acc,test_acc)

        return self.loss_list,self.train_acc_list,self.test_acc_list