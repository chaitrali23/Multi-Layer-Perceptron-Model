#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 23:13:55 2019

@author: chaitralikshirsagar
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import pylab

#PART A
DATA_FNAME = '/Users/chaitralikshirsagar/Desktop/USC_Fall2018/EE599/HW3/mnist_traindata.hdf5'
f=h5py.File(DATA_FNAME,'r')

print("Keys: %s" %f.keys())

xdata=list(f.keys())[0]
xdata = list(f[xdata])
xdata =np.asarray(xdata)

ydata=list(f.keys())[1]
ydata = list(f[ydata])
ydata =np.asarray(ydata)

train_x=xdata[0:50000]
vald_x=xdata[50000:]

train_y=ydata[0:50000]
vald_y=ydata[50000:]


def init_parameters(layer_params):
    parameters={}  
    # Initializing weight and biases for each layer and storing in dictionary
    for i in range(1,len(layer_params)):
        #parameters['W' + str(i)] = np.random.randn(layer_params[i], layer_params[i-1])*np.sqrt(1/(layer_params[i]+layer_params[i-1]))
        parameters['W' + str(i)]=np.random.normal(0,2/layer_params[i],(layer_params[i],layer_params[i-1]))*0.01
        parameters['b' + str(i)] = np.zeros((layer_params[i], 1))
        
    return parameters
 
#defining activation functions and their derivatives
def ReLU(a):
    
    S1=np.maximum(0,a)
    return S1

def dRelu(dZ):
    
    dZ[dZ<=0]=0
    dZ[dZ>0]=1
    
    return dZ

def tanh(b1):
    
    S2=np.tanh(b1)
    return S2

def dtanh(db):
    
    tanhder=1-np.square(tanh(db))
    return tanhder

def softmax(c):
    
    S3=(np.exp(c))/np.sum(np.exp(c),axis=0)
    return S3

def linear_forward(X,W,b):
    
    Z=np.dot(W,X)
    Z=Z+b
    return Z


def linear_act_fwd(Aprev,W,b,activation):
    
    if(activation=="relu"):
        Z=linear_forward(Aprev,W,b)
        A=ReLU(Z)
        
    if(activation=="tanh"):
        Z=linear_forward(Aprev,W,b)
        A=tanh(Z)
        
    if(activation=="softmax"):
        Z=linear_forward(Aprev,W,b)
        A=softmax(Z)
    
    return A , Z


def fwd_prop(train_x,parameters):
    
    A={}
    Z={}
    
    A["A0"]=train_x.T
    l=len(parameters)//2    #3
    
    for i in range (1,l):
        Aprev=A['A'+str(i-1)]
        A['A'+str(i)],Z['Z'+str(i)]=linear_act_fwd(Aprev,parameters['W'+str(i)],parameters['b'+str(i)],activation='tanh')
    
    A['A'+str(l)],Z['Z'+str(l)]=linear_act_fwd(A['A'+str(l-1)],parameters['W'+str(l)],parameters['b'+str(l)],activation='softmax')
    
    return A, Z

def CrossEntropy(A, train_y):
    
    m=train_y.shape[0]
    error = np.sum(np.multiply(train_y,np.log(A['A'+str(len(layer_params)-1)]).T),axis=0)
    sum1 = np.sum(error)
    cost = (-1/m)*sum1

    return cost

def bkd_prop(A,train_y,parameters,Z,alpha):
    grad={}
    L=len(layer_params)    #4
    m=train_y.shape[0]
    train_y=train_y.T           #make size of train_y and AL same

    dAL1 = A['A'+str(L-1)]-train_y
    grad["dA"+str(L-2)]=np.dot((parameters['W'+str(L-1)]).T,dAL1)
    grad["dW"+ str(L-1)]=(1/m)*np.dot(dAL1,A["A"+str(L-2)].T)
    grad["db"+str(L-1)]=(1/m)*np.sum(dAL1,axis=1)
    
    for i in reversed(range(L-2)):
        
        dZ=np.multiply(grad["dA"+str(i+1)],dtanh(Z["Z"+str(i+1)]))
        grad["dA"+str(i)]=np.dot(parameters["W"+str(i+1)].T,dZ)
        grad["dW"+ str(i+1)]=(1/m)*np.dot(dZ,A["A"+str(i)].T)
        grad["db"+str(i+1)]=(1/m)*np.sum(dZ,axis=1)
    
    for i in range(L-1):
        parameters["W"+str(i+1)]=parameters["W"+str(i+1)]-alpha*grad["dW"+str(i+1)]
        parameters["b"+str(i+1)]=parameters["b"+str(i+1)]- alpha*np.reshape(grad['db'+str(i+1)],(layer_params[i+1],1))

        
    return grad,parameters

layer_params=[784,500,10]

def Llayermodel(train_x,train_y,vald_x,vald_y,layer_params,alpha=0.25,epochs=50,print_cost=True):
    
    costs=[]
    costs1=[]
    
    cc=[]
    cc_train=[]
    #train_acc=[]
    parameters=init_parameters(layer_params)
    #l=len(layer_params)
    
    for i in range(0,epochs):
        count=0
        
        for j in range(0,100):
            count_train=0
            A,Z=fwd_prop(train_x[j*500:((j+1)*500)],parameters)
            cost=CrossEntropy(A,train_y[j*500:((j+1)*500)])
        
            grad,parameters=bkd_prop(A,train_y[j*500:((j+1)*500)],parameters,Z,alpha)
            costs.append(cost)
        
        if (i==20):
            alpha=alpha/2
        if(i==40):
            alpha=alpha/2
        
        t=len(layer_params)-1
        Atrain,Ztrain=fwd_prop(train_x,parameters)
        Atrain=Atrain["A"+str(t)].T
        
        AV,ZV = fwd_prop(vald_x,parameters)
        cost1=CrossEntropy(AV,vald_y)
        costs1.append(cost1) 
        
       
        AV1=AV["A"+str(t)]
        AV1=AV1.T
        
        for k in range(train_y.shape[0]):
            a=np.argmax(Atrain[k])
            b=next((i for i, x in enumerate(train_y[k]) if x), None)
        
            if(a==b):
                count_train=count_train+1
        
        for k in range(vald_y.shape[0]):
            a=np.argmax(AV1[k])
            b=next((i for i, x in enumerate(vald_y[k]) if x), None)
        
            if(a==b):
                count=count+1
        
        count=count/100
        count_train=count_train/500
        cc.append(count)
        cc_train.append(count_train)
        print('train_acc',count_train)
        print("Count for correctly classified digits is: ",count)
    
    plt.figure(1)
    plt.plot(costs1)
    plt.ylabel('cost')
    plt.xlabel('epochs')
    plt.title("COST FUNCTION")
    plt.show()
    
    plt.figure(2)
    pylab.plot(cc,'-r',label='Validation Accuracy')
    pylab.plot(cc_train,'-b',label='Training Accuracy')
    pylab.ylabel('Accuracy')
    pylab.xlabel('epochs')
    pylab.title("VALIDATION ACCURACY FOR TANH (0.25)")
    pylab.legend(loc='upper left')
    pylab.show()

    return parameters, costs, costs1,A,Z,grad,cc
    
parameters,costs,cost1,A,Z,grad,cc = Llayermodel(train_x, train_y, vald_x, vald_y, layer_params, epochs = 50, print_cost = True)

DATA_FNAME1 = '/Users/chaitralikshirsagar/Desktop/USC_Fall2018/EE599/HW3/mlp_ck_tanh25.hdf5'
with h5py.File(DATA_FNAME1, 'w') as hf:
	hf.attrs['act'] = np.string_("relu")
	hf.create_dataset('w1',data=parameters["W1"])
	hf.create_dataset('b1',data=parameters["b1"])
	hf.create_dataset('w2',data=parameters["W2"])
	hf.create_dataset('b2',data=parameters["b2"])
	#hf.create_dataset('w3',data=parameters["W3"])
	#hf.create_dataset('b3',data=parameters["b3"])
	hf.attrs['act'] = np.string_("relu")
	hf.close()
