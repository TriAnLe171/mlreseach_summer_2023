# World's worst classifier
# Uses a single image from NIST dataset
# Uses a single layer classifier
# Can train to learn that 5 is 5
# Wow



import numpy as np
import random
import math
import pickle
import gzip 


def load_data():
    f = open('mnist.pkl','rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='iso-8859-1')
    f.close()
    return (training_data, validation_data, test_data)

tr_d, va_d, te_d = load_data()
#print(tr_d[1][0])
#print(va_d[0][0])
#print(len(va_d[0][0]))
#print(len(tr_d))
  # the simplest way to generate random numbers


weights = np.random.rand(784,10) 

inputs = tr_d[0][0]
trueval = np.array([0,0,0,0,0,1,0,0,0,0])
outputval = np.zeros(10)

for j in range(10):
    for i in range(784):
        outputval[j] += weights[i][j] * inputs[i]

#outputval = outputval / np.linalg.norm(outputval)

print(outputval)

error = .5*np.linalg.norm(outputval - trueval)**2
print(error)

#print(inputs)
gradient = np.zeros((784,10))

for j in range(10):
    for i in range(784):
        gradient[i][j] = (outputval[j]-trueval[j])*inputs[i]

#print(gradient)
#print(gradient.shape)

eta = 0.01

for k in range(100):

    gradient = np.zeros((784,10))

    for j in range(10):
        for i in range(784):
            gradient[i][j] = (outputval[j]-trueval[j])*inputs[i]

    weights = weights - eta * gradient

    outputval = np.zeros(10)
    for j in range(10):
        for i in range(784):
            outputval[j] += weights[i][j] * inputs[i]

    print(outputval)

    error = .5*np.linalg.norm(outputval - trueval)**2

    print(error)