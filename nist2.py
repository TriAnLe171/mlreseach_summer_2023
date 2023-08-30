# NIST example with one layer, linear activation function
# but using multiple inputs
import numpy as np
import math
import pickle

def load_data():
    f = open('mnist.pkl','rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='iso-8859-1')
    f.close()
    return (training_data, validation_data, test_data)

def convertlabels(labels):
    converted = np.zeros((labels.size,10))
    #print(converted.shape)
    for i in range(labels.size):
        a = np.zeros(10)
        a[labels[i]] = 1
        converted[i,:] = a
    return converted

def sigmoid(x):
    return 1/(1+math.exp(-x))

def sigmoidprime(x):
    return math.exp(-x)/((1+math.exp(-x))**2)


tr_d, va_d, te_d = load_data()
labels = convertlabels(tr_d[1][:])
numimages = 5000

weights = np.random.rand(784,10) 

output_values = np.zeros((numimages,10))
for k in range(numimages):
    outputval = np.zeros(10)
    for j in range(10):
        for i in range(784):
            outputval[j] += sigmoid(weights[i][j]*tr_d[0][k][i])
    output_values[k,:] = outputval    


# compute error
error = 0
for i in range(numimages):
    error += (1/(2*numimages)) * np.linalg.norm(output_values[i][:] - labels[i])**2
#print(error)

#print(inputs)
eta = 0.01

for n in range(10):
    gradient = np.zeros((784,10))
    for k in range(numimages):
        for j in range(10):
            for i in range(784):
                gradient[i][j] = (output_values[k][j] - labels[k][j]) * sigmoidprime(tr_d[0][k][i]) * tr_d[0][k][i]


    weights = weights - eta * gradient

    output_values = np.zeros((numimages,10))
    for k in range(numimages):
        outputval = np.zeros(10)
        for j in range(10):
            for i in range(784):
                outputval[j] += sigmoid(weights[i][j]*tr_d[0][k][i])
        output_values[k,:] = outputval    

    error = 0
    for i in range(numimages):
        error += (1/(2*numimages)) * np.linalg.norm(output_values[i][:] - labels[i])**2
    print(error)
    print(output_values[0])