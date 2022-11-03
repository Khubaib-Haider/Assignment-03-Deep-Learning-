import winsound
import numpy as np
import matplotlib.pyplot as plt
import NFunctions as f
import loadimage as lm

Dataset_path = 'F:\Masters RIME\Third Semester\Deep learning\Assignemnts\Train_Images'
Prediction_files_path = 'F:\Masters RIME\Third Semester\Deep learning\Assignments\Test_Images'

x_train,y_train = lm.train(Dataset_path)
x_test,y_test = lm.test(Prediction_files_path)
x_train = x_train.T
y_train = y_train.reshape(1,x_train.shape[1])

x_test = x_test.T
y_test = y_test.reshape(1,x_test.shape[1])

print('shape of x_train:',x_train.shape) # shape of x_train: ( 10000,20)  (n,m)
print('shape of y_train:',y_train.shape) # shape of y_train: (1,10000)
print('shape of x_test:',x_test.shape)
print('shape of y_test:',y_test.shape)

m_train = x_train.shape[1]
print('# of training examples is ',m_train)
n = x_train.shape[0]

def training_model (x, y, l1, l2, l3, learning_rate, iterations, L):    
    
    n_x = x.shape[0]
    n_y = y.shape[0]
    w1, b1, w2, b2, w3, b3, w4, b4 = f.initializing(n_x, l1, l2, l3, n_y)
    
    cost_list = []
    for i in range(iterations):
        z1, a1, z2, a2, z3, a3, z4, a4 = f.forward_prop(x, w1, b1, w2, b2, w3, b3, w4, b4)
        cost = f.cost_function(a4, y) + f.regularization(w1, w2, w3, w4, L, m_train)
        dz4, dw4, db4, dz3, dw3, db3, dz2, dw2, db2, dz1, dw1, db1 = f.back_prop(x, y, w1, b1, w2,b2, w3, b3, w4,b4, z1, a1, z2, a2, z3, a3, z4, a4)

        w1, b1, w2, b2, w3, b3, w4, b4 = f.update_parameters(w1, b1, w2, b2, w3, b3, w4, b4, dz4, dw4, db4, dz3, dw3, db3,dz2, dw2, db2, dz1, dw1, db1, learning_rate)
        
        cost_list.append(cost)
        if (i % 10 == 0):
            print('cost after ', i, 'th iteration :', cost )
            
    print('Mean Training Error   :',(np.min(cost_list)))    
        
    return  w1, b1, w2, b2, w3, b3, w4, b4, cost_list    

l1 = 1500             # neurons in hidden layer 1
l2 = 750              # neurons in hidden layer 2
l3 = 200               # neurons in hidden layer 3
learning_rate = 0.99    # Learning rate
iterations = 1000        # number of iterations 
L = 0.001                 # regularization parameter


w1, b1, w2, b2, w3, b3, w4, b4, cost_list = training_model (x_train, y_train, l1, l2, l3, learning_rate, iterations, L)

plot = np.arange(0,iterations)
plt.plot(plot,cost_list) 
plt.show()

Accuracy,tp,tn,fp,fn, recall, precision,F1Score = f.prediction( x_test,y_test,w1, w2, w3, w4, b1, b2, b3, b4, threshold=0.5)  

duration = 1000
freq = 550
winsound.Beep(freq, duration)