#importing libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pn

#importing data set
test_dataset = pn.read_csv("test.csv")

#creating variables x & y for testing & training
x_train =  test_dataset.iloc[0:241 , 0].values
y_train = test_dataset.iloc[0:241 , 1].values
x_test = test_dataset.iloc[241: , 0].values
y_test = test_dataset.iloc[241: , 1].values

#Plotting the points
plt.scatter(x_train,y_train)
plt.scatter(x_test,y_test)

#Cost Function : Mean Squared Error
def costfunction(m,c,x,y):
    n=len(x_test)
    error = 0
    cost = 0
    for i in range (0,n):
        error = ((m*x[i]+c)-y[i])
        cost = cost + error**2
    return cost/float(n)

#Function to find new values of c & m
def gradientdescent(m,c,x,y,learning_rate):
    c_gradient = 0
    m_gradient = 0
    n=len(x)
    for i in range(0,len(x)):
        x_value = x[i]
        y_value = y[i]
        c_gradient += 2/n * ((m*x_value + c)-y_value)
        m_gradient += 2/n * ((m*x_value + c)-y_value)
    new_m= m - (learning_rate*m_gradient)
    new_c= c - (learning_rate*c_gradient)
    return new_m,new_c

def gradient_algo(x,y,learning_rate):
    m=0
    c=0
    for i in range(0,len(x)):
        m, c = gradientdescent(m,c,x,y,learning_rate)
    return m ,c 

#prediction function
def predict(m,c,x,y):
    y_predict=[]
    for i in range(0,len(x)):
        prediction=m*x[i]+c
        y=y_predict.append(prediction)
    return y_predict

#main Function
    
learning_rate=0.001
final_m,final_c=gradient_algo(x_train,y_train,learning_rate)
y_predict = predict(final_m,final_c,x_test,y_test)
print("Y-Test  Y-Prediction")
for i in range(len(x_test)):
    print(str(y_test[i])+"  "+str(y_predict[i]))
print("MSE on train dataset:"+str(costfunction(0,0,x_train,y_train)))

#plots the Line using final_c & final_m

plt.plot([0,100],[final_c,100*final_m+final_c])
plt.show()

