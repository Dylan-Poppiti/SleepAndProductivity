# array
import numpy as np
from numpy import asarray
from numpy import savetxt
import scipy
import pickle
import itertools
import pandas as pd
import scipy.optimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#from sklearn.externals.joblib as extjoblib
#allows for models to be saved and loaded
import joblib
# models for creating machine learning
from sklearn.tree import DecisionTreeClassifier
#ploting on a graph
import matplotlib.pyplot as plt
from statistics import mean
import os.path

# loop for input of hours of sleep and productivity
ender = 0

#create x, y, x_test, y_test, x_last,y_last arrays

x_last = y_last = x = y = x_test = y_test = np.array([], dtype=np.float64);


#checks to see if there are the saved files present
if os.path.exists('./SleepNSex.npy'):
    # load x and y from respective files
    x = np.load("SleepNSex.npy")
    x_last=np.load("SleepNSex.npy")
    y_last = np.load("Productive.npy")
    y = np.load("Productive.npy");
else:
    #gives random data
    x = np.array([[8,1], [6,1], [24,1], [12,1], [2,1], [7,1], [9,1], [14,1], [4,1],[18,1],[8,0],[9,0],[7,0],[18,0],[22,0],[24,0],[0,0],[6,0],[4,0],[15,0]], dtype=np.float64);
    y =np.array([10, 3, 0, 8, 0, 7, 8, 6, 1,3,12,9,8,2,0,0,0,4,1,4], dtype=np.float64);




# use while loop to give user input until input is end

while ender == 0:
    MoF= input("Enter 0 for Female or 1 For Male(type end to end): ")
    if MoF =="end":
        ender=1
        break
    numHoursOfSleep = input("Enter the amount you have slept today (Type end to end): ")
    if numHoursOfSleep =="end":
        ender = 1
        break
    Productiveness = input("Enter how productive you were today (Type end to end): ")
    if Productiveness =="end":
        ender = 1
        break
    if Productiveness!="end" or numHoursOfSleep !="end" or MoF !="end":
        if MoF == 1 or MoF == 0:
            x.append(numHoursOfSleep, MoF)

            y.append(Productiveness)


# naming axises
hours = 'HoursSlept.pk'
prod = 'Product.pk'

# If the two lists have been added to, they will be saved
if np.array_equal(x_last,x) and np.array_equal(y_last,y):
    print("Old list");
else:
    np.save("SleepNSex.npy",x);
    np.save("Productive.npy",y);

# organizing data by x value, smallest to largest
point =0;

while point < len(x):
    innard = point+1;
    pointer = x[point][0];
    index = point;
    while innard < len(x):
        if pointer > x[innard][0]:
            pointer = x[innard][0];
            index = innard;
        innard =innard+1;
    if index != point:
        temp = x[point][0];
        temp2 = y[point];
        x[point][0] = x[index][0];
        y[point] = y[index];
        x[index][0] = temp;
        y[index] = temp2;
    point = point + 1;

#split data between male and female points only

yy=xx=xm=xf=yf=ym=[];
count =0;
while count < len(x):
    if(x[count][1]==0):
        xf.append(x[count][0])
        yf.append(y[count])
    else:
        xm.append(x[count][0])
        ym.append(y[count])
    xx[count]=x[count][0]
    yy[count]=y[count]
    count=count+1;

xp= np.array(xx)
yp= np.array(yy)
xs = xp.reshape(-1,1)
ys = yp.reshape(-1,1)

xmp = np.array(xm);
ymp = np.array(ym);
yms = ymp.reshape(-1,1)
xms = xmp.reshape(-1,1)

xfp = np.array(xf);
yfp = np.array(yf);
yfs = yfp.reshape(-1,1)
xfs = xfp.reshape(-1,1)

#tester model

#checks if they are already saved files
if os.path.exists('./productive-predict.joblib'):
    model = joblib.load('productive-predict.joblib');
    modelM=joblib.load('prodM_predict.joblib');
    modelF=joblib.load('prodF_predict.joblib')
else:
    model = DecisionTreeClassifier();
    modelM = DecisionTreeClassifier();
    modelF = DecisionTreeClassifier();

# gives test and train data for each set of data
x_train,y_train,x_test,y_test = train_test_split(xs,ys,test_size=.5)
xf_train,yf_train,xf_test,yf_test = train_test_split(xfs,yfs,test_size=.5)
xm_train,ym_train,xm_test,ym_test = train_test_split(xms,yms,test_size=.5)

#fits the data in the model
model.fit(x_train, y_train);
modelM.fit(xm_train,ym_train);
modelF.fit(xf_train,yf_train);

#make predictions using the training and fit
predictions = model.predict(x_test);
predictM = model.predict(x_test);
predictF = model.predict(y_test);

#checks the accuracy of each model
score = accuracy_score(y_test, predictions);
scoreM = accuracy_score(ym_test, predictM);
scoreF = accuracy_score(yf_test,predictF);

#save the models
joblib.dump(model, 'productive-predict.joblib')
joblib.dump(modelM,'prodM_predict.joblib')
joblib.dump(modelF,'prodF_predict.joblib')

print("Score for the whole data set: ", score)
print("Score for the male data set: ", scoreM)
print("Score for the female data set: ", scoreF)


#creates scatter plot graph
plt.scatter(xx, yy, label ="All", color = "green", marker ="*", s=30);
plt.scatter(xm,ym, label ="Men",color="blue",marker ="*",s=30);
plt.scatter(xf,yf, label="Women",color="red",marker="*", s=30)
#prediction point
plt.scatter(x_test, predictions, label="Prediction", color="green", marker="x", s=30);
plt.scatter(xm_test,predictM, label ="M Guess",color = "blue",marker="x",s=30);
plt.scatter(xf_test,predictF, label="F Guess",color="red", marker="x",s=30);

plt.xlabel('Hours Slept')
plt.ylabel('Productiveness')
plt.show()
