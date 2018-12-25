#import all required libraries   
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

#reading and analysing dataset and splitting it
dataset=pd.read_csv(r"heart.csv")
dataset.isnull().sum()
dataset.head()
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,13].values


data = pd.read_csv('heart1.csv')
X = data.iloc[:, [2, 3]].values
Y = data.iloc[:, 4].values
'''
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='?',strategy='mean',axis=0)
imputer=imputer.fit(x[:,:-1])
x[:,:-1]=imputer.transform(x[:,:-1])
'''
#Encoding Categorical Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features =[2,6,10,11,12])
x= onehotencoder.fit_transform(x).toarray()
x = np.delete(x, 0, 1)
x=np.delete(x,3,1)
x=np.delete(x,5,1)
x=np.delete(x,7,1)
x=np.delete(x,10,1)


#replace 1,2,3 with 1 in y_data
for index, item in enumerate(y):   # Last row gives 4 diff types of output , so convert them to 0  or 1 
	if not (item == 0.0):       # that is either Yes or No
		y[index] = 1

#splitting train and test dataset
from sklearn.cross_validation import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,train_size=0.75,random_state=0)
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#scaling variables on -3 to 3 scale
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
train_x[14:16]=sc_x.fit_transform(train_x[14:16])
test_x=sc_x.transform(test_x)
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

from sklearn.naive_bayes import GaussianNB #Naive bayes
model=GaussianNB()
model.fit(train_x,train_y)
prediction=model.predict(test_x)
print('The accuracy of the NaiveBayes is',100*metrics.accuracy_score(prediction,test_y))
from sklearn.naive_bayes import GaussianNB #Naive bayes
classifier3=GaussianNB()
classifier3.fit(X_train,Y_train)
Y_pred=classifier3.predict(X_test)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier3.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Training set)')
plt.xlabel('Cholestrol')
plt.ylabel('Rest Blood Pressure')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier3.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Test set)')
plt.xlabel('Cholestrol')
plt.ylabel('Rest Blood Pressure')
plt.legend()
plt.show()