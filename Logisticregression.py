import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

dataset=pd.read_csv(r"heart.csv")
dataset.isnull().sum()
dataset.head()
from sklearn.preprocessing import Imputer

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,13].values
'''
imputer=Imputer(missing_values='?',strategy='mean',axis=0)
imputer=imputer.fit(x[:,:-1])
x[:,:-1]=imputer.transform(x[:,:-1])
'''
data = pd.read_csv('heart1.csv')
X = data.iloc[:, [2, 3]].values
Y = data.iloc[:, 4].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features =[2,6,10,11,12])
x= onehotencoder.fit_transform(x).toarray()
x = np.delete(x, 0, 1)
x=np.delete(x,3,1)
x=np.delete(x,5,1)
x=np.delete(x,7,1)
x=np.delete(x,10,1)





 
for index, item in enumerate(y):   # Last row gives 4 diff types of output , so convert them to 0  or 1 
	if not (item == 0.0):       # that is either Yes or No
		y[index] = 1

from sklearn.cross_validation import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,train_size=0.75,random_state=0)
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


#plotting how much percentage is effected by heart problems
#percentage of heart disease
f,ax=plt.subplots(1,2,figsize=(18,8))
dataset['prediction'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Heart_diseased')
ax[0].set_ylabel('')
sns.countplot('prediction',data=data,ax=ax[1])
ax[1].set_title('prediction')
plt.show()



#Correlation Matrix
sns.heatmap(dataset.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()



#age and gender for heartdisease
f,ax=plt.subplots(1,2,figsize=(18,8))
sns.violinplot("sex","age", hue="prediction", data=dataset,split=True,ax=ax[0])
ax[0].set_title('Age vs Gender vs Heart_disease')
ax[0].set_yticks(range(0,110,10))
plt.show()

'''
from sklearn.preprocessing import scale
train_x_scale=scale(train_x[['age','trestbps','chol','thalach','oldpeak']])
test_x_scale=scale(test_x[['age','trestbps','chol','thalach','oldpeak']])
'''

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
train_x[14:16]=sc_x.fit_transform(train_x[14:16])
test_x=sc_x.transform(test_x)
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)
'''
from sklearn.decomposition import PCA
modelSVM = LinearSVC(C=0.1)
pca = PCA(n_components=2, whiten=True).fit(x)   # n denotes number of components to keep after Dimensionality Reduction
X_new = pca.transform(x)
from sklearn.cross_validation import train_test_split
train_x1,test_x1,train_y1,test_y1=train_test_split(X_new,y,train_size=0.75,random_state=0)
'''

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(train_x,train_y)
accuracy = 0.38
pred_y=classifier.predict(test_x)

from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(random_state = 0)
classifier_lr.fit(X_train, Y_train)
Y_pred = classifier_lr.predict(X_test)
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(pred_y,test_y)
from sklearn.metrics import classification_report
print(classification_report(test_y, pred_y))
print('Accuracy for logistic_regression is ',100*(metrics.accuracy_score(pred_y,test_y)))




# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier_lr.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Cholestrol')
plt.ylabel('Rest Blood Pressure')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier_lr.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Cholestrol')
plt.ylabel('Rest Blood Pressure')
plt.legend()
plt.show()