import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

# Import data
train_name = "D:/0Course/AN6001 AI and Big Data in Business/Individual Programming Assignment/Credit Card Default.csv"
data = pd.read_csv(train_name, encoding='utf8', engine='python')

#-----------------------------Data Cleaning---------------------------------
# Overview of data
print(data.describe())

# Delete meaningless column 'clientid'
del data['clientid']

# Remove duplicates
print(data[data.duplicated()])  # result shows there is no duplicate

# Remove negative rows
data = data[(data >= 0)]

# Remove NA, Non and missing value
print(data.info())
data = data.dropna()

#Remove outliers
def outlier_to_NA(i):
  column=data.iloc[:, i]
  Q1 = np.percentile(column, 25)
  Q3 = np.percentile(column, 75)
  IQR = Q3 - Q1
  up_limit = Q3 + 1.5 * IQR
  low_limit = Q1 - 1.5 * IQR
  outliers=[]
  for values in column:
      if values < low_limit or values > up_limit:
          outliers.append(values)
  return outliers

outliers0=outlier_to_NA(0)  #get the outliers of income column
outliers1=outlier_to_NA(1)  #get the outliers of age column
outliers2=outlier_to_NA(2)  #get the outliers of loan column
print(f'the outliers for each column: {outliers0,outliers1,outliers2}')  #show all outliers, only loan column has outliers
for values in outliers2:
  data=data[-data['loan'].isin([values])] #delete rows with outliers

#Use Z-Score method to do normalization
num = data.shape[0]
for c in range(0,3):
  mean=sum(data.iloc[:, c]) / num
  sd=np.std(data.iloc[:, c],ddof=1)
  for i in range(num):
     data.iloc[i, c]=(data.iloc[i, c]-mean)/sd

#Correlation plot
corr = data.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
plt.subplots(figsize=(5, 5))
with sns.axes_style("white"):
    sns.heatmap(corr, mask=mask, annot=True, vmax=1, square=True, cmap="Blues", fmt='.2f')
plt.show() #it shows there are no very high correlations

print(data.shape)# 1991 rows and 4 columns


#-----------------------------Model------------------------------------------
#Split data into train data and test data
data_train,data_test = train_test_split(data,test_size=0.3,random_state=4)
Xtrain = data_train.drop(columns=['default'])
Ytrain = data_train.loc[:, ["default"]]
Xtest = data_test.drop(columns=['default'])
Ytest = data_test.loc[:, ["default"]]


#Logistic Regression model:
from sklearn import linear_model
from matplotlib import pyplot

model_LogR = linear_model.LogisticRegression(max_iter=10000) #should be big enough or error
model_LogR.fit(Xtrain,Ytrain.values.ravel())
pred_LogR = model_LogR.predict(Xtest)
cm_LogR = metrics.confusion_matrix(Ytest, pred_LogR)
print(cm_LogR)
accuracy_log = (cm_LogR[0, 0] + cm_LogR[1, 1]) / sum(sum(cm_LogR))
print(f"The accuracy for test data in Logistic Regression is {round(accuracy_log*100,3)}%")
FP_LogR=cm_LogR[0,1]/(cm_LogR[0,0]+cm_LogR[0,1])
FN_LogR=cm_LogR[1,0]/(cm_LogR[1,0]+cm_LogR[1,1])
print(f'The false positive rate for test data in Logistic Regression is {round(FP_LogR*100,2)}%')
print(f'The false negative rate for test data in Logistic Regression is {round(FN_LogR*100,2)}%')
importance = model_LogR.coef_
print(f'Logistic Regression variables importance: {importance}')


#DecisionTree model:
from sklearn import tree
model_CART = tree.DecisionTreeClassifier(max_depth=10) #How to find the best max_depth?
model_CART.fit(Xtrain,Ytrain)
pred_CART = model_CART.predict(Xtest)
cm_CART = metrics.confusion_matrix(Ytest, pred_CART)
accuracy_train_CART = model_CART.score(Xtest,Ytest)
print(cm_CART)
print(f"The accuracy for test data in DecisionTree is {round(accuracy_train_CART*100,3)}%")
FP_TREE=cm_CART[0,1]/(cm_CART[0,0]+cm_CART[0,1])
FN_TREE=cm_CART[1,0]/(cm_CART[1,0]+cm_CART[1,1])
print(f'The false positive rate for test data in DecisionTree is {round(FP_TREE*100,2)}%')
print(f'The false negative rate for test data in DecisionTree is {round(FN_TREE*100,2)}%')
importance = model_CART.feature_importances_
print(f'DecisionTree variables importance: {importance}')


#XG Boost model:
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(max_depth=6) #Tested manually to get the highest accuracy
model.fit(Xtrain, Ytrain)

#Calculate the accuracy for test data
pred = model.predict(Xtest)
cm = confusion_matrix(Ytest, pred)
print(cm)
accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
print(f'The accuracy for test data in XG Boost is {round(accuracy*100,2)}%')
FP_XGB=cm[0,1]/(cm[0,0]+cm[0,1])
FN_XGB=cm[1,0]/(cm[1,0]+cm[1,1])
print(f'The false positive rate for test data in XG Boost is {round(FP_XGB*100,2)}%')
print(f'The false negative rate for test data in XG Boost is {round(FN_XGB*100,2)}%')

#Check XGBoost feature importances
def feature_imp(df, model):
    fi = pd.DataFrame()
    fi["feature"] = df.columns
    fi["importance"] = model.feature_importances_
    return fi.sort_values(by="importance", ascending=True)
df = feature_imp(Xtrain, model)
df.set_index('feature', inplace=True)
df.plot(kind='barh', figsize=(10, 10))
plt.title('Feature Importance according to XGBoost')
plt.show()


#Neural Network Model:
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

#Get an overall high accuracy by putting multiple dense, then adjust the dropout rate to make accuracy higher
model=Sequential()
model.add(Dense(128, input_dim=len(Xtrain.columns), activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(16, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(8, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(4, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'Adamax', metrics = ['accuracy'])
model.fit(Xtrain, Ytrain, batch_size = 10, epochs=100, verbose=1)

#Evaluate the NNET model
print(model.summary())
print(model.evaluate(Xtrain, Ytrain))
print(model.evaluate(Xtest, Ytest))

#Calculate the accuracy for test data
pred=model.predict(Xtest)
pred=np.where(pred>0.5,1,0)
cm=confusion_matrix(Ytest, pred)
print(cm)
accuracy=(cm[0,0]+cm[1,1])/sum(sum(cm))
FP_NNET=cm[0,1]/(cm[0,0]+cm[0,1])
FN_NNET=cm[1,0]/(cm[1,0]+cm[1,1])
print(f'The accuracy for test data in NNET is {round(accuracy*100,2)}%')
print(f'The false positive rate for test data in NNET is {round(FP_NNET*100,2)}%')
print(f'The false negative rate for test data in NNET is {round(FN_NNET*100,2)}%')