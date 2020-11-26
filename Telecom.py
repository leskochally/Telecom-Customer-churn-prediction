
"Telecom Churn"

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

data=pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

#PREPROCESSING
data.info()
data.isnull().sum()# There arent any null values but there are some columns that we need to change

data.nunique()# Here we can there is some problem with the data as Internet Service is having three values
#wherein it should be either yes or no , so need to find those columns and convert it

data["TotalCharges"] = pd.to_numeric(data["TotalCharges"],errors="coerce")
col_replaced = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection','TechSupport','StreamingTV', 'StreamingMovies']
for i in col_replaced  : 
    data[i]  = data[i].replace({'No internet service' : 'No'})

data["MultipleLines"] = data["MultipleLines"].replace({'No phone service':'No'})

#Coverting the tenure period to specific Category to easily Identify
def tenure_lab(data) :
    
    if data["tenure"] <= 12 :
        return "0-12"
    elif (data["tenure"] > 12) & (data["tenure"] <= 24 ):
        return "12-24"
    elif (data["tenure"] > 24) & (data["tenure"] <= 48) :
        return "24-48"
    elif (data["tenure"] > 48) & (data["tenure"] <= 60) :
        return "48-60"
    elif data["tenure"] > 60 :
        return ">60"
data["tenure_group"] = data.apply(lambda data:tenure_lab(data),axis = 1)

#Converting the Dependend Variable to 1 and 0 (1-Yes,0-N0)
for i in range(0,len(data["Churn"])):
    if(data["Churn"][i]=="Yes"):
        data["Churn"][i]=1
    else:
        data["Churn"][i]=0
data=data.iloc[:,2:]
data["Churn"] = pd.to_numeric(data["Churn"],errors="coerce")        
telecom_encoded_df = pd.get_dummies(data)

#Plotting the Correlation Matrix
correlation = telecom_encoded_df.corr()
#Took Columns which are Positively correlated with the Depended Variable
telecom_encoded_df_new = telecom_encoded_df.iloc[:,[0,2,4,5,7,10,12,14,16,18,20,22,25,27,28,32,35,37,38]]  
correlation_check = telecom_encoded_df_new.corr()

#Splitting into X and Y features 
Y = telecom_encoded_df_new.iloc[:,[2]]
X = telecom_encoded_df_new.drop(['Churn'], axis=1)

#Spliting into Training and Testing
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y =  train_test_split(X,Y,train_size= 0.8 , random_state=5)


#Logistic Regression to find out the Summary of the model
import statsmodels.api as sm
model_1 = sm.Logit(train_y, train_x).fit()
model_1.summary2()

X_features= X.columns
#Checking Multicolinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
def get_vif_factors( X ):
    X_matrix = X.as_matrix()
    vif = [ variance_inflation_factor( X_matrix, i ) for i in range( X_matrix.shape[1] ) ]
    vif_factors = pd.DataFrame()
    vif_factors['column'] = X.columns
    vif_factors['vif'] = vif
    return vif_factors

vif_factors = get_vif_factors( X[X_features] )
vif_factors

#Dropping Variable which have Multicolinearity Properties as well as those columns whose p values are more than 0.05
X=X.drop(['MonthlyCharges','PhoneService_Yes','Partner_No','Dependents_No'],axis = 1)
train_x,test_x,train_y,test_y =  train_test_split(X,Y,train_size= 0.8 , random_state=5)



#Final Model
model_2 = sm.Logit(train_y, train_x).fit()
model_2.summary2()
#Used Model2 as the basis for all the otheer models i.e using the same independend variables

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC


#Logistic Regression
model_lr = LogisticRegression(solver='lbfgs')
model_lr.fit(train_x,train_y)


#Buliding the NB model
model_nb = GaussianNB()
model_nb.fit(train_x,train_y)

print("Coefficients of the Logistic Regression model")
coef = model_lr.coef_
intercept = model_lr.intercept_

#KNN
model_knn = KNeighborsClassifier(n_neighbors=83)#took the square root of the number of rows if the classes are evn take an odd number
model_knn.fit(train_x,train_y)

#SVC
model_svm = SVC(kernel = 'rbf', random_state = 0)
model_svm.fit(train_x, train_y)

#predicting the Training Dataset
predicted_classes_lr = model_lr.predict(train_x)
predicted_classes_nb = model_nb.predict(train_x)
predicted_classes_knn = model_knn.predict(train_x)
predicted_classes_svm = model_svm.predict(train_x)


#Confusion Matrices for all the models(Training)
print("Confusion Matrix for LR Model")
conf_mat_lr = confusion_matrix(train_y,predicted_classes_lr)
print(conf_mat_lr)
sn.heatmap(conf_mat_lr,annot=True)
plt.xlabel("Predicted Classes")
plt.ylabel("Actual Classes")

print("Confusion Matrix for NB Model")
conf_mat_nb = confusion_matrix(train_y,predicted_classes_nb)
print(conf_mat_nb)
sn.heatmap(conf_mat_nb,annot=True)
plt.xlabel("Predicted Classes")
plt.ylabel("Actual Classes")

print("Confusion Matrix for KNN Model")
conf_mat_knn = confusion_matrix(train_y,predicted_classes_knn)
print(conf_mat_knn)
sn.heatmap(conf_mat_knn,annot=True)
plt.xlabel("Predicted Classes")
plt.ylabel("Actual Classes")

print("Confusion Matrix for SVM Model")
conf_mat_svm = confusion_matrix(train_y,predicted_classes_svm)
print(conf_mat_svm)
sn.heatmap(conf_mat_svm,annot=True)
plt.xlabel("Predicted Classes")
plt.ylabel("Actual Classes")

#Predicting the Depended Variable from the Testing Dataset
test_predicted_classes_lr = model_lr.predict(test_x)
test_predicted_classes_nb = model_nb.predict(test_x)
test_predicted_classes_knn = model_knn.predict(test_x)
test_predicted_classes_svm = model_svm.predict(test_x)

#Confusion Matrices for all the models(Testing)
print("Confusion Matrix for LR Model")
test_conf_mat_lr = confusion_matrix(test_y,test_predicted_classes_lr)
print(test_conf_mat_lr)
sn.heatmap(test_conf_mat_lr,annot=True)
plt.xlabel("Predicted Classes")
plt.ylabel("Actual Classes")

print("Confusion Matrix for NB Model")
test_conf_mat_nb = confusion_matrix(test_y,test_predicted_classes_nb)
print(test_conf_mat_nb)
sn.heatmap(test_conf_mat_nb,annot=True)
plt.xlabel("Predicted Classes")
plt.ylabel("Actual Classes")

print("Confusion Matrix for KNN Model")
test_conf_mat_knn = confusion_matrix(test_y,test_predicted_classes_knn)
print(test_conf_mat_knn)
sn.heatmap(test_conf_mat_knn,annot=True)
plt.xlabel("Predicted Classes")
plt.ylabel("Actual Classes")

print("Confusion Matrix for SVM Model")
test_conf_mat_svm = confusion_matrix(test_y,test_predicted_classes_svm)
print(test_conf_mat_svm)
sn.heatmap(test_conf_mat_svm,annot=True)
plt.xlabel("Predicted Classes")
plt.ylabel("Actual Classes")


#Accuracy Score for Training 
accuracy_lr = accuracy_score(train_y,predicted_classes_lr)
print(accuracy_lr)
accuracy_nb = accuracy_score(train_y,predicted_classes_nb)
print(accuracy_nb)
accuracy_knn = accuracy_score(train_y,predicted_classes_knn)
print(accuracy_knn)
accuracy_svm = accuracy_score(train_y,predicted_classes_svm)
print(accuracy_svm)

#Accuracy Score for Testing 
test_accuracy_lr = accuracy_score(test_y,test_predicted_classes_lr)
print(test_accuracy_lr)
test_accuracy_nb = accuracy_score(test_y,test_predicted_classes_nb)
print(test_accuracy_nb)
test_accuracy_knn = accuracy_score(test_y,test_predicted_classes_knn)
print(test_accuracy_knn)
test_accuracy_svm = accuracy_score(test_y,test_predicted_classes_svm)
print(test_accuracy_svm)

