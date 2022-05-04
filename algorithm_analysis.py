import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,mean_squared_error
import xgboost
data=pd.read_csv("police_stop_data.csv")
data=data.loc[:,["problem","personSearch","race","gender","vehicleSearch"]].dropna(axis=0)

data=data[data["race"]!="Unknown"]
#data['problem']=data["problem"].apply(lambda x:1 if "Suspicious Person" in x else 0)
data['personSearch']=data['personSearch'].apply(lambda x:1 if "YES" in x else 0)
data["vehicleSearch"]=data["vehicleSearch"].apply(lambda x:1 if "YES" in x else 0)
data["race"]=data["race"].apply(lambda x:0 if "White" in x else 0)
data["gender"]=data["gender"].apply(lambda x:1 if"Male" in x else 0)
data=pd.get_dummies(data)
print(data.columns)

train_data=data.iloc[:int(data.shape[0]*0.8)]
test_data=data.iloc[int(data.shape[0]*0.8):]
miniroty=data[data["race"]!="White"]
others=data[data["race"]=="White"]
model=LogisticRegression(max_iter=1000)
#
model.fit(train_data.iloc[:,1:],train_data.iloc[:,0])
res=model.predict(test_data.iloc[:,1:])
print("Logistic Regression Accuracy=",accuracy_score(test_data.iloc[:,0],res))
print("Logistic RMSE=",np.sqrt(mean_squared_error(test_data.iloc[:,0],res)))

xgb=xgboost.XGBRegressor()
xgb.fit(train_data.iloc[:,1:],train_data.iloc[:,0])
res2=xgb.predict(test_data.iloc[:,1:])
res_xgb=[]
for i in res2:
    if i<0.5:
        res_xgb.append(0)
    else:
        res_xgb.append(1)
print("XGboost Accuracy=",accuracy_score(test_data.iloc[:,0],res_xgb))
print("XGboost RMSE:",np.sqrt(mean_squared_error(test_data.iloc[:,0],res_xgb)))
def stat_parity(preds, sens):

    cone = 0
    czero = 0
    one = 0
    zero = 0
    for i in range(len(preds)):
        if sens[i] == 1:
            one = one + 1
        else:
            zero = zero + 1
        if sens[i] == 1 and preds[i] == 1:
            cone = cone + 1
        if sens[i] == 0 and preds[i] == 1:
            czero = czero + 1

    if zero == 0 and one == 0:
        return 0
    elif zero == 0:
        return 0 - cone / one
    elif one == 0:
        return czero / zero
    else:
        return czero / zero - cone / one

def eq_oppo(preds, sens, labels):

    cone=0
    czero=0
    one=0
    zero=0
    for i in range(len(preds)):
        if sens[i]==1 and labels[i]==1:
            one=one+1
            if preds[i]==1:
                cone=cone+1
        if sens[i]==0 and labels[i]==1:
            zero=zero+1
            if preds[i]==1:
                czero=czero+1
    if zero==0 and one==0:
        return 0
    elif zero==0:
        return 0-cone/one
    elif one==0:
        return czero/zero
    else:
        return czero/zero-cone/one
#
print("statistical parity:",stat_parity(res,np.array(test_data.iloc[:,1])))
# print(stat_parity(res2,np.array(test_data.iloc[:,1])))
# # print()
# print(data.drop(["race"],axis=1).columns)
# for r in res2:
#     if r>0.5:
#         print(r)
print("equalized opportunity: ",eq_oppo(res,np.array(test_data.iloc[:,1]),np.array(test_data.iloc[:,0])))