#Import Required Library 
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import seaborn as sns

#Import Model from scikit 
from sklearn.impute  import SimpleImputer

#Import train_test_split model 
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
#Model Evaluation
from sklearn.metrics import classification_report , confusion_matrix,accuracy_score

from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute  import SimpleImputer



for i in df.columns:
        if df[i].dtype == 'object':
            df[i] = le.fit_transform(df[i])
for i in df.columns:
      if df[i].dtype == 'category':
           df[i] = le.fit_transform(df[i])

df = pd.DataFrame(df,columns=df.columns)
x = df.drop('sex',axis=1)
y  = df['sex']

x_train,x_test , y_train,y_test = train_test_split(x,y,test_size=0.3)

rf = RandomForestClassifier(n_estimators=100)

rf.fit(x_train,y_train)
y_pred = rf.predict(x_test)



print(" classification report \n ", classification_report(y_test,y_pred))
print("-----------------------------------------------")
print("confusion MAtrix  \n",confusion_matrix(y_test,y_pred))

print("-----------------------------------------------",accuracy_score(y_test,y_pred))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)









