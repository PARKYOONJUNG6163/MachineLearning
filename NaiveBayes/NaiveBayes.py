
# coding: utf-8

# In[108]:


import pandas as pd
import numpy as np

from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
from sklearn.cross_validation import train_test_split


# In[109]:


# 훈련데이터파일, 입력(독립변수),출력(종속변수), 베이즈 예측 값 읽어오기
train_df = pd.read_table('C:/Users/User/Desktop/NaiveBayes/cyberbullying_attitude_N.txt',sep='\s+')
var_input = pd.read_table('C:/Users/User/Desktop/NaiveBayes/input_GST.txt',sep=',').columns.values.tolist()
var_output = pd.read_table('C:/Users/User/Desktop/NaiveBayes/output_attitude.txt',sep=',').columns.values.tolist()
bayes_pre = pd.read_table('C:/Users/User/Desktop/NaiveBayes/p_output_bayes.txt',sep=',')


# In[110]:


# X_data = train_df[var_output[0]].to_frame()


# In[111]:


# positive_count = 0
# negative_count = 0

X_data = train_df.iloc[:,9:17].values.tolist()
Y_data = train_df.iloc[:,7].values.tolist()
# split_size = int(len(X_data)*0.75)

# X_train = X_data[:split_size]
# X_test = X_data[split_size:]
# Y_train = Y_data[:split_size]
# Y_test = Y_data[split_size:]


# In[112]:


X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.25, random_state=0)


# In[113]:


estimator = BernoulliNB(alpha=1.0)
estimator.fit(X_train, Y_train)

a = estimator.predict(X_train) 
b = estimator.predict_proba(X_train)

df_a = pd.DataFrame(a)
df_b = pd.DataFrame(b)
df_a.columns = ['Att']
df_b.columns = ['zero','one']

df_b.zero.mean()
df_b.one.mean()


# In[115]:


estimator = BernoulliNB(alpha=1.0)
estimator.fit(X_train, Y_train)

Y_predict = estimator.predict(X_train) 
score = metrics.accuracy_score(Y_train, Y_predict)
print(score) 

Y_predict = estimator.predict(X_test) 
score = metrics.accuracy_score(Y_test, Y_predict)
print(score) 

# Y_predict = estimator.predict([[1,1,0,1,1,1,0,1]])
# print(Y_predict)


# In[22]:


print(metrics.classification_report(Y_test, Y_predict)) 
print(metrics.confusion_matrix(Y_test, Y_predict))


# In[7]:


prob_att = train_df.Attitude.mean()
prob_att

