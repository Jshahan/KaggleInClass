
# coding: utf-8

# In[10]:

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import cross_validation
from sklearn.feature_selection import VarianceThreshold

SEED = 42


# In[24]:

train = pd.read_csv('train_final.csv')
train = train.fillna(train.mean())
X_train = train.loc[:, 'F1':'F27']
y_train = train.loc[:, 'Y']
test = pd.read_csv(filepath_or_buffer= 'test_final.csv', index_col = 'id')
test = test.fillna(test.mean())
X_train.corr()
X_train = X_train.drop('F6',1)
test = test.drop('F6',1)
X_train = X_train.drop('F26', 1)
test = test.drop('F26', 1)


# In[25]:

def create_test_submission(filename, prediction):
    content = ['id,Y']
    for i, p in enumerate(prediction):
        content.append('%i,%f' %(i+49999,p))
    f = open(filename, 'w')
    f.write('\n'.join(content))
    f.close()
    print 'Saved'


# In[26]:

X_new = VarianceThreshold(1.3).fit_transform(X_train)
X_test = VarianceThreshold(1.25).fit_transform(test)


# In[27]:

model = xgb.XGBClassifier(max_depth=4, n_estimators=1000, learning_rate=0.01, min_child_weight=5, gamma=.8, subsample=.4, reg_alpha=.5, colsample_bytree=.4,reg_lambda=.93)
model.fit(X_new, y_train)
print "Making prediction and saving results..."
preds = model.predict_proba(X_test)[:,1]
create_test_submission('XGB.csv', preds)

