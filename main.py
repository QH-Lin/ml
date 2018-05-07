# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn import metrics

data = pd.read_csv("./bank-additional/bank-additional-full.csv", sep=";")

# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=100)
    model.fit(train_x, train_y)
    return model

test_classifiers = ['KNN', 'LR', 'RF', 'GBDT']
classifiers = {
              'KNN':knn_classifier,
               'LR':logistic_regression_classifier,
               'RF':random_forest_classifier,
             'GBDT':gradient_boosting_classifier
}


def predict(model, X, y):
    y_pred = model.predict(X)
    y_predprob = model.predict_proba(X)[:,1]
    acc = metrics.accuracy_score(y.values, y_pred)

    auc =  metrics.roc_auc_score(y, y_predprob)
    print("Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred))
    print("AUC Score : %f" % metrics.roc_auc_score(y, y_predprob))
    return acc, auc

# 预测列 yes no 变为 1 0
data['y'].replace(['yes', 'no'], [1, 0], inplace=True)
# one-hot处理
data_one_hot = pd.get_dummies(data)

result = {}
# result['acc'] = {'KNN':[], 'LR':[], 'RF':[], 'GBDT':[]}
# result['auc'] = {'KNN':[], 'LR':[], 'RF':[], 'GBDT':[]}

for classifier in classifiers:
    result[classifier] = {}
    result[classifier]['acc'] = []
    result[classifier]['auc'] = []

    for i in np.arange(0.1,1,0.1):
        train, test = train_test_split(data_one_hot, test_size=1 - i,
                                            random_state=123)
        train_x = train.drop('y', axis=1)
        train_y = train['y']
        test_x = test.drop('y', axis=1)
        test_y = test['y']

        print('******************* %s ********************' % classifier)
        model = classifiers[classifier](train_x, train_y)
        acc, auc = predict(model, test_x, test_y)
        result[classifier]['acc'].append(acc)
        result[classifier]['auc'].append(auc)


import numpy as np
import matplotlib.pyplot as plt
#X轴，Y轴数据
x = np.arange(0.1,1,0.1)
plt.figure(figsize=(8,4)) #创建绘图对象
for i in result:
    y = result[i]['acc']
    plt.plot(x,y,"--",linewidth=1,label = i)   #在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）

plt.xlabel("training set ratio") #X轴标签
plt.ylabel("Accuracy")  #Y轴标签
plt.legend()
plt.show()  #显示图
# plt.savefig("line.jpg") #保存图

import numpy as np
import matplotlib.pyplot as plt
#X轴，Y轴数据
x = np.arange(0.1,1,0.1)
plt.figure(figsize=(8,4)) #创建绘图对象
for i in result:
    y = result[i]['auc']
    plt.plot(x,y,"--",linewidth=1,label = i)   #在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）

plt.xlabel("training set ratio") #X轴标签
plt.ylabel("AUC")  #Y轴标签
plt.legend()
plt.show()  #显示图
# plt.savefig("line.jpg") #保存图


import xgboost as xgb
train_x = X
train_y = y
test_x = X
test_y = y
dtrain=xgb.DMatrix(train_x,label=train_y)
dtest=xgb.DMatrix(test_x)

params={'booster':'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth':4,
    'lambda':10,
    'subsample':0.75,
    'colsample_bytree':0.75,
    'min_child_weight':2,
    'eta': 0.025,
    'seed':0,
    'nthread':8,
     'silent':1}

watchlist = [(dtrain,'train')]

bst=xgb.train(params,dtrain,num_boost_round=100,evals=watchlist)

ypred=bst.predict(dtest)

# 设置阈值, 输出一些评价指标
y_pred = (ypred >= 0.5)*1

from sklearn import metrics
print('AUC: %.4f' % metrics.roc_auc_score(test_y,ypred))
print('ACC: %.4f' % metrics.accuracy_score(test_y,y_pred))
print('Recall: %.4f' % metrics.recall_score(test_y,y_pred))
print('F1-score: %.4f' %metrics.f1_score(test_y,y_pred))
print('Precesion: %.4f' %metrics.precision_score(test_y,y_pred))
metrics.confusion_matrix(test_y,y_pred)













from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
clf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=123)
scores = cross_val_score(clf, data_one_hot.drop('y',axis=1), data_one_hot['y'],cv=5)
print(scores)

from sklearn import metrics
X = data_one_hot.drop('y',axis=1)
y = data_one_hot['y']
from sklearn.linear_model import LogisticRegression
# model = LogisticRegression(penalty='l2')
# model.fit(X, y)

model = RandomForestClassifier(n_estimators=8)
model.fit(X, y)

y_pred = model.predict(X)
y_predprob = model.predict_proba(X)[:,1]
print("Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred))
print("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))


from sklearn import ensemble
clf = ensemble.GradientBoostingRegressor()
gbdt_model = clf.fit(data_one_hot.drop('y',axis=1), data_one_hot['y'])  # Training model

# from sklearn import metrics
# from sklearn.ensemble import GradientBoostingClassifier
# gbm0 = GradientBoostingClassifier(random_state=10)
X = data_one_hot.drop('y',axis=1)
y = data_one_hot['y']
# gbm0.fit(X,y)
# y_pred = gbm0.predict(X)
# y_predprob = gbm0.predict_proba(X)[:,1]
# print("Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred))
# print("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))

from sklearn.svm import SVC
model = SVC(kernel='rbf', probability=True)
model.fit(X, y)
y_pred = model.predict(X)
y_predprob = model.predict_proba(X)[:,1]
print("Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred))
print("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))


# # SVM Classifier
# def svm_classifier(train_x, train_y):
#     from sklearn.svm import SVC
#     model = SVC(kernel='rbf', probability=True)
#     model.fit(train_x, train_y)
#     return model