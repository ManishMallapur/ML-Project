#!/usr/bin/env python
# coding: utf-8

#    ### Importing Required Libraries

# In[1]:


import csv,os,re,sys,codecs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib,  statistics
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from collections import Counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn import metrics
import warnings


# In[2]:


warnings.filterwarnings('ignore')


# ### Importing the dataset

# In[3]:


data = pd.read_csv('training_data.csv')
labels = pd.read_csv('training_data_targets.csv', names=['y']) 
test = pd.read_csv('test_data.csv')
class_names = ['P', 'H']
print(data.shape)
print(labels.shape)
print(test.shape)


# ### Data Preprocessing

# In[4]:


merged_dataset = data.append(test)
print(merged_dataset.shape)


# In[5]:


scaler = MinMaxScaler()
scaled_merged_dataset = scaler.fit_transform(merged_dataset)
scaled_data = scaled_merged_dataset[0:156,:]
scaled_test = scaled_merged_dataset[156:174,:]
pca = PCA(n_components=0.8)
pca_merged_dataset = pca.fit_transform(scaled_merged_dataset)
pca_data = pca_merged_dataset[0:156,:]
pca_test = pca_merged_dataset[156:174,:]
selector = SelectKBest(chi2, k=50)
print(labels.shape)
selector.fit(scaled_data, labels)
kbest_merged_dataset = selector.transform(scaled_merged_dataset)
kbest_data = kbest_merged_dataset[0:156, :]
kbest_test = kbest_merged_dataset[156:174, :]


# In[6]:


print(pca_data.shape, pca_test.shape)


# In[7]:


training_data, validation_data, training_cat, validation_cat = train_test_split(data, labels,test_size=0.1, random_state=42,stratify=labels)
scaled_training_data, scaled_validation_data, scaled_training_cat, scaled_validation_cat = train_test_split(scaled_data, labels,test_size=0.1, random_state=42,stratify=labels)
pca_training_data, pca_validation_data, pca_training_cat, pca_validation_cat = train_test_split(pca_data, labels,test_size=0.1, random_state=42,stratify=labels)
kbest_training_data, kbest_validation_data, kbest_training_cat, kbest_validation_cat = train_test_split(kbest_data, labels,test_size=0.1, random_state=42,stratify=labels)


# # Training Different Models:

# ## Logistic Regression:

# ### Using Data with no preprocessing

# In[8]:


clf = LogisticRegression()
clf_params = [{'penalty': ['l2','l2'],'C':[0.001,.009,0.01,.09,1,5,10,25]}]
grid_clf = GridSearchCV(clf, clf_params, cv = StratifiedKFold(10), scoring='f1_macro', n_jobs=-1)
grid_clf.fit(training_data, training_cat)
predicted_clf = grid_clf.predict(validation_data)
print('The best parameters: ', grid_clf.best_params_,'\n')
print(classification_report(validation_cat, predicted_clf, target_names=class_names))
pr=precision_score(validation_cat, predicted_clf, average='macro') 
print ('\n Precision:\t'+str(pr))
rl=recall_score(validation_cat, predicted_clf, average='macro') 
print ('\n Recall:\t'+str(rl))
fm=f1_score(validation_cat, predicted_clf, average='macro') 
print ('\n F1-Score:\t'+str(fm))

print("\n CONFUSION MATRIX: \n")
print(metrics.confusion_matrix(validation_cat, predicted_clf))


# ### Using MinMaxScaler

# In[9]:


clf = LogisticRegression()
clf_params = [{'penalty': ['l1', 'l2'],'C':[0.001,.009,0.01,.09,1,5,10,25]}]
grid_clf = GridSearchCV(clf, clf_params, cv = StratifiedKFold(10), scoring='f1_macro', n_jobs=-1)
grid_clf.fit(scaled_training_data, training_cat)
predicted_clf = grid_clf.predict(scaled_validation_data)
print('The best parameters:', grid_clf.best_params_,'\n')
print(classification_report(scaled_validation_cat, predicted_clf, target_names=class_names))
pr=precision_score(scaled_validation_cat, predicted_clf, average='macro') 
print ('\n Precision:\t'+str(pr))         
rl=recall_score(scaled_validation_cat, predicted_clf, average='macro') 
print ('\n Recall:\t'+str(rl))
fm=f1_score(scaled_validation_cat, predicted_clf, average='macro') 
print ('\n F1-Score:\t'+str(fm))

print("\n CONFUSION MATRIX: \n")
print(metrics.confusion_matrix(scaled_validation_cat, predicted_clf))


# ### Using PCA

# In[10]:


clf = LogisticRegression()
clf_params = [{'penalty': ['l1', 'l2'],'C':[0.001,.009,0.01,.09,1,5,10,25]}]
grid_clf = GridSearchCV(clf, clf_params, cv = StratifiedKFold(10), scoring='f1_macro', n_jobs=-1)
grid_clf.fit(pca_training_data, training_cat)
predicted_clf = grid_clf.predict(pca_validation_data)
print('The best parameters: ', grid_clf.best_params_,'\n')
print(classification_report(pca_validation_cat, predicted_clf, target_names=class_names))
pr=precision_score(pca_validation_cat, predicted_clf, average='macro') 
print ('\n Precision:\t'+str(pr))         
rl=recall_score(pca_validation_cat, predicted_clf, average='macro') 
print ('\n Recall:\t'+str(rl))
fm=f1_score(pca_validation_cat, predicted_clf, average='macro') 
print ('\n F1-Score:\t'+str(fm))

print("\n CONFUSION MATRIX: \n")
print(metrics.confusion_matrix(pca_validation_cat, predicted_clf))


# ### Using SelectKBest

# In[11]:


clf = LogisticRegression()
clf_params = [{'penalty': ['l1', 'l2'],'C':[0.001,.009,0.01,.09,1,5,10,25]}]
grid_clf = GridSearchCV(clf, clf_params, cv = StratifiedKFold(10), scoring='f1_macro', n_jobs=-1)
grid_clf.fit(kbest_training_data, training_cat)
predicted_clf = grid_clf.predict(kbest_validation_data)
print('The best parameters: ', grid_clf.best_params_,'\n')
print(classification_report(kbest_validation_cat, predicted_clf, target_names=class_names))
pr=precision_score(kbest_validation_cat, predicted_clf, average='macro') 
print ('\n Precision:\t'+str(pr))         
rl=recall_score(kbest_validation_cat, predicted_clf, average='macro') 
print ('\n Recall:\t'+str(rl))
fm=f1_score(kbest_validation_cat, predicted_clf, average='macro') 
print ('\n F1-Score:\t'+str(fm))

print("\n CONFUSION MATRIX: \n")
print(metrics.confusion_matrix(kbest_validation_cat, predicted_clf))


# ## Support Vector Classifier:

# ### No preprocessing

# In[12]:


clf = svm.SVC(class_weight='balanced',probability=True)  
clf_params = {'C':(0.001,.009,0.01,.09,1,5,10,25),'kernel':('linear','rbf','polynomial'),}
grid_clf = GridSearchCV(clf, clf_params, cv = StratifiedKFold(10), scoring='f1_macro', n_jobs=-1)
grid_clf.fit(training_data, training_cat)
predicted_clf = grid_clf.predict(validation_data)
print('The best parameters: ', grid_clf.best_params_,'\n')
print(classification_report(validation_cat, predicted_clf, target_names=class_names))
pr=precision_score(validation_cat, predicted_clf, average='macro') 
print ('\n Precision:\t'+str(pr))         
rl=recall_score(validation_cat, predicted_clf, average='macro') 
print ('\n Recall:\t'+str(rl))
fm=f1_score(validation_cat, predicted_clf, average='macro') 
print ('\n F1-Score:\t'+str(fm))

print("\n CONFUSION MATRIX: \n")
print(metrics.confusion_matrix(validation_cat, predicted_clf))


# ### MinMaxScaler

# In[13]:


clf = svm.SVC(class_weight='balanced',probability=True)  
clf_params = {'C':(0.001,.009,0.01,.09,1,5,10,25),'kernel':('linear','rbf','polynomial'),}
grid_clf = GridSearchCV(clf, clf_params, cv = StratifiedKFold(10), scoring='f1_macro', n_jobs=-1)
grid_clf.fit(scaled_training_data, training_cat)
predicted_clf = grid_clf.predict(scaled_validation_data)
print('The best parameters:', grid_clf.best_params_,'\n')
print(classification_report(scaled_validation_cat, predicted_clf, target_names=class_names))
pr=precision_score(scaled_validation_cat, predicted_clf, average='macro') 
print ('\n Precision:\t'+str(pr))         
rl=recall_score(scaled_validation_cat, predicted_clf, average='macro') 
print ('\n Recall:\t'+str(rl))
fm=f1_score(scaled_validation_cat, predicted_clf, average='macro') 
print ('\n F1-Score:\t'+str(fm))

print("\n CONFUSION MATRIX: \n")
print(metrics.confusion_matrix(scaled_validation_cat, predicted_clf))


# ### PCA

# In[14]:


clf = svm.SVC(class_weight='balanced',probability=True)  
clf_params = {'C':(0.001,.009,0.01,.09,1,5,10,25),'kernel':('linear','rbf','polynomial'),}
grid_clf = GridSearchCV(clf, clf_params, cv = StratifiedKFold(10), scoring='f1_macro', n_jobs=-1)
grid_clf.fit(pca_training_data, training_cat)
predicted_clf = grid_clf.predict(pca_validation_data)
print('The best parameters: ', grid_clf.best_params_,'\n')
print(classification_report(pca_validation_cat, predicted_clf, target_names=class_names))
pr=precision_score(pca_validation_cat, predicted_clf, average='macro') 
print ('\n Precision:\t'+str(pr))         
rl=recall_score(pca_validation_cat, predicted_clf, average='macro') 
print ('\n Recall:\t'+str(rl))
fm=f1_score(pca_validation_cat, predicted_clf, average='macro') 
print ('\n F1-Score:\t'+str(fm))

print("\n CONFUSION MATRIX: \n")
print(metrics.confusion_matrix(pca_validation_cat, predicted_clf))


# ### SelectKBest

# In[15]:


clf = svm.SVC(class_weight='balanced',probability=True)  
clf_params = {'C':(0.001,.009,0.01,.09,1,5,10,25),'kernel':('linear','rbf','polynomial'),}
grid_clf = GridSearchCV(clf, clf_params, cv = StratifiedKFold(10), scoring='f1_macro', n_jobs=-1)
grid_clf.fit(kbest_training_data, training_cat)
predicted_clf = grid_clf.predict(kbest_validation_data)
print('The best parameters: ', grid_clf.best_params_,'\n')
print(classification_report(kbest_validation_cat, predicted_clf, target_names=class_names))
pr=precision_score(kbest_validation_cat, predicted_clf, average='macro') 
print ('\n Precision:\t'+str(pr))         
rl=recall_score(kbest_validation_cat, predicted_clf, average='macro') 
print ('\n Recall:\t'+str(rl))
fm=f1_score(kbest_validation_cat, predicted_clf, average='macro') 
print ('\n F1-Score:\t'+str(fm))

print("\n CONFUSION MATRIX: \n")
print(metrics.confusion_matrix(kbest_validation_cat, predicted_clf))


# ## Random Forest Classifier:

# ### No Preprocessing

# In[16]:


clf = RandomForestClassifier(max_features=None,class_weight='balanced')
clf_params = {'criterion':('entropy','gini'),'n_estimators':(30,50,100),'max_depth':(10,20,30,50,100,200),}
grid_clf = GridSearchCV(clf, clf_params, cv = StratifiedKFold(10), scoring='f1_macro', n_jobs=-1)
grid_clf.fit(training_data, training_cat)
predicted_clf = grid_clf.predict(validation_data)
print('The best parameters: ', grid_clf.best_params_,'\n')
print(classification_report(validation_cat, predicted_clf, target_names=class_names))
pr=precision_score(validation_cat, predicted_clf, average='macro') 
print ('\n Precision:\t'+str(pr))         
rl=recall_score(validation_cat, predicted_clf, average='macro') 
print ('\n Recall:\t'+str(rl))
fm=f1_score(validation_cat, predicted_clf, average='macro') 
print ('\n F1-Score:\t'+str(fm))

print("\n CONFUSION MATRIX: \n")
print(metrics.confusion_matrix(validation_cat, predicted_clf))


# ### MinMaxScaler

# In[17]:


clf = RandomForestClassifier(max_features=None,class_weight='balanced')
clf_params = {'criterion':('entropy','gini'),'n_estimators':(30,50,100),'max_depth':(10,20,30,50,100,200),}
grid_clf = GridSearchCV(clf, clf_params, cv = StratifiedKFold(10), scoring='f1_macro', n_jobs=-1)
grid_clf.fit(scaled_training_data, training_cat)
predicted_clf = grid_clf.predict(scaled_validation_data)
print('The best parameters:', grid_clf.best_params_,'\n')
print(classification_report(scaled_validation_cat, predicted_clf, target_names=class_names))
pr=precision_score(scaled_validation_cat, predicted_clf, average='macro') 
print ('\n Precision:\t'+str(pr))         
rl=recall_score(scaled_validation_cat, predicted_clf, average='macro') 
print ('\n Recall:\t'+str(rl))
fm=f1_score(scaled_validation_cat, predicted_clf, average='macro') 
print ('\n F1-Score:\t'+str(fm))

print("\n CONFUSION MATRIX: \n")
print(metrics.confusion_matrix(scaled_validation_cat, predicted_clf))


# ### PCA

# In[18]:


clf = RandomForestClassifier(max_features=None,class_weight='balanced')
clf_params = {'criterion':('entropy','gini'),'n_estimators':(30,50,100),'max_depth':(10,20,30,50,100,200),}
grid_clf = GridSearchCV(clf, clf_params, cv = StratifiedKFold(10), scoring='f1_macro', n_jobs=-1)
grid_clf.fit(pca_training_data, training_cat)
predicted_clf = grid_clf.predict(pca_validation_data)
print('The best parameters: ', grid_clf.best_params_,'\n')
print(classification_report(pca_validation_cat, predicted_clf, target_names=class_names))
pr=precision_score(pca_validation_cat, predicted_clf, average='macro') 
print ('\n Precision:\t'+str(pr))         
rl=recall_score(pca_validation_cat, predicted_clf, average='macro') 
print ('\n Recall:\t'+str(rl))
fm=f1_score(pca_validation_cat, predicted_clf, average='macro') 
print ('\n F1-Score:\t'+str(fm))

print("\n CONFUSION MATRIX: \n")
print(metrics.confusion_matrix(pca_validation_cat, predicted_clf))


# ### SelectKBest

# In[19]:


clf = RandomForestClassifier(max_features=None,class_weight='balanced')
clf_params = {'criterion':('entropy','gini'),'n_estimators':(30,50,100),'max_depth':(10,20,30,50,100,200),}
grid_clf = GridSearchCV(clf, clf_params, cv = StratifiedKFold(10), scoring='f1_macro', n_jobs=-1)
grid_clf.fit(kbest_training_data, training_cat)
predicted_clf = grid_clf.predict(kbest_validation_data)
print('The best parameters: ', grid_clf.best_params_,'\n')
print(classification_report(kbest_validation_cat, predicted_clf, target_names=class_names))
pr=precision_score(kbest_validation_cat, predicted_clf, average='macro') 
print ('\n Precision:\t'+str(pr))         
rl=recall_score(kbest_validation_cat, predicted_clf, average='macro') 
print ('\n Recall:\t'+str(rl))
fm=f1_score(kbest_validation_cat, predicted_clf, average='macro') 
print ('\n F1-Score:\t'+str(fm))

print("\n CONFUSION MATRIX: \n")
print(metrics.confusion_matrix(kbest_validation_cat, predicted_clf))


# ## Decision Tree:

# ### No Preprocessing

# In[20]:


clf = DecisionTreeClassifier(random_state=40) 
clf_params = {'criterion':('gini', 'entropy'), 'max_features':('auto', 'sqrt', 'log2'),'max_depth':(10,40,45,60),'ccp_alpha':(0.009,0.01,0.05,0.1),} 
grid_clf = GridSearchCV(clf, clf_params, cv = StratifiedKFold(10), scoring='f1_macro', n_jobs=-1)
grid_clf.fit(training_data, training_cat)
predicted_clf = grid_clf.predict(validation_data)
print('The best parameters: ', grid_clf.best_params_,'\n')
print(classification_report(validation_cat, predicted_clf, target_names=class_names))
pr=precision_score(validation_cat, predicted_clf, average='macro') 
print ('\n Precision:\t'+str(pr))         
rl=recall_score(validation_cat, predicted_clf, average='macro') 
print ('\n Recall:\t'+str(rl))
fm=f1_score(validation_cat, predicted_clf, average='macro') 
print ('\n F1-Score:\t'+str(fm))

print("\n CONFUSION MATRIX: \n")
print(metrics.confusion_matrix(validation_cat, predicted_clf))


# ### MinMaxScaler

# In[21]:


clf = DecisionTreeClassifier(random_state=40) 
clf_params = {'criterion':('gini', 'entropy'), 'max_features':('auto', 'sqrt', 'log2'),'max_depth':(10,40,45,60),'ccp_alpha':(0.009,0.01,0.05,0.1),} 
grid_clf = GridSearchCV(clf, clf_params, cv = StratifiedKFold(10), scoring='f1_macro', n_jobs=-1)
grid_clf.fit(scaled_training_data, training_cat)
predicted_clf = grid_clf.predict(scaled_validation_data)
print('The best parameters:', grid_clf.best_params_,'\n')
print(classification_report(scaled_validation_cat, predicted_clf, target_names=class_names))
pr=precision_score(scaled_validation_cat, predicted_clf, average='macro') 
print ('\n Precision:\t'+str(pr))         
rl=recall_score(scaled_validation_cat, predicted_clf, average='macro') 
print ('\n Recall:\t'+str(rl))
fm=f1_score(scaled_validation_cat, predicted_clf, average='macro') 
print ('\n F1-Score:\t'+str(fm))

print("\n CONFUSION MATRIX: \n")
print(metrics.confusion_matrix(scaled_validation_cat, predicted_clf))


# ### PCA

# In[22]:


clf = DecisionTreeClassifier(random_state=40) 
clf_params = {'criterion':('gini', 'entropy'), 'max_features':('auto', 'sqrt', 'log2'),'max_depth':(10,40,45,60),'ccp_alpha':(0.009,0.01,0.05,0.1),} 
grid_clf = GridSearchCV(clf, clf_params, cv = StratifiedKFold(10), scoring='f1_macro', n_jobs=-1)
grid_clf.fit(pca_training_data, training_cat)
predicted_clf = grid_clf.predict(pca_validation_data)
print('The best parameters: ', grid_clf.best_params_,'\n')
print(classification_report(pca_validation_cat, predicted_clf, target_names=class_names))
pr=precision_score(pca_validation_cat, predicted_clf, average='macro') 
print ('\n Precision:\t'+str(pr))         
rl=recall_score(pca_validation_cat, predicted_clf, average='macro') 
print ('\n Recall:\t'+str(rl))
fm=f1_score(pca_validation_cat, predicted_clf, average='macro') 
print ('\n F1-Score:\t'+str(fm))

print("\n CONFUSION MATRIX: \n")
print(metrics.confusion_matrix(pca_validation_cat, predicted_clf))


# ### SelectKBest

# In[23]:


clf = DecisionTreeClassifier(random_state=40) 
clf_params = {'criterion':('gini', 'entropy'), 'max_features':('auto', 'sqrt', 'log2'),'max_depth':(10,40,45,60),'ccp_alpha':(0.009,0.01,0.05,0.1),} 
grid_clf = GridSearchCV(clf, clf_params, cv = StratifiedKFold(10), scoring='f1_macro', n_jobs=-1)
grid_clf.fit(kbest_training_data, training_cat)
predicted_clf = grid_clf.predict(kbest_validation_data)
print('The best parameters: ', grid_clf.best_params_,'\n')
print(classification_report(kbest_validation_cat, predicted_clf, target_names=class_names))
pr=precision_score(kbest_validation_cat, predicted_clf, average='macro') 
print ('\n Precision:\t'+str(pr))
rl=recall_score(kbest_validation_cat, predicted_clf, average='macro') 
print ('\n Recall:\t'+str(rl))
fm=f1_score(kbest_validation_cat, predicted_clf, average='macro') 
print ('\n F1-Score:\t'+str(fm))

print("\n CONFUSION MATRIX: \n")
print(metrics.confusion_matrix(kbest_validation_cat, predicted_clf))


# ## Adaptive Boosting:

# ### No Preprocessing

# In[24]:


be1 = svm.SVC(kernel='linear', class_weight='balanced',probability=True, C=0.09)              
be2 = LogisticRegression(solver='liblinear',class_weight='balanced', C=1, penalty='l2') 
be3 = DecisionTreeClassifier(ccp_alpha = 0.05, criterion = 'gini', max_depth = 10, max_features= 'auto')
be4 = RandomForestClassifier(max_features=None,class_weight='balanced',criterion = 'entropy', max_depth= 20, n_estimators= 30 )
be5 = KNeighborsClassifier(n_neighbors= 5, weights= 'uniform')
clf = AdaBoostClassifier(algorithm='SAMME.R',n_estimators=10)
clf_params = {'base_estimator':(be1,be2,be3,be4,be5),'random_state':(0,10),}
grid_clf = GridSearchCV(clf, clf_params, cv = StratifiedKFold(10), scoring='f1_macro', n_jobs=-1)
grid_clf.fit(training_data, training_cat)
predicted_clf = grid_clf.predict(validation_data)
print('The best parameters: ', grid_clf.best_params_,'\n')
print(classification_report(validation_cat, predicted_clf, target_names=class_names))
pr=precision_score(validation_cat, predicted_clf, average='macro') 
print ('\n Precision:\t'+str(pr))         
rl=recall_score(validation_cat, predicted_clf, average='macro') 
print ('\n Recall:\t'+str(rl))
fm=f1_score(validation_cat, predicted_clf, average='macro') 
print ('\n F1-Score:\t'+str(fm))

print("\n CONFUSION MATRIX: \n")
print(metrics.confusion_matrix(validation_cat, predicted_clf))


# ### MinMaxScaler

# In[25]:


be1 = svm.SVC(kernel='linear', class_weight='balanced',probability=True, C=0.09)              
be2 = LogisticRegression(solver='liblinear',class_weight='balanced', C=1, penalty='l2') 
be3 = DecisionTreeClassifier(ccp_alpha = 0.05, criterion = 'gini', max_depth = 10, max_features= 'auto')
be4 = RandomForestClassifier(max_features=None,class_weight='balanced',criterion = 'entropy', max_depth= 20, n_estimators= 30 )
be5 = KNeighborsClassifier(n_neighbors= 5, weights= 'uniform')
clf = AdaBoostClassifier(algorithm='SAMME.R',n_estimators=10)
clf_params = {'base_estimator':(be1,be2,be3,be4,be5),'random_state':(0,10),}
grid_clf = GridSearchCV(clf, clf_params, cv = StratifiedKFold(10), scoring='f1_macro', n_jobs=-1)
grid_clf.fit(scaled_training_data, training_cat)
predicted_clf = grid_clf.predict(scaled_validation_data)
print('The best parameters:', grid_clf.best_params_,'\n')
print(classification_report(scaled_validation_cat, predicted_clf, target_names=class_names))
pr=precision_score(scaled_validation_cat, predicted_clf, average='macro') 
print ('\n Precision:\t'+str(pr))         
rl=recall_score(scaled_validation_cat, predicted_clf, average='macro') 
print ('\n Recall:\t'+str(rl))
fm=f1_score(scaled_validation_cat, predicted_clf, average='macro') 
print ('\n F1-Score:\t'+str(fm))

print("\n CONFUSION MATRIX: \n")
print(metrics.confusion_matrix(scaled_validation_cat, predicted_clf))


# ### PCA

# In[26]:


be1 = svm.SVC(kernel='linear', class_weight='balanced',probability=True, C=0.09)              
be2 = LogisticRegression(solver='liblinear',class_weight='balanced', C=1, penalty='l2') 
be3 = DecisionTreeClassifier(ccp_alpha = 0.05, criterion = 'gini', max_depth = 10, max_features= 'auto')
be4 = RandomForestClassifier(max_features=None,class_weight='balanced',criterion = 'entropy', max_depth= 20, n_estimators= 30 )
be5 = KNeighborsClassifier(n_neighbors= 5, weights= 'uniform')
clf = AdaBoostClassifier(algorithm='SAMME.R',n_estimators=10)
clf_params = {'base_estimator':(be1,be2,be3,be4,be5),'random_state':(0,10),}
grid_clf = GridSearchCV(clf, clf_params, cv = StratifiedKFold(10), scoring='f1_macro', n_jobs=-1)
grid_clf.fit(pca_training_data, training_cat)
predicted_clf = grid_clf.predict(pca_validation_data)
print('The best parameters: ', grid_clf.best_params_,'\n')
print(classification_report(pca_validation_cat, predicted_clf, target_names=class_names))
pr=precision_score(pca_validation_cat, predicted_clf, average='macro') 
print ('\n Precision:\t'+str(pr))         
rl=recall_score(pca_validation_cat, predicted_clf, average='macro') 
print ('\n Recall:\t'+str(rl))
fm=f1_score(pca_validation_cat, predicted_clf, average='macro') 
print ('\n F1-Score:\t'+str(fm))

print("\n CONFUSION MATRIX: \n")
print(metrics.confusion_matrix(pca_validation_cat, predicted_clf))


# ### SelectKBest

# In[27]:


be1 = svm.SVC(kernel='linear', class_weight='balanced',probability=True, C=0.09)              
be2 = LogisticRegression(solver='liblinear',class_weight='balanced', C=1, penalty='l2') 
be3 = DecisionTreeClassifier(ccp_alpha = 0.05, criterion = 'gini', max_depth = 10, max_features= 'auto')
be4 = RandomForestClassifier(max_features=None,class_weight='balanced',criterion = 'entropy', max_depth= 20, n_estimators= 30 )
be5 = KNeighborsClassifier(n_neighbors= 5, weights= 'uniform')
clf = AdaBoostClassifier(algorithm='SAMME.R',n_estimators=10)
clf_params = {'base_estimator':(be1,be2,be3,be4,be5),'random_state':(0,10),}
grid_clf = GridSearchCV(clf, clf_params, cv = StratifiedKFold(10), scoring='f1_macro', n_jobs=-1)
grid_clf.fit(kbest_training_data, training_cat)
predicted_clf = grid_clf.predict(kbest_validation_data)
print('The best parameters: ', grid_clf.best_params_,'\n')
print(classification_report(kbest_validation_cat, predicted_clf, target_names=class_names))
pr=precision_score(kbest_validation_cat, predicted_clf, average='macro') 
print ('\n Precision:\t'+str(pr))
rl=recall_score(kbest_validation_cat, predicted_clf, average='macro') 
print ('\n Recall:\t'+str(rl))
fm=f1_score(kbest_validation_cat, predicted_clf, average='macro') 
print ('\n F1-Score:\t'+str(fm))

print("\n CONFUSION MATRIX: \n")
print(metrics.confusion_matrix(kbest_validation_cat, predicted_clf))


# ## K Nearest Neighbours:

# ### No Preprocessing

# In[28]:


clf = KNeighborsClassifier()
clf_params = [{'n_neighbors':[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31], 'weights':['uniform', 'distance']}]
grid_clf = GridSearchCV(clf, clf_params, cv = StratifiedKFold(10), scoring='f1_macro', n_jobs=-1)
grid_clf.fit(training_data, training_cat)
predicted_clf = grid_clf.predict(validation_data)
print('The best parameters: ', grid_clf.best_params_,'\n')
print(classification_report(validation_cat, predicted_clf, target_names=class_names))
pr=precision_score(validation_cat, predicted_clf, average='macro') 
print ('\n Precision:\t'+str(pr))         
rl=recall_score(validation_cat, predicted_clf, average='macro') 
print ('\n Recall:\t'+str(rl))
fm=f1_score(validation_cat, predicted_clf, average='macro') 
print ('\n F1-Score:\t'+str(fm))

print("\n CONFUSION MATRIX: \n")
print(metrics.confusion_matrix(validation_cat, predicted_clf))


# ### MinMaxScaler

# In[29]:


clf = KNeighborsClassifier()
clf_params = [{'n_neighbors':[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31], 'weights':['uniform', 'distance']}]
grid_clf = GridSearchCV(clf, clf_params, cv = StratifiedKFold(10), scoring='f1_macro', n_jobs=-1)
grid_clf.fit(scaled_training_data, training_cat)
predicted_clf = grid_clf.predict(scaled_validation_data)
print('The best parameters:', grid_clf.best_params_,'\n')
print(classification_report(scaled_validation_cat, predicted_clf, target_names=class_names))
pr=precision_score(scaled_validation_cat, predicted_clf, average='macro') 
print ('\n Precision:\t'+str(pr))         
rl=recall_score(scaled_validation_cat, predicted_clf, average='macro') 
print ('\n Recall:\t'+str(rl))
fm=f1_score(scaled_validation_cat, predicted_clf, average='macro') 
print ('\n F1-Score:\t'+str(fm))

print("\n CONFUSION MATRIX: \n")
print(metrics.confusion_matrix(scaled_validation_cat, predicted_clf))


# ### PCA

# In[30]:


clf = KNeighborsClassifier()
clf_params = [{'n_neighbors':[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31], 'weights':['uniform', 'distance']}]
grid_clf = GridSearchCV(clf, clf_params, cv = StratifiedKFold(10), scoring='f1_macro', n_jobs=-1)
grid_clf.fit(pca_training_data, training_cat)
predicted_clf = grid_clf.predict(pca_validation_data)
print('The best parameters: ', grid_clf.best_params_,'\n')
print(classification_report(pca_validation_cat, predicted_clf, target_names=class_names))
pr=precision_score(pca_validation_cat, predicted_clf, average='macro') 
print ('\n Precision:\t'+str(pr))         
rl=recall_score(pca_validation_cat, predicted_clf, average='macro') 
print ('\n Recall:\t'+str(rl))
fm=f1_score(pca_validation_cat, predicted_clf, average='macro') 
print ('\n F1-Score:\t'+str(fm))

print("\n CONFUSION MATRIX: \n")
print(metrics.confusion_matrix(pca_validation_cat, predicted_clf))


# ### SelectKBest

# In[31]:


clf = KNeighborsClassifier()
clf_params = [{'n_neighbors':[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31], 'weights':['uniform', 'distance']}]
grid_clf = GridSearchCV(clf, clf_params, cv = StratifiedKFold(10), scoring='f1_macro', n_jobs=-1)
grid_clf.fit(kbest_training_data, training_cat)
predicted_clf = grid_clf.predict(kbest_validation_data)
print('The best parameters: ', grid_clf.best_params_,'\n')
print(classification_report(kbest_validation_cat, predicted_clf, target_names=class_names))
pr=precision_score(kbest_validation_cat, predicted_clf, average='macro') 
print ('\n Precision:\t'+str(pr))
rl=recall_score(kbest_validation_cat, predicted_clf, average='macro') 
print ('\n Recall:\t'+str(rl))
fm=f1_score(kbest_validation_cat, predicted_clf, average='macro') 
print ('\n F1-Score:\t'+str(fm))

print("\n CONFUSION MATRIX: \n")
print(metrics.confusion_matrix(kbest_validation_cat, predicted_clf))


# # Predicting the labels for the given test data:

# In[34]:


# I will use the model and the preprocessing that gave the best F1-Score to predict the test data:
clf = KNeighborsClassifier()
clf_params = [{'n_neighbors':[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31], 'weights':['uniform', 'distance']}]
grid_clf = GridSearchCV(clf, clf_params, cv = StratifiedKFold(10), scoring='f1_macro', n_jobs=-1)
grid_clf.fit(kbest_training_data, training_cat)
predicted = grid_clf.predict(kbest_test)
print(predicted)


# In[ ]:




