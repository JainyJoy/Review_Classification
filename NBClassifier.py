#!/usr/bin/env python
# coding: utf-8

# 
# # Naive Bayes Text Classifier for Amazon Mobile Phone Reviews

# This data set contains 400 thousand reviews of unlocked mobile phones sold on Amazon.com.
# ATTRIBUTE LIST
# 1.Product Title
# 2.Brand
# 3.Price
# 4.Rating
# 5.Review text
# 6.Number of people who found the review helpful

# In[31]:


#Importing packages
import numpy as np
import pandas as pd


# In[32]:


#Reading dataset using pandas
df=pd.read_csv("C://Users//Butterfly//Documents//Data//Amazon_Unlocked_Mobile.csv")


# In[33]:


#Getting idea about the dataset
print(df.info())
print(df.describe())
print(pd.isnull(df).sum())


# In[34]:


#Removing unwanted columns from the imported dataset
df1=df.drop(columns=['ProductName','Price','ReviewVotes'])
print(df1.head())


# In[35]:


#Removing null values from the dataset
df_new=df1.dropna(axis=0,how='any')
print(pd.isnull(df_new).sum())
print(df_new.shape)


# In[36]:


#Getting count of the different brands
df_new['BrandName']=df_new['BrandName'].str.lower()
print(df_new['BrandName'].value_counts())


# In[37]:


#Subsetting the dataset into two sets
#one for samsung mobile phones and the other for apple phones
Samsung=df_new[df_new['BrandName']=='samsung']

Apple=df_new[df_new['BrandName']=='apple']


# In[41]:


#Plotting count of ratings for samsung 
import seaborn as sns
sns.countplot(x='Rating',hue='Rating',data=Samsung)


# In[42]:


#Plotting count of ratings for apple 
sns.countplot(x='Rating',hue='Rating',data=Apple)


# In[38]:


#Gathering packages and tools required for preprocessing the text
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


# In[39]:


#Choosing the instances which are having either one or five rating in the samsung brand
Sam_data=Samsung[(Samsung['Rating']==1) | (Samsung['Rating']==5)]
print(Sam_data.shape)
print(Sam_data.describe())


# In[43]:


#Function for preprocessing the text inputs
import string
def text_process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[44]:


#Choosing target and predictor variables
SY=Sam_data['Rating']
SX=Sam_data['Reviews']


# In[45]:


#Converting the predictor data(reviews) into vector and fitting 
count_matrix = CountVectorizer(analyzer=text_process).fit(SX)


# In[47]:


SX=count_matrix.transform(SX)


# In[49]:


print('Shape of Sparse Matrix: ', SX.shape)
print('Amount of Non-Zero occurrences: ', SX.nnz)


# In[50]:


#Splitting data as train and test set and fitting the model
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# In[51]:



X_train, X_test, Y_train, Y_test = train_test_split(SX,SY, test_size=0.2, random_state=101)


# In[56]:


print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# In[59]:


nb = MultinomialNB()
nb.fit(X_train, Y_train)


# In[60]:


preds = nb.predict(X_test)
preds


# In[87]:


post="i hate it"
rate=nb.predict(X_test[1000])
print(rate)


# In[61]:


#Model Evaluation
from sklearn.metrics import confusion_matrix, classification_report


# In[64]:


print(confusion_matrix(Y_test, preds))
print('\n')
print(classification_report(Y_test, preds))


# In[66]:



metrics.accuracy_score(Y_test, preds)


# In[67]:


# examine class distribution
print(Y_test.value_counts())


# In[71]:


#vocabulary for the test data
X_train_tokens=count_matrix.get_feature_names()
len(X_train_tokens)
print(X_train_tokens[0:200])


# In[70]:


print(X_train_tokens[-200:])


# In[72]:


nb.feature_count_.shape
#2 classes and 33965 tokens


# In[90]:


from sklearn.linear_model import LogisticRegression


# In[ ]:





# In[88]:


App_data=Apple[(Apple['Rating']==1) | (Apple['Rating']==5)]
App_data.shape


# In[13]:


n=len(App_data['Reviews'])
n


# In[14]:


import string
def text_process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[17]:


AY=App_data['Rating']
AX=App_data['Reviews']


# In[18]:


count_matrix1 = CountVectorizer(analyzer=text_process).fit(AX)


# In[21]:


AX=count_matrix1.transform(AX)


# In[22]:


print('Shape of Sparse Matrix: ', AX.shape)
print('Amount of Non-Zero occurrences: ', AX.nnz)


# In[23]:


from sklearn.model_selection import train_test_split
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(AX,AY, test_size=0.3, random_state=101)


# In[26]:


nb1 = MultinomialNB()
nb1.fit(X_train1, Y_train1)


# In[28]:


preds1 = nb1.predict(X_test1)


# In[29]:


from sklearn.metrics import confusion_matrix, classification_report


# In[30]:


print(confusion_matrix(Y_test1, preds1))
print('\n')
print(classification_report(Y_test1, preds1))


# In[ ]:




