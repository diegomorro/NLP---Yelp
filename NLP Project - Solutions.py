
# coding: utf-8

# In[32]:


import numpy as np
import pandas as pd


# In[33]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[34]:


yelp = pd.read_csv('yelp.csv')


# In[35]:


yelp.head()


# In[36]:


yelp.info()


# In[37]:


yelp.describe()


# In[38]:


yelp['text length'] = yelp['text'].apply(len)


# In[39]:


sns.set_style('white')


# In[40]:


g = sns.FacetGrid(yelp,col='stars')
g.map(plt.hist,'text length')


# In[41]:


sns.boxplot(x='stars',y='text length',data=yelp,palette='rainbow')


# In[42]:


sns.countplot(x='stars',data=yelp,palette='rainbow')


# In[43]:


stars = yelp.groupby('stars').mean()
stars


# In[44]:


stars.corr()


# In[45]:


sns.heatmap(stars.corr(),cmap='coolwarm',annot=True)


# In[46]:


yelp_class = yelp[(yelp.stars==1) | (yelp.stars==5)]


# In[47]:


X = yelp_class['text']
y = yelp_class['stars']


# In[48]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()


# In[49]:


X = cv.fit_transform(X)


# In[50]:


from sklearn.model_selection import train_test_split


# In[51]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)


# In[52]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# In[53]:


nb.fit(X_train,y_train)


# In[54]:


predictions = nb.predict(X_test)


# In[55]:


from sklearn.metrics import confusion_matrix,classification_report


# In[56]:


print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))

