#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Creating random dataset with random function(with SD)
np.random.randn(10)


# In[3]:


#Converting to Pandas DataFrame
df=pd.DataFrame(np.random.randn(10),columns=['data'])


# In[4]:


df


# In[5]:


# Plotting graph
# X- axis - index , Y-axis - Data

df.plot()


# In[6]:


# coverting index to date range
df=pd.DataFrame(np.random.randn(10), columns=['data'],index=pd.date_range('2022/05/15',periods=10))


# In[7]:


df


# In[8]:


df.plot()


# In[9]:


# Expanding figure size
df.plot(figsize=(10,20))


# In[10]:


#Importing Iris dataset
df=pd.read_csv('https://raw.githubusercontent.com/venky14/Machine-Learning-with-Iris-Dataset/master/Iris.csv')


# In[11]:


df.head()


# In[12]:


# Analysis
df.plot(figsize=(20,10))


# In[13]:


# Dropping Id column from dataset as it is disturding the rest of the attributes
df=df.drop('Id',axis=1)


# In[14]:


df.plot(figsize=(20,10))


# In[15]:


df


# In[16]:


# Taken dataset from row 5 to 11 to make the analysis easier
df.iloc[5:11]


# In[17]:


# X -Axis - index , Y-Axis - numerical attributes column
df.iloc[5:11].plot(kind='bar',figsize=(20,10))


# In[18]:


df.iloc[5:11]


# In[19]:


# Slicing the 1st row from the record df.iloc[5:11]
df.iloc[5:11][0:1]


# In[20]:


df.iloc[5:11][0:1].plot(kind='bar',figsize=(20,10))


# In[21]:


# OR - 5th Row
df.iloc[[5]].plot(kind='bar',figsize=(20,10))


# In[22]:


# Plot barh
df.iloc[[5]].plot(kind='barh',figsize=(20,10))


# In[23]:


df.iloc[5:11].plot(kind='barh',figsize=(20,10))


# In[24]:


df


# In[25]:


# Hist
df.plot(kind='hist')
# It tells between 0-1 around 50 datasets are available
# X axis - Value range of attributes and Y - frequency


# In[26]:


df


# In[27]:


# We will take one attribute to make the analysis easier
df['SepalLengthCm'].plot(kind='hist')
# 4-4.7 range/bins/bucket there are around 8 datasets available


# In[28]:


# Switching the axes
df['SepalLengthCm'].plot(kind='hist', orientation='horizontal')


# In[29]:


# Histogram with multiple figures(calling hist as function)
df.hist(figsize=(20,10),color='r',alpha=0.2)
# alpha - intensity of color


# In[30]:


df


# In[31]:


# Box plot --> To identify outlier in dataset
df.plot(kind='box',figsize=(10,20),color={'boxes':'g','whiskers':'r'})
# Q1-Q3=IQR(Inter Quartile Range) - Q1(25%)-Q3(75%), Q2-Median


# In[32]:


# Show vertically
df.plot(kind='box',figsize=(10,20),color={'boxes':'g','whiskers':'r'},vert=False)


# In[33]:


df.describe()


# ![1_etrgPKNszZQ2mX8OK3eE1w.jpeg](attachment:1_etrgPKNszZQ2mX8OK3eE1w.jpeg)

# In[34]:


IQR=6.4-5.1
IQR


# In[35]:


#Min=Q1-1.5*IQR(In describe function we got the approximate value)
Min=5.1-(1.5*1.3)
Min


# In[36]:


#Max=Q1+1.5*IQR(In describe function we got the approximate value)
Max=6.4+(1.5*1.3)
Max


# #### Below minimum and maximum the dataset pertaining are called outliers.
# #### The graph represents SepalWidthCm having outliers
# #### SepalLengthCm( The data are distributed almost equally between Min to Max
# #### PetalLengthCm( Majority of data are pertaining (Q1 & Median)

# In[37]:


#Area Graph - Without Stacked
df.plot(kind='area',figsize=(20,10),alpha=0.4,stacked =False )


# In[38]:


#Area Graph - With Stacked
df.plot(kind='area',figsize=(20,10),alpha=0.4,stacked =True)


# In[39]:


df


# In[40]:


# Scatter Plot( to show relationship between 2 attributes)
df.plot.scatter(x='SepalLengthCm',y='PetalLengthCm')


# #### When SepalLengthCm is increasing PetalLengthCm also increasing

# In[41]:


df.plot.scatter(x='SepalLengthCm',y='PetalWidthCm')


# #### The data are quiet scattered not able to identify relationship

# In[42]:


# Bringing 3 attributes
df.plot.scatter(x='SepalLengthCm',y='PetalLengthCm',c='SepalWidthCm',s=300)


# In[43]:


df.head(1)


# #### X - axis --> SepalLengthCm, Y- axis --> PetalLengthCm, color axis -- >SepalWidthCm
# #### It indicates the intensity of the colors for data points are not same.
# #### If we consider the the 1st record SepalLengthCm =5.1,PetalLengthCm=1.4 and SepalWidthCm 3.5, the color code is according to 3.5 which is plotted in color axis

# In[44]:


# Making Size of the scatter plot datasets wrt PetalWidthCm(Incresing of the size of PetalWidthCm the bubble size increses )
df.plot.scatter(x='SepalLengthCm',y='PetalLengthCm',c='SepalWidthCm',s=df['PetalWidthCm']*500)


# In[45]:


#  hexbin with controll the color wrt SepalWidthCm
df.plot.hexbin(x='SepalLengthCm',y='PetalLengthCm',gridsize=10, C='SepalWidthCm')


# In[46]:


df.iloc[0]


# In[47]:


# Pie Chart(% age of distribution of data)(Took only 1 record from dataset)
df1=df.drop(columns='Species').iloc[0]


# In[48]:


df1.plot(kind='pie')


# ## Plotly

# In[49]:


pip install plotly


# In[50]:


pip install cufflinks


# In[51]:


import cufflinks as cf


# In[52]:


pip install chart_studio


# In[55]:


cf.go_offline()


# In[57]:


df.head()


# In[56]:


df.iplot()


# In[62]:


#ScatterPlot
df.iplot(x='SepalLengthCm',y='PetalLengthCm',mode='markers',kind='scatter')


# In[66]:


df.iplot(mode='markers',kind='scatter',size=5)


# In[68]:


df.iplot(kind='bubble',x='SepalLengthCm',y='PetalLengthCm',size='SepalWidthCm')


# In[70]:


# Scatter Matrix
df.scatter_matrix()


# In[71]:


df


# In[72]:


#3D plot
df.iplot(kind='scatter3d',x='SepalLengthCm',y='SepalWidthCm',z='PetalLengthCm')


# In[73]:


# To analyze in better way we need to take couple of rows
df[0:5].iplot(kind='scatter3d',x='SepalLengthCm',y='SepalWidthCm',z='PetalLengthCm')


# In[74]:


df[0:5]


# In[75]:


# Bubble 3d
df.iplot(kind='bubble3d',x='SepalLengthCm',y='SepalWidthCm',z='PetalLengthCm',size='PetalWidthCm')


# In[ ]:




