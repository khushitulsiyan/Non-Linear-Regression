#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


df = pd.read_csv("housing.csv")
df.head(10)


# In[7]:


plt.figure(figsize=(8,5))
x_data, y_data = (df["total_rooms"].values, df["population"].values)
plt.plot(x_data, y_data, 'ro')
plt.ylabel('Total Rooms')
plt.xlabel('Population')
plt.show()


# In[8]:


X = np.arange(-5.0, 5.0, 0.1)
Y = 1.0 / (1.0 + np.exp(-X))

plt.plot(X,Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()


# In[9]:


def sigmoid(x, Beta_1, Beta_2):
    y = 1/(1+np.exp(-Beta_1*(x-Beta_2)))
    return y


# In[10]:


beta_1 = 0.10
beta_2 = 1990.0

#Logistic function
Y_pred = sigmoid(x_data, beta_1, beta_2)

#plot initial prediction against datapoints

plt.plot(x_data, Y_pred*15000000000000.)
plt.plot(x_data, y_data, 'ro')


# In[11]:


#lets normalize our data
xdata = x_data/max(x_data)
ydata = y_data/max(y_data)


# In[12]:


from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, xdata, ydata)

#print the final parameters

print("beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))


# In[14]:


x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.xlabel('Total Rooms')
plt.ylabel('Population')
plt.show()


# In[ ]:




