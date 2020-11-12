#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("/Users/awesome/Desktop/trust.csv")
df.head()

файл trust.csv - это результаты расчетов одного из наших скриптов,

задача кластеризировать эти результаты


BalanceID - номер балансовой группы, для которой мы проводили расчет (он уникальный)
DateMonth - месяц, по которому произведен расчет
DateYear - год, по которому произведен расчет
PercentTransmissionPU - процент передачи показаний от приборов учета
IndexComplianceForecastPresentUnbalance - индекс соответствия спрогонозированного и реального небаланса
TrustIndexPSKfiz - индекс доверия к данным ПСК физ лица
TrustIndexPSKODN- индекс доверия к данным ПСК ОДН
TrustIndexPSKurik - индекс доверия к данным ПСК юридические лица
GUID - технический id (его не трогать)

Целевая задача:
1. сгруппируйте данные в этой таблице, 
условия:
первая группа - 3 индекса доверия (физ, юрид, ОДН) находятся в интервале от 100 до 75% включительно
вторая - от 74 до 50% включительно
третья - от 49 до 1% включительно
четвретая - все что больше 101% и меньше 0%
сохранить результаты
# In[3]:


# первая группа - 3 индекса доверия (физ, юрид, ОДН) находятся в интервале от 100 до 75% включительно
# пусто, таких кейсов нет

group_1 = df[((df['TrustIndexPSKfiz'] >= 75) & (df['TrustIndexPSKfiz'] <= 100)) & ((df['TrustIndexPSKODN'] >= 75) & (df['TrustIndexPSKODN'] <= 100)) & ((df['TrustIndexPSKurik'] >= 75) & (df['TrustIndexPSKurik'] <= 100))]
group_1.to_csv(r'/Users/awesome/Desktop/group_1.csv', index = False)
print(len(group_1))


# In[4]:


# вторая - от 74 до 50% включительно
# пусто, таких кейсов нет 

group_2 = df[((df['TrustIndexPSKfiz'] >= 50) & (df['TrustIndexPSKfiz'] <= 74)) & ((df['TrustIndexPSKODN'] >= 50) & (df['TrustIndexPSKODN'] <= 74)) & ((df['TrustIndexPSKurik'] >= 50) & (df['TrustIndexPSKurik'] <= 74))]
group_2.to_csv(r'/Users/awesome/Desktop/group_2.csv', index = False)
print(len(group_2))


# In[5]:


# третья - от 49 до 1% включительно
# под эти условия подходят 16% кейсов (417 из 2576)

group_3 = df[((df['TrustIndexPSKfiz'] >= 1) & (df['TrustIndexPSKfiz'] <= 49)) & ((df['TrustIndexPSKODN'] >= 1) & (df['TrustIndexPSKODN'] <= 49)) & ((df['TrustIndexPSKurik'] >= 1) & (df['TrustIndexPSKurik'] <= 49))]
group_3.to_csv(r'/Users/awesome/Desktop/group_3.csv', index = False)
print(len(group_3))

Четвретая - все что больше 101% и меньше 0% 
При условии, что 0 включен - подходит 25 кейсов (индексы у физ и одн = 0, индекс у юрид > 101)
Если 0 не включать в условие - будет пусто
# In[6]:


# 0 включен
group_4 = df[((df['TrustIndexPSKfiz'] >= 101) | (df['TrustIndexPSKfiz'] <= 0)) & ((df['TrustIndexPSKODN'] >= 101) | (df['TrustIndexPSKODN'] <= 0)) & ((df['TrustIndexPSKurik'] >= 101) | (df['TrustIndexPSKurik'] <= 0))]
group_4.to_csv(r'/Users/awesome/Desktop/group_4.csv', index = False)
print(len(group_4))


# In[7]:


# 0 не входит
group_4_extra = df[((df['TrustIndexPSKfiz'] >= 101) | (df['TrustIndexPSKfiz'] < 0)) & ((df['TrustIndexPSKODN'] >= 101) | (df['TrustIndexPSKODN'] < 0)) & ((df['TrustIndexPSKurik'] >= 101) | (df['TrustIndexPSKurik'] < 0))]
group_4_extra.to_csv(r'/Users/awesome/Desktop/group_4_extra.csv', index = False)
print(len(group_4_extra))

2. если хотя бы один из индексов доверия превышал 100% дважды за все время наблюдения
(т.е. во всей выборке), то сохранять 
BalanceID,
месяцы (в которые было превышение),
значение индексов (которые повторялись)
# In[8]:


fiz_index = df[df['TrustIndexPSKfiz'] > 100]
fiz_index.shape


# In[9]:


fix_index = fiz_index[['BalanceId', 'DateMonth', 'TrustIndexPSKfiz']]
fix_index.to_csv(r'/Users/awesome/Desktop/fix_index.csv', index = False)


# In[10]:


ODN_index = df[df['TrustIndexPSKODN'] > 100]
ODN_index.shape


# In[11]:


ODN_index = ODN_index[['BalanceId', 'DateMonth', 'TrustIndexPSKODN']]
ODN_index.to_csv(r'/Users/awesome/Desktop/ODN_index.csv', index = False)


# In[12]:


urik_index = df[df['TrustIndexPSKurik'] > 100]
urik_index.shape


# In[13]:


urik_index = urik_index[['BalanceId', 'DateMonth', 'TrustIndexPSKurik']]
urik_index.to_csv(r'/Users/awesome/Desktop/urik_index.csv', index = False)


# In[14]:


print(fix_index, ODN_index, urik_index, sep='\n'+'-'*45+'\n')


# In[15]:


# для проверки дубликатов индексов
# pd.concat(i for _, i in ODN.groupby("TrustIndexPSKODN") if len(i) > 1)
# pd.concat(i for _, i in fiz.groupby("TrustIndexPSKfiz") if len(i) > 1)
# pd.concat(i for _, i in urik.groupby("TrustIndexPSKurik") if len(i) > 1)


# In[ ]:




