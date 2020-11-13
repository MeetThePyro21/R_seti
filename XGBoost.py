#!/usr/bin/env python
# coding: utf-8
XGBoost — библиотека, реализующая методы градиентного бустинга. Ее производительность сделала данную библиотеку одним из лидеров в области машинного обучения.
# In[1]:


import numpy
import xgboost
# from sklearn import cross_validation
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


df = numpy.loadtxt('pima-indians-diabetes.csv', delimiter=",")


# In[3]:


# split data into X and Y
X = df[:,0:8]
Y = df[:,8]

разделить данные Х и У на обучающий и тестовый наборы данных. обучающий комплект будет использоваться для создания модели XGBoost, а тестовый набор будет использоваться, для того, чтобы сделать прогнозы, по которым мы можем оценить качество модели.
# In[4]:


# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

Модель XGBoost для классификации называется XGBClassifier. Мы можем создать ее и обучить с помощью тренировочного набора данных. Эта модель имеет функцию fit() для тренировки модели.
# In[5]:


# fit model no training data
model = xgboost.XGBClassifier()
model.fit(X_train, y_train)

Теперь мы можем делать прогнозы, применив нашу обученную модель на тестовом наборе данных.

Для того, чтобы делать прогнозы мы используем scikit-learn функцию model.predict().
Это бинарная задача классификации, каждое предсказание является вероятностью принадлежности к первому классу. Поэтому мы можем легко преобразовать их в значения двоичных классов путем округления до 0 или 1.
# In[6]:


# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

мы можем оценить качество предсказаний, сравнивая их с реальными значениями. Для этого мы будем использовать встроенную в scikit-learn функцию accuracy_score().
# In[7]:


# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

https://www.machinelearningmastery.ru/xgboost-python-mini-course/
на будущее
# ## на датасете iris

# In[8]:


from sklearn import datasets
import xgboost as xgb

iris = datasets.load_iris()
X = iris.data
y = iris.target


# In[9]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

Чтобы XGBoost мог использовать наши данные, нам нужно преобразовать их в определенный формат, который может обрабатывать XGBoost. Этот формат называетсяDMatrix, Преобразовать простой массив данных в формат DMatrix очень просто.
# In[10]:


D_train = xgb.DMatrix(X_train, label=Y_train)
D_test = xgb.DMatrix(X_test, label=Y_test)

Теперь, когда все наши данные загружены, мы можем определить параметры нашего ансамбля повышения градиента. Мы настроили некоторые из наиболее важных ниже, чтобы начать нас. Для более сложных задач и моделей полный список возможных параметров доступен на официальном сайте: "https://xgboost.readthedocs.io/en/latest/parameter.html"
# In[11]:


param = {
    'eta': 0.3, 
    'max_depth': 3,  
    'objective': 'multi:softprob',  
    'num_class': 3} 

steps = 20  # The number of training iterations

Простейшими параметрами являются:
- etc (learning rate. Уменьшение размера шага, используемое в обновлении, чтобы предотвратить переоснащение. После каждого шага повышения мы можем напрямую получить веса новых функций, а eta уменьшает веса функций, чтобы сделать процесс повышения более консервативным.
- Максимальная глубина(максимальная глубина обучаемых деревьев решений),
- задача(используемая функция потерь),
- num_class(количество классов в наборе данных).Согласно нашей теории, Gradient Boosting включает в себя последовательное создание и добавление деревьев решений в модель ансамбля. Новые деревья создаются для исправления остаточных ошибок в прогнозах существующего ансамбля.

Из-за природы ансамбля, то есть наличие нескольких моделей, объединенных в очень сложную модель, делает эту технику склонной к переоснащению. расчетное время прибытия параметр дает нам возможность предотвратить это переоснащение

Эту можно рассматривать более интуитивно как скорость обучения, Вместо того, чтобы просто добавлять прогнозы новых деревьев в ансамбль с полным весом, эта сумма будет умножаться на остатки, добавляемые для уменьшения их веса. Это эффективно уменьшает сложность общей модели.

Обычно небольшие значения находятся в диапазоне от 0,1 до 0,3. Меньший вес этих остатков по-прежнему поможет нам подготовить мощную модель, но не позволит этой модели уйти в глубокую сложность, где, вероятно, произойдет переоснащение.
# In[12]:


# обучаем модель
model = xgb.train(param, D_train, steps)

Давайте теперь проведем оценку Опять же, процесс очень похож на учебные модели в Scikit Learn:
# In[13]:


import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score

preds = model.predict(D_test)
best_preds = np.asarray([np.argmax(line) for line in preds])

print("Precision = {}".format(precision_score(Y_test, best_preds, average='macro')))
print("Recall = {}".format(recall_score(Y_test, best_preds, average='macro')))
print("Accuracy = {}".format(accuracy_score(Y_test, best_preds)))

Точность 93,3%
# In[ ]:




