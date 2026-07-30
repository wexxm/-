#Конечно же, всё исследование проводилось в среде Jupiter Notebook

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import r2_score

# Загружаем данные
df = pd.read_csv('data/data.csv')

# Ренейм колонок
df.columns = ['date', 'house_age', 'dist_to_metro', 'num_stores', 'latitude', 'longitude', 'price']

# Ознакомление с датасетом
print(df.head())
print(df.info())
print(df.describe())

# Смотрим нуллы
print(df.isnull().sum())

# Строим матрицу корреляций
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Матрица корреляций')
plt.show()

# Смотрим зависимость цены от расстояния до метро
plt.figure(figsize=(8, 6))
sns.scatterplot(x='dist_to_metro', y='price', data=df)
plt.xlabel('Расстояние до метро (метры)')
plt.ylabel('Цена за единицу площади')
plt.title('Цена vs расстояние до метро')
plt.show()

# Смотрим зависимость цены от возраста дома
plt.figure(figsize=(8, 6))
sns.scatterplot(x='house_age', y='price', data=df)
plt.xlabel('Возраст дома (годы)')
plt.ylabel('Цена за единицу площади')
plt.title('Цена vs возраст дома')
plt.show()

# Смотрим зависимость цены от количества магазинов
plt.figure(figsize=(8, 6))
sns.scatterplot(x='num_stores', y='price', data=df)
plt.xlabel('Количество магазинов / кафе')
plt.ylabel('Цена за единицу площади')
plt.title('Цена vs количество магазинов')
plt.show()

# Строим регрессионную модель
X = df[['dist_to_metro', 'house_age', 'num_stores']]
y = df['price']

X_sm = sm.add_constant(X)
model = sm.OLS(y, X_sm).fit()

# Смотрим результаты
print(model.summary())
print(f'R²: {r2_score(y, model.predict(X_sm)):.3f}')

# Проверяем остатки
residuals = model.resid
plt.figure(figsize=(8, 6))
plt.scatter(model.predict(X_sm), residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Предсказанные значения')
plt.ylabel('Остатки')
plt.title('График остатков')
plt.show()
