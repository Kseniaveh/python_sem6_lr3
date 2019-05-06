%matplotlib inline

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np

"""
Загрузка исходных данных, добавление имен столбцов 0 и 1
"""
df = pd.read_csv('data.csv', header = None) 
x, y = df[0], df[1] 
print(df)

"""
Строем линейную функцию по двум точка, также
можно использовать линейную функцию y = 2x -10
"""
x1,y1 = [0,22.5],[0,25]

"""
Реализация алгоритма градиентного спуска
Нахождение theta0 и theta1
"""
def gradient_descent(X, Y, koef, n):
    l = len(x)
    theta0, theta1 = 0, 0
    for i in range(n):
        sum1 = 0
        for i in range(l):
            sum1 += theta0 + theta1 * x[i] - y[i]
        res1 = theta0 - koef * (1 / l) * sum1

        sum2 = 0
        for i in range(l):
            sum2 += (theta0 + theta1 * x[i] - y[i]) * x[i]
        res2 = theta1 - koef * (1 / l) * sum2

        theta0, theta1 = res1, res2

    return theta0, theta1

"""
График с коэффицентами, найденными по градиентному спуску
"""
x2 = [1, 25]
y2 = [0, 0]
t0, t1 = gradient_descent(x, y, 0.01, len(x))
y2[0] = t0 + x2[0] * t1
y2[1] = t0 + x2[1] * t1
plt.plot(x2, y2, 'r')

"""
График с коэффциентами, найденными с помощью метода polyfit
"""
numpy_x = np.array(x)
numpy_y = np.array(y)
numpy_t1, numpy_t0 = np.polyfit(numpy_x, numpy_y, 1)

num_y1 = [0, 0]
num_y1[0] = numpy_t0 + x1[0] * numpy_t1
num_y1[1] = numpy_t0 + x1[1] * numpy_t1
plt.plot(x1, num_y1, 'b')

print(numpy_t0, numpy_t1)

"""
Визуализация данных
"""
fig = plt.plot(x, y, 'g*') 
fig1 = plt.plot(x1, y1, 'y') 

"""
Сохранение изображения с указанными параметрами 
"""
#plt.savefig('plot.png', format='png', dpi=100)
