
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import csv

def sq_error(sq_x, sq_y, f_x=None):
    """
        Вычисление среднеквадратичной ошибки
    """
    squared_error = []
    for i in range(len(sq_x)):
        squared_error.append((f_x(sq_x[i]) - sq_y[i])**2)
    return sum(squared_error)
    
    """
        Считываем данные с файла, добавляем имена для столбцов
    """
df = pd.read_csv('task8.tsv', sep='\t', header=None)

X, Y = df[0], df[1]

x = list(X)
y = list(Y)

"""
    Приравнивание значений nan = 0
"""
for i in range(len(y)):
    if math.isnan(y[i]):
        y[i] = 0
    else:
        y[i] = y[i]
"""
    Отображение координат точек из данных файла
"""
plt.plot(x, y, 'k*')

#print(df)

#Создаем массив из списков 
#array - функция, создающая объект типа ndarray;
#для использования в np.polyfit - работа + возврат массива
numpy_x = np.array(x)
numpy_y = np.array(y)

x1 = list(range(743))

"""
    Использование функции polyfit для подбора кэффициента в зависимости от степени полиномы
"""
th0, th1 = np.polyfit(numpy_x, numpy_y, 1)
th2, th3, th4 = np.polyfit(numpy_x, numpy_y, 2)
th5, th6, th7, th8 = np.polyfit(numpy_x, numpy_y, 3)
th9, th10, th11, th12, th13 = np.polyfit(numpy_x, numpy_y, 4)
th14, th15, th16, th17, th18, th19 = np.polyfit(numpy_x, numpy_y, 5)

f1 = lambda x: th0*x + th1
f2 = lambda x: th2*x**2 + th3*x + th4
f3 = lambda x: th5*x**3 + th6*x**2 + th7*x + th8
f4 = lambda x: th9*x**4 + th10*x**3 + th11*x**2 + th12*x + th13
f5 = lambda x: th14*x**5 + th15*x**4 + th16*x**3 + th17*x**2 + th18*x + th19

result_one = sq_error(x, y, f1)
result_two = sq_error(x, y, f2)
result_three = sq_error(x, y, f3)
result_four = sq_error(x, y, f4)
result_five = sq_error(x, y, f5)

print("Среднее квадратичное отклонение: ", result_one)
print("Среднее квадратичное отклонение: ", result_two)
print("Среднее квадратичное отклонение: ", result_three)
print("Среднее квадратичное отклонение: ", result_four)
print("Среднее квадратичное отклонение: ", result_five)

"""
    Отображении функций в зависимости от степени полиномы от 1 до 5
    poly1d - функция помогает определить полиномиальную функцию (составление многочлена взависимости от степени полиномы)
"""
func1 = np.poly1d(np.polyfit(numpy_x, numpy_y, 1))
plt.plot(x1, func1(x1))
func2 = np.poly1d(np.polyfit(numpy_x, numpy_y, 2))
plt.plot(x1, func2(x1))
func3 = np.poly1d(np.polyfit(numpy_x, numpy_y, 3))
plt.plot(x1, func3(x1))
func4 = np.poly1d(np.polyfit(numpy_x, numpy_y, 4))
plt.plot(x1, func4(x1))
func5 = np.poly1d(np.polyfit(numpy_x, numpy_y, 5))
plt.plot(x1, func5(x1))

#plt.show()
