# python_sem6_lr3

## Лабораторная работа 3 "Регрессия"

#### Градиентный спуск для линейной регрессии с одной переменной
##### Градиентный спуск — метод нахождения минимального значения функции потерь (существует множество видов этой функции).
      Минимизация любой функции означает поиск самой глубокой впадины в этой функции. 
      
##### Суть алгоритма – процесс получения наименьшего значения ошибки. 
##### Цель линейной регрессии — поиск линии, которая наилучшим образом соответствует этим точкам:
      
 ![](https://neurohive.io/wp-content/uploads/2018/10/lineinaja-regressija-e1539097909123.png)

##### Нахождение коэффицентов theta0 и theta1:
![](https://github.com/Kseniaveh/python_sem6_lr3/blob/master/kof.png)

m - количество элементов выборки

a - скорость обучения

Коэффициент скорости обучения, его называют гипер-параметром. Гипер-параметр – значение, 
требуемое вашей моделью, о котором мы действительно имеем очень смутное представление. 
Коэффициент скорости обучения можно рассматривать как «шаг в правильном 
направлении», где направление происходит от dJ/dw.

Следует, что 
h(x[i]) = theta0 + theta1 * x[i]

#### Результаты первой части лабораторной рабыты

![](https://github.com/Kseniaveh/python_sem6_lr3/blob/master/lr3_1.png)

#### Результаты второй части лабораторной рабыты

![](https://github.com/Kseniaveh/python_sem6_lr3/blob/master/result2.png)

      Среднее квадратичное отклонение/средняя квадратичная ошибка чаще всего используется для оценки точности измерений, 
      то есть для определения степени близости результата измерения к истинному значению измеряемой величины.
