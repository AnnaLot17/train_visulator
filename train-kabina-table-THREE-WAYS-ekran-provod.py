from math import log, pi, atan, exp
import numpy as np
from shapely.geometry import Polygon, LineString, Point

# ГРАФИКА
'''
Ось х - ось вдоль рельс
Ось y - ось поперёк рельс
Ось z - ось вверх к проводам

'''

# РЕЖИМ РАБОТЫ СЕТИ

# первый путь
I1 = 300  # cуммарная сила тока, А
U1 = 30000  # cуммарное напряжение, В

# второй путь
I2 = 300  # cуммарная сила тока, А
U2 = 30000  # cуммарное напряжение, В

# третий путь
I3 = 300  # cуммарная сила тока, А
U3 = 30000  # cуммарное напряжение, В
'''
I_ted = 880  # сила тока в ТЭД, А
U_ted = 1950  # напряжение в ТЭД, В
'''
# СТАТИСТИЧЕСКИЕ ДАННЫЕ
x_chel = 0.9  # положение человека по оси х
y_chel = 0.9  # положение человека по оси y
floor = 2  # расстояние от земли до дна кабины
z_chair = floor + 1.2  # сидушка стула
z_chel = floor + 1.5  # где находится человек по оси z
a = 1.75  # высота человека метры
b = 80  # масса человека килограммы
ti = 1  # длительность пребывания работника на рабочем месте, часов
z_graph = z_chel  # высота среза

# КОНСТАНТЫ

dis = 100  # дискретизация графиков
harm = {50: [1, 1],
        150: [0.3061, 0.400],
        250: [0.1469, 0.115],
        350: [0.0612, 0.050],
        450: [0.0429, 0.040],
        550: [0.0282, 0.036],
        650: [0.0196, 0.032],
        750: [0.0147, 0.022]}

# ДАННЫЕ О КОНТАКТНОЙ СЕТИ

xp = 0.760  # m - половина расстояния между рельсами

d_kp = 12.81 / 1000  # mm толщина провода КП
d_nt = 12.5 / 1000  # mm толщина провода НТ
d_up = 17.5 / 1000  # mm толщина провода УП
d_ep = 12.5 / 1000  # mm толщина провода ЕП
h_kp = 6.0  # КП высота провода
h_nt = 7.8  # НТ высота провода
h_up = 8.0  # УП высота провода
h_ep = 8.4  # ЕП высота провода

# первый путь
xp_kp = 0  # m - расстояние от центра между рельсами до КП (если левее центра - поставить минус)
xp_nt = 0  # m - расстояние от центра между рельсами до НТ (если левее центра - поставить минус)
xp_up = -3.7  # m - расстояние от центра между рельсами до УП (если правее центра - убрать минус)
xp_ep = -2.7   # m - расстояние от центра между рельсами до ЭП (если правее центра - убрать минус)

# второй путь
xp_mid12 = 4.2  # расстояние между центрами перового и второго путей
xp_kp2 = 0  # m - расстояние от центра между рельсами до КП2 (если левее центра - поставить минус)
xp_nt2 = 0  # m - расстояние от центра между рельсами до НТ2 (если левее центра - поставить минус)
xp_up2 = 3.7  # m - расстояние от центра между рельсами до УП2 (если левее центра - поставить минус)
xp_ep2 = 2.7  # m - расстояние от центра между вторыми рельсами до ЭП2 (если левее центра - поставить минус)

# третий путь
# todo провода справа или слева?
xp_mid23 = 4.2  # расстояние между центрами второго и третьего путей
xp_mid13 = xp_mid12 + xp_mid23
xp_kp3 = 0  # m - расстояние от центра между рельсами до КП3 (если левее центра - поставить минус)
xp_nt3 = 0  # m - расстояние от центра между рельсами до НТ3 (если левее центра - поставить минус)
xp_up3 = 3.7  # m - расстояние от центра между рельсами до УП2
xp_ep3 = 2.7  # m - расстояние от центра между вторыми рельсами до ЭП2


# ДАННЫЕ О ЛОКОМОТИВЕ

length = 1.3  # длина кабины
all_length = 15.2  # длина всего локомотива
width = 2.8  # ширина кабины
height = 2.6  # высота кабины
# min_x, max_x, min_y, max_y, min_z, max_z
bor = [0.2, 0.6, -1.2, 1.2, floor+1.5, floor+2.2]  # узлы окна
# min_x, max_x, min_z, max_z
sbor = [0.3, 1, floor+1.5, floor+2.2]  # узлы для бокового окна

# расчёты геомертрии окна
frontWindleft = Polygon([(bor[0], bor[2], bor[4]),
                         (bor[1], bor[2], bor[5]),
                         (bor[1], -0.22, bor[5]),
                         (bor[0], -0.22, bor[4])])

frontWindright = Polygon([(bor[0], 0.22, bor[4]),
                          (bor[1], 0.22, bor[5]),
                          (bor[1], bor[3], bor[5]),
                          (bor[0], bor[3], bor[4])])

min_nt = Point(0.5*width, sbor[3]).distance(Point(xp_nt, h_nt))
max_nt = Point(0.5*width, sbor[2]).distance(Point(xp_nt, h_nt))

min_kp = Point(0.5*width, sbor[3]).distance(Point(xp_kp, h_kp))
max_kp = Point(0.5*width, sbor[2]).distance(Point(xp_kp, h_kp))

min_up_l = Point(-0.5 * width, sbor[3]).distance(Point(xp_up, h_up))
max_up_l = Point(-0.5 * width, sbor[2]).distance(Point(xp_up, h_up))
min_up_r = Point(0.5 * width, sbor[3]).distance(Point(xp_up, h_up))
max_up_r = Point(0.5 * width, sbor[2]).distance(Point(xp_up, h_up))

min_ep_l = Point(-0.5*width, sbor[3]).distance(Point(xp_ep, h_ep))
max_ep_l = Point(-0.5*width, sbor[2]).distance(Point(xp_ep, h_ep))
min_ep_r = Point(0.5*width, sbor[3]).distance(Point(xp_ep, h_ep))
max_ep_r = Point(0.5*width, sbor[2]).distance(Point(xp_ep, h_ep))

min_nt2_l = Point(-0.5 * width, sbor[3]).distance(Point(xp_nt2 + xp_mid12, h_nt))
max_nt2_l = Point(-0.5 * width, sbor[2]).distance(Point(xp_nt2 + xp_mid12, h_nt))
min_nt2_r = Point(0.5 * width, sbor[3]).distance(Point(xp_nt2 + xp_mid12, h_nt))
max_nt2_r = Point(0.5 * width, sbor[2]).distance(Point(xp_nt2 + xp_mid12, h_nt))

min_kp2_l = Point(-0.5 * width, sbor[3]).distance(Point(xp_kp2 + xp_mid12, h_kp))
max_kp2_l = Point(-0.5 * width, sbor[2]).distance(Point(xp_kp2 + xp_mid12, h_kp))
min_kp2_r = Point(0.5 * width, sbor[3]).distance(Point(xp_kp2 + xp_mid12, h_kp))
max_kp2_r = Point(0.5 * width, sbor[2]).distance(Point(xp_kp2 + xp_mid12, h_kp))

min_up2_l = Point(-0.5 * width, sbor[3]).distance(Point(xp_up2 + xp_mid12, h_up))
max_up2_l = Point(-0.5 * width, sbor[2]).distance(Point(xp_up2 + xp_mid12, h_up))
min_up2_r = Point(0.5 * width, sbor[3]).distance(Point(xp_up2 + xp_mid12, h_up))
max_up2_r = Point(0.5 * width, sbor[2]).distance(Point(xp_up2 + xp_mid12, h_up))

min_ep2_l = Point(-0.5*width, sbor[3]).distance(Point(xp_ep2+xp_mid12, h_ep))
max_ep2_l = Point(-0.5*width, sbor[2]).distance(Point(xp_ep2+xp_mid12, h_ep))
min_ep2_r = Point(0.5*width, sbor[3]).distance(Point(xp_ep2+xp_mid12, h_ep))
max_ep2_r = Point(0.5*width, sbor[2]).distance(Point(xp_ep2+xp_mid12, h_ep))

min_nt3_l = Point(-0.5 * width, sbor[3]).distance(Point(xp_nt3 + xp_mid13, h_nt))
max_nt3_l = Point(-0.5 * width, sbor[2]).distance(Point(xp_nt3 + xp_mid13, h_nt))
min_nt3_r = Point(0.5 * width, sbor[3]).distance(Point(xp_nt3 + xp_mid13, h_nt))
max_nt3_r = Point(0.5 * width, sbor[2]).distance(Point(xp_nt3 + xp_mid13, h_nt))

min_kp3_l = Point(-0.5 * width, sbor[3]).distance(Point(xp_kp3 + xp_mid13, h_kp))
max_kp3_l = Point(-0.5 * width, sbor[2]).distance(Point(xp_kp3 + xp_mid13, h_kp))
min_kp3_r = Point(0.5 * width, sbor[3]).distance(Point(xp_kp3 + xp_mid13, h_kp))
max_kp3_r = Point(0.5 * width, sbor[2]).distance(Point(xp_kp3 + xp_mid13, h_kp))

min_up3_l = Point(-0.5 * width, sbor[3]).distance(Point(xp_up3 + xp_mid13, h_up))
max_up3_l = Point(-0.5 * width, sbor[2]).distance(Point(xp_up3 + xp_mid13, h_up))
min_up3_r = Point(0.5 * width, sbor[3]).distance(Point(xp_up3 + xp_mid13, h_up))
max_up3_r = Point(0.5 * width, sbor[2]).distance(Point(xp_up3 + xp_mid13, h_up))

min_ep3_l = Point(-0.5*width, sbor[3]).distance(Point(xp_ep3+xp_mid13, h_ep))
max_ep3_l = Point(-0.5*width, sbor[2]).distance(Point(xp_ep3+xp_mid13, h_ep))
min_ep3_r = Point(0.5*width, sbor[3]).distance(Point(xp_ep3+xp_mid13, h_ep))
max_ep3_r = Point(0.5*width, sbor[2]).distance(Point(xp_ep3+xp_mid13, h_ep))


Z0 = 377  # волновое сопротивление поля, Ом
mu = 1000  # относительная магнитная проницаемость стали
dst = 0.0025  # толщина стали м
sigma = 10 ** 7  # удельная проводимость стали

v_kab = 1.3 * 2.8 * 2.6  # объём кабины, м
r_kab = (v_kab * 3 / (4 * pi)) ** (1 / 3)  # эквивалентный радиус кабины, м

# расчёты коэффициентов экранирования
kh = (1 + 1000 * dst / (2 * r_kab)) ** 2
ke_post = 60 * pi * dst * sigma
ke_per = {}
Ce = exp(2 * pi * 0.0025 / 0.01)
for fr in harm.keys():
    lam = 300000000 / (fr / 1000000)
    Ze = Z0 * lam / (2 * pi * r_kab)
    delta = 0.03 * ((10 ** -7) * lam) ** 0.5
    A = (delta * Ze / (10 ** -7)) ** 0.5
    B = (lam / r_kab) ** (1 / 3)
    ke_per[fr] = 0.024 * A * B * Ce * 0.001

'''
# ОБОРУДОВАНИЕ
x_td1_sr = 0.9  # тяговый двигатель
dy_td = 0.8
r_td = 0.604
l_td = 0.66
z_td = 1
'''

# РАСЧЁТЫ

# теорема Пифагора
def mix(h_x, h_zz):
    return (h_x ** 2 + h_zz ** 2) ** 0.5


# расчёт магнитного поля
def magnetic_calc(x_m, z_m, f_m):
    # первый путь
    
    I_h = I1 * harm.get(f_m)[0]

    Ikp = 0.41 * I_h
    Int = 0.20 * I_h
    Iup = 0.39 * I_h
    Iep = 0.40 * I_h

    x = x_m - xp_kp
    h1xkp = Ikp / (4 * pi) * (
                -z_m / ((x + xp) ** 2 + z_m**2) + (z_m - h_kp)/(x ** 2 + (h_kp - z_m)**2))
    h1zkp = Ikp / (4 * pi) * (x + xp) * (
                1 / ((x + xp) ** 2 + z_m ** 2) - 1/(x ** 2 + (h_kp - z_m) ** 2))
    h1kp = mix(h1xkp, h1zkp)
    x = x_m - 2*xp - xp_kp
    h2xkp = Ikp / (4 * pi) * (
                -z_m / ((x + xp) ** 2 + z_m ** 2) + (z_m - h_kp) / ((x + 2*xp) ** 2 + (h_kp - z_m) ** 2))
    h2zkp = Ikp / (4 * pi) * (x + xp) * (
                1 / ((x + xp) ** 2 + z_m ** 2) - 1 / ((x + 2*xp) ** 2 + (h_kp - z_m) ** 2))
    h2kp = mix(h2xkp, h2zkp)
    hkp = h1kp + h2kp
    
    x = x_m - xp_nt
    h1xnt = Int / (4 * pi) * (
            -z_m / ((x + xp) ** 2 + z_m ** 2) + (z_m - h_nt) / (x ** 2 + (h_nt - z_m) ** 2))
    h1znt = Int / (4 * pi) * (x + xp) * (
            1 / ((x + xp) ** 2 + z_m ** 2) - 1 / (x ** 2 + (h_nt - z_m) ** 2))
    h1nt = mix(h1xnt, h1znt)
    x = x_m - 2 * xp - xp_nt
    h2xnt = Int / (4 * pi) * (
            -z_m / ((x + xp) ** 2 + z_m ** 2) + (z_m - h_nt) / ((x + 2 * xp) ** 2 + (h_nt - z_m) ** 2))
    h2znt = Int / (4 * pi) * (x + xp) * (
            1 / ((x + xp) ** 2 + z_m ** 2) - 1 / ((x + 2 * xp) ** 2 + (h_nt - z_m) ** 2))
    h2nt = mix(h2xnt, h2znt)
    hnt = h1nt + h2nt

    x = x_m - xp_up
    x2 = -xp + xp_up
    h1xup = Iup / (4 * pi) * (
            -z_m / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) + (z_m - h_up) / (x ** 2 + (h_up - z_m) ** 2))
    h1zup = Iup / (4 * pi) * (x2 + 2 * xp + x) * (
            1 / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) - 1 / (x ** 2 + (h_up - z_m) ** 2))
    h1up = mix(h1xup, h1zup)
    x = x_m - xp_up - 2 * xp
    x2 = -xp + xp_up
    h2xup = Iup / (4 * pi) * (
            -z_m / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) + (z_m - h_up) / ((x + 2 * xp) ** 2 + (h_up - z_m) ** 2))
    h2zup = Iup / (4 * pi) * (
            (x2 + 2 * xp + x) / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) - (x + 2 * xp) / (
            (x + 2 * xp) ** 2 + (h_up - z_m) ** 2))
    h2up = mix(h2xup, h2zup)
    hup = h1up + h2up
    
    x = x_m - xp_ep
    x2 = -xp + xp_ep
    h1xep = Iep / (4 * pi) * (
            -z_m / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) + (z_m - h_ep) / (x ** 2 + (h_ep - z_m) ** 2))
    h1zep = Iep / (4 * pi) * (x2 + 2 * xp + x) * (
            1 / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) - 1 / (x ** 2 + (h_ep - z_m) ** 2))
    h1ep = mix(h1xep, h1zep)
    x = x_m - xp_ep - 2 * xp
    x2 = -xp + xp_ep
    h2xep = Iep / (4 * pi) * (
            -z_m / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) + (z_m - h_ep) / ((x + 2 * xp) ** 2 + (h_ep - z_m) ** 2))
    h2zep = Iep / (4 * pi) * (
            (x2 + 2 * xp + x) / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) - (x + 2 * xp) / (
            (x + 2 * xp) ** 2 + (h_ep - z_m) ** 2))
    h2ep = mix(h2xep, h2zep)
    hep = h1ep + h2ep
    
    # второй путь
    
    I_h = I2 * harm.get(f_m)[0]

    Ikp = 0.41 * I_h
    Int = 0.20 * I_h
    Iup = 0.39 * I_h
    Iep = 0.40 * I_h

    x = x_m - (xp_kp2 + xp_mid12)
    h1xkp = Ikp / (4 * pi) * (
                -z_m / ((x + xp) ** 2 + z_m**2) + (z_m - h_kp)/(x ** 2 + (h_kp - z_m)**2))
    h1zkp = Ikp / (4 * pi) * (x + xp) * (
                1 / ((x + xp) ** 2 + z_m ** 2) - 1/(x ** 2 + (h_kp - z_m) ** 2))
    h1kp = mix(h1xkp, h1zkp)
    x = x_m - 2*xp - (xp_kp2 + xp_mid12)
    h2xkp = Ikp / (4 * pi) * (
                -z_m / ((x + xp) ** 2 + z_m ** 2) + (z_m - h_kp) / ((x + 2*xp) ** 2 + (h_kp - z_m) ** 2))
    h2zkp = Ikp / (4 * pi) * (x + xp) * (
                1 / ((x + xp) ** 2 + z_m ** 2) - 1 / ((x + 2*xp) ** 2 + (h_kp - z_m) ** 2))
    h2kp = mix(h2xkp, h2zkp)
    hkp_scd = h1kp + h2kp

    x = x_m - (xp_nt2 + xp_mid12)
    h1xnt = Int / (4 * pi) * (
            -z_m / ((x + xp) ** 2 + z_m ** 2) + (z_m - h_nt) / (x ** 2 + (h_nt - z_m) ** 2))
    h1znt = Int / (4 * pi) * (x + xp) * (
            1 / ((x + xp) ** 2 + z_m ** 2) - 1 / (x ** 2 + (h_nt - z_m) ** 2))
    h1nt = mix(h1xnt, h1znt)
    x = x_m - 2 * xp - (xp_nt2 + xp_mid12)
    h2xnt = Int / (4 * pi) * (
            -z_m / ((x + xp) ** 2 + z_m ** 2) + (z_m - h_nt) / ((x + 2 * xp) ** 2 + (h_nt - z_m) ** 2))
    h2znt = Int / (4 * pi) * (x + xp) * (
            1 / ((x + xp) ** 2 + z_m ** 2) - 1 / ((x + 2 * xp) ** 2 + (h_nt - z_m) ** 2))
    h2nt = mix(h2xnt, h2znt)
    hnt_scd = h1nt + h2nt

    x = x_m - (xp_up2 + xp_mid12)
    x2 = -xp + xp_up2
    h1xup = Iup / (4 * pi) * (
            -z_m / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) + (z_m - h_up) / (x ** 2 + (h_up - z_m) ** 2))
    h1zup = Iup / (4 * pi) * (x2 + 2 * xp + x) * (
            1 / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) - 1 / (x ** 2 + (h_up - z_m) ** 2))
    h1up = mix(h1xup, h1zup)
    x = x_m - (xp_up2 + xp_mid12) - 2 * xp
    x2 = -xp + xp_up2
    h2xup = Iup / (4 * pi) * (
            -z_m / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) + (z_m - h_up) / ((x + 2 * xp) ** 2 + (h_up - z_m) ** 2))
    h2zup = Iup / (4 * pi) * (
            (x2 + 2 * xp + x) / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) - (x + 2 * xp) / (
                (x + 2 * xp) ** 2 + (h_up - z_m) ** 2))
    h2up = mix(h2xup, h2zup)
    hup_sec = h1up + h2up

    x = x_m - (xp_ep2 + xp_mid12)
    x2 = -xp + xp_ep2
    h1xep_2 = Iep / (4 * pi) * (
            -z_m / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) + (z_m - h_ep) / (x ** 2 + (h_ep - z_m) ** 2))
    h1zep_2 = Iep / (4 * pi) * (x2 + 2 * xp + x) * (
            1 / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) - 1 / (x ** 2 + (h_ep - z_m) ** 2))
    h1ep_2 = mix(h1xep_2, h1zep_2)
    x = x_m - (xp_ep2 + xp_mid12) - 2 * xp
    x2 = -xp + xp_ep2
    h2xep_2 = Iep / (4 * pi) * (
            -z_m / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) + (z_m - h_ep) / ((x + 2 * xp) ** 2 + (h_ep - z_m) ** 2))
    h2zep_2 = Iep / (4 * pi) * (
            (x2 + 2 * xp + x) / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) - (x + 2 * xp) / (
            (x + 2 * xp) ** 2 + (h_ep - z_m) ** 2))
    h2ep_2 = mix(h2xep_2, h2zep_2)
    hep_sec = h1ep_2 + h2ep_2
    
    # третий путь
    
    I_h = I3 * harm.get(f_m)[0]

    Ikp = 0.41 * I_h
    Int = 0.20 * I_h
    Iup = 0.39 * I_h
    Iep = 0.40 * I_h

    x = x_m - (xp_kp2 + xp_mid13)
    h1xkp = Ikp / (4 * pi) * (
                -z_m / ((x + xp) ** 2 + z_m**2) + (z_m - h_kp)/(x ** 2 + (h_kp - z_m)**2))
    h1zkp = Ikp / (4 * pi) * (x + xp) * (
                1 / ((x + xp) ** 2 + z_m ** 2) - 1/(x ** 2 + (h_kp - z_m) ** 2))
    h1kp = mix(h1xkp, h1zkp)
    x = x_m - 2*xp - (xp_kp2 + xp_mid13)
    h2xkp = Ikp / (4 * pi) * (
                -z_m / ((x + xp) ** 2 + z_m ** 2) + (z_m - h_kp) / ((x + 2*xp) ** 2 + (h_kp - z_m) ** 2))
    h2zkp = Ikp / (4 * pi) * (x + xp) * (
                1 / ((x + xp) ** 2 + z_m ** 2) - 1 / ((x + 2*xp) ** 2 + (h_kp - z_m) ** 2))
    h2kp = mix(h2xkp, h2zkp)
    hkp_thd = h1kp + h2kp

    x = x_m - (xp_nt2 + xp_mid13)
    h1xnt = Int / (4 * pi) * (
            -z_m / ((x + xp) ** 2 + z_m ** 2) + (z_m - h_nt) / (x ** 2 + (h_nt - z_m) ** 2))
    h1znt = Int / (4 * pi) * (x + xp) * (
            1 / ((x + xp) ** 2 + z_m ** 2) - 1 / (x ** 2 + (h_nt - z_m) ** 2))
    h1nt = mix(h1xnt, h1znt)
    x = x_m - 2 * xp - (xp_nt2 + xp_mid13)
    h2xnt = Int / (4 * pi) * (
            -z_m / ((x + xp) ** 2 + z_m ** 2) + (z_m - h_nt) / ((x + 2 * xp) ** 2 + (h_nt - z_m) ** 2))
    h2znt = Int / (4 * pi) * (x + xp) * (
            1 / ((x + xp) ** 2 + z_m ** 2) - 1 / ((x + 2 * xp) ** 2 + (h_nt - z_m) ** 2))
    h2nt = mix(h2xnt, h2znt)
    hnt_thd = h1nt + h2nt

    x = x_m - (xp_up2 + xp_mid13)
    x2 = -xp + xp_up2
    h1xup = Iup / (4 * pi) * (
            -z_m / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) + (z_m - h_up) / (x ** 2 + (h_up - z_m) ** 2))
    h1zup = Iup / (4 * pi) * (x2 + 2 * xp + x) * (
            1 / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) - 1 / (x ** 2 + (h_up - z_m) ** 2))
    h1up = mix(h1xup, h1zup)
    x = x_m - (xp_up2 + xp_mid13) - 2 * xp
    x2 = -xp + xp_up2
    h2xup = Iup / (4 * pi) * (
            -z_m / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) + (z_m - h_up) / ((x + 2 * xp) ** 2 + (h_up - z_m) ** 2))
    h2zup = Iup / (4 * pi) * (
            (x2 + 2 * xp + x) / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) - (x + 2 * xp) / (
                (x + 2 * xp) ** 2 + (h_up - z_m) ** 2))
    h2up = mix(h2xup, h2zup)
    hup_thd = h1up + h2up

    x = x_m - (xp_ep2 + xp_mid13)
    x2 = -xp + xp_ep2
    h1xep_2 = Iep / (4 * pi) * (
            -z_m / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) + (z_m - h_ep) / (x ** 2 + (h_ep - z_m) ** 2))
    h1zep_2 = Iep / (4 * pi) * (x2 + 2 * xp + x) * (
            1 / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) - 1 / (x ** 2 + (h_ep - z_m) ** 2))
    h1ep_2 = mix(h1xep_2, h1zep_2)
    x = x_m - (xp_ep2 + xp_mid13) - 2 * xp
    x2 = -xp + xp_ep2
    h2xep_2 = Iep / (4 * pi) * (
            -z_m / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) + (z_m - h_ep) / ((x + 2 * xp) ** 2 + (h_ep - z_m) ** 2))
    h2zep_2 = Iep / (4 * pi) * (
            (x2 + 2 * xp + x) / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) - (x + 2 * xp) / (
            (x + 2 * xp) ** 2 + (h_ep - z_m) ** 2))
    h2ep_2 = mix(h2xep_2, h2zep_2)
    hep_thd = h1ep_2 + h2ep_2

    # результат: первый путь КП НТ УП, второй путь КП НТ УП, третий путь КП НТ УП, ЭП для всех
    return [hkp, hnt, hup, hkp_scd, hnt_scd, hup_sec, hkp_thd, hnt_thd, hup_thd, -hep, -hep_sec, -hep_thd]


def electric_calc(x_e, z_e, f_e):

    U_h = U1 * harm.get(f_e)[1]

    ekp = U_h * log(1 + 4 * h_nt * z_e / ((x_e - xp_nt) ** 2 + (h_nt - z_e) ** 2)) / (2 * z_e * log(2 * h_nt / d_nt))
    ent = U_h * log(1 + 4 * h_kp * z_e / ((x_e - xp_kp) ** 2 + (h_kp - z_e) ** 2)) / (2 * z_e * log(2 * h_kp / d_kp))
    eup = U_h * log(1 + 4 * h_up * z_e / ((x_e - xp_up) ** 2 + (h_up - z_e) ** 2)) / (2 * z_e * log(2 * h_up / d_up))
    eep = U_h * log(1 + 4 * h_ep * z_e / ((x_e - xp_ep) ** 2 + (h_ep - z_e) ** 2)) / (2 * z_e * log(2 * h_ep / d_ep))
    
    U_h = U2 * harm.get(f_e)[1]

    ekp_scd = U_h * log(1 + 4 * h_nt * z_e / ((x_e - xp_nt2 - xp_mid12) ** 2 + (h_nt - z_e) ** 2)) / (
                2 * z_e * log(2 * h_nt / d_nt))
    ent_scd = U_h * log(1 + 4 * h_kp * z_e / ((x_e - xp_kp2 - xp_mid12) ** 2 + (h_kp - z_e) ** 2)) / (
                2 * z_e * log(2 * h_kp / d_kp))
    eup_scd = U_h * log(1 + 4 * h_up * z_e / ((x_e - xp_up2 - xp_mid12) ** 2 + (h_up - z_e) ** 2)) / (
                2 * z_e * log(2 * h_up / d_up))
    eep_scd = U_h * log(1 + 4 * h_ep * z_e / ((x_e - xp_ep2 - xp_mid12) ** 2 + (h_ep - z_e) ** 2)) / (
                2 * z_e * log(2 * h_ep / d_ep))
                
    U_h = U3 * harm.get(f_e)[1]

    ekp_thd = U_h * log(1 + 4 * h_nt * z_e / ((x_e - xp_nt2 - xp_mid13) ** 2 + (h_nt - z_e) ** 2)) / (
                2 * z_e * log(2 * h_nt / d_nt))
    ent_thd = U_h * log(1 + 4 * h_kp * z_e / ((x_e - xp_kp2 - xp_mid13) ** 2 + (h_kp - z_e) ** 2)) / (
                2 * z_e * log(2 * h_kp / d_kp))
    eup_thd = U_h * log(1 + 4 * h_up * z_e / ((x_e - xp_up2 - xp_mid13) ** 2 + (h_up - z_e) ** 2)) / (
                2 * z_e * log(2 * h_up / d_up))
    eep_thd = U_h * log(1 + 4 * h_ep * z_e / ((x_e - xp_ep2 - xp_mid13) ** 2 + (h_ep - z_e) ** 2)) / (
                2 * z_e * log(2 * h_ep / d_ep))
   # результат: первый путь КП НТ УП, второй путь КП НТ УП, третий путь КП НТ УП, ЭП для всех
    return [ekp, ent, eup, ekp_scd, ent_scd, eup_scd, ekp_thd, ent_thd, eup_thd, -eep, -eep_scd, -eep_thd]


def full_field(res_en):
    sum_h, sum_e, sum_g = 0, 0, 0
    for en in res_en[0].values():
        sum_h += sum(en[0])
        sum_e += sum(en[1])
        sum_g += en[0][0] * en[1][0] + en[0][1] * en[1][1] + en[0][2] * en[1][2] + \
                 en[0][3] * en[1][3] + en[0][4] * en[1][4] + en[0][5] * en[1][5] + \
                 en[0][6] * en[1][6] + en[0][7] * en[1][7] + en[0][8] * en[1][8] + \
                 (-en[0][9] * en[1][9] - en[0][10] * en[1][10])- en[0][11] * en[1][11]
    return [sum_h, sum_e, sum_g]


def ekran(en):
    x, y, z = en[1]  # координаты точки

    # расстояние от текущей точки до проводов - для расчёта лобовых окон
    kppth = LineString([(x, y, z), (x, xp_kp, h_kp)])
    ntpth = LineString([(x, y, z), (x, xp_nt, h_nt)])
    uppth = LineString([(x, y, z), (x, xp_up, h_up)])
    eppth = LineString([(x, y, z), (x, xp_ep, h_ep)])
    # проверяем, попадает ли лобовое окно по направлению от текущей точки до проводов
    kp_pass = kppth.intersects(frontWindleft) or kppth.intersects(frontWindright)
    nt_pass = ntpth.intersects(frontWindleft) or ntpth.intersects(frontWindright)
    up_pass = uppth.intersects(frontWindleft) or uppth.intersects(frontWindright)
    ep_pass = eppth.intersects(frontWindleft) or eppth.intersects(frontWindright)

    kp_dist = Point(y, z).distance(Point(xp_kp, h_kp))
    kp_pass |= (kp_dist >= min_kp) and (kp_dist <= max_kp) and (x >= sbor[0]) and (x <= sbor[1]) \
               and (z >= sbor[2]) and (z <= sbor[3])
    kp_pass |= (x >= sbor[0]) and (x <= sbor[1]) and (z >= sbor[2]) and (z <= sbor[3]) \
               and (z >= sbor[2]) and (z <= sbor[3])

    nt_dist = Point(y, z).distance(Point(xp_nt, h_nt))
    nt_pass |= (nt_dist >= min_nt) and (nt_dist <= max_nt) and (x >= sbor[0]) and (x <= sbor[1]) \
               and (z >= sbor[2]) and (z <= sbor[3])
    nt_pass |= (x >= sbor[0]) and (x <= sbor[1]) and (z >= sbor[2]) and (z <= sbor[3]) \
               and (z >= sbor[2]) and (z <= sbor[3])

    up_dist = Point(y, z).distance(Point(xp_up, h_up))
    up_pass |= (up_dist >= min_up_l) and (up_dist <= max_up_l) and (x >= sbor[0]) and (x <= sbor[1]) \
               and (z >= sbor[2]) and (z <= sbor[3])
    up_pass |= (up_dist >= min_up_r) and (up_dist <= max_up_r) and (x >= sbor[0]) and (x <= sbor[1]) \
               and (z >= sbor[2]) and (z <= sbor[3])

    ep_dist = Point(y, z).distance(Point(xp_ep, h_ep))
    ep_pass |= (ep_dist >= min_ep_l) and (ep_dist <= max_ep_l) and (x >= sbor[0]) and (x <= sbor[1]) \
               and (z >= sbor[2]) and (z <= sbor[3])
    ep_pass |= (ep_dist >= min_ep_r) and (ep_dist <= max_ep_r) and (x >= sbor[0]) and (x <= sbor[1]) \
               and (z >= sbor[2]) and (z <= sbor[3])

    kp_sec_dist = Point(y, z).distance(Point(xp_kp2 + xp_mid12, h_kp))
    kp_sec_pass = (kp_sec_dist >= min_kp2_l) and (kp_sec_dist <= max_kp2_l) and (x >= sbor[0]) and (x <= sbor[1]) \
                  and (z >= sbor[2]) and (z <= sbor[3])
    kp_sec_pass |= (kp_sec_dist >= min_kp2_r) and (kp_sec_dist <= max_kp2_r) and (x >= sbor[0]) and (x <= sbor[1]) \
                   and (z >= sbor[2]) and (z <= sbor[3])

    nt_sec_dist = Point(y, z).distance(Point(xp_nt2 + xp_mid12, h_nt))
    nt_sec_pass = (nt_sec_dist >= min_nt2_l) and (nt_sec_dist <= max_nt2_l) and (x >= sbor[0]) and (x <= sbor[1]) \
                  and (z >= sbor[2]) and (z <= sbor[3])
    nt_sec_pass |= (nt_sec_dist >= min_nt2_r) and (nt_sec_dist <= max_nt2_r) and (x >= sbor[0]) and (x <= sbor[1]) \
                   and (z >= sbor[2]) and (z <= sbor[3])

    up_sec_dist = Point(y, z).distance(Point(xp_up2 + xp_mid12, h_up))
    up_sec_pass = (up_sec_dist >= min_up2_l) and (up_sec_dist <= max_up2_l) and (x >= sbor[0]) and (x <= sbor[1]) \
                  and (z >= sbor[2]) and (z <= sbor[3])
    up_sec_pass |= (up_sec_dist >= min_up2_r) and (up_sec_dist <= max_up2_r) and (x >= sbor[0]) and (x <= sbor[1]) \
                   and (z >= sbor[2]) and (z <= sbor[3])

    ep_sec_dist = Point(y, z).distance(Point(xp_ep2 + xp_mid12, h_ep))
    ep_sec_pass = (ep_sec_dist >= min_ep2_l) and (ep_sec_dist <= max_ep2_l) and (x >= sbor[0]) and (x <= sbor[1]) \
                    and (z >= sbor[2]) and (z <= sbor[3])
    ep_sec_pass |= (ep_sec_dist >= min_ep2_r) and (ep_sec_dist <= max_ep2_r) and (x >= sbor[0]) and (x <= sbor[1]) \
                    and (z >= sbor[2]) and (z <= sbor[3])

    kp_thd_dist = Point(y, z).distance(Point(xp_kp3 + xp_mid13, h_kp))
    kp_thd_pass = (kp_thd_dist >= min_kp3_l) and (kp_thd_dist <= max_kp3_l) and (x >= sbor[0]) and (x <= sbor[1]) \
                  and (z >= sbor[2]) and (z <= sbor[3])
    kp_thd_pass |= (kp_thd_dist >= min_kp3_r) and (kp_thd_dist <= max_kp3_r) and (x >= sbor[0]) and (x <= sbor[1]) \
                   and (z >= sbor[2]) and (z <= sbor[3])

    nt_thd_dist = Point(y, z).distance(Point(xp_nt3 + xp_mid13, h_nt))
    nt_thd_pass = (nt_thd_dist >= min_nt3_l) and (nt_thd_dist <= max_nt3_l) and (x >= sbor[0]) and (x <= sbor[1]) \
                  and (z >= sbor[2]) and (z <= sbor[3])
    nt_thd_pass |= (nt_thd_dist >= min_nt3_r) and (nt_thd_dist <= max_nt3_r) and (x >= sbor[0]) and (x <= sbor[1]) \
                   and (z >= sbor[2]) and (z <= sbor[3])

    up_thd_dist = Point(y, z).distance(Point(xp_up3 + xp_mid13, h_up))
    up_thd_pass = (up_thd_dist >= min_up3_l) and (up_thd_dist <= max_up3_l) and (x >= sbor[0]) and (x <= sbor[1]) \
                  and (z >= sbor[2]) and (z <= sbor[3])
    up_thd_pass |= (up_thd_dist >= min_up3_r) and (up_thd_dist <= max_up3_r) and (x >= sbor[0]) and (x <= sbor[1]) \
                   and (z >= sbor[2]) and (z <= sbor[3])

    ep_thd_dist = Point(y, z).distance(Point(xp_ep3 + xp_mid13, h_ep))
    ep_thd_pass = (ep_thd_dist >= min_ep3_l) and (ep_thd_dist <= max_ep3_l) and (x >= sbor[0]) and (x <= sbor[1]) \
                    and (z >= sbor[2]) and (z <= sbor[3])
    ep_thd_pass |= (ep_thd_dist >= min_ep3_r) and (ep_thd_dist <= max_ep3_r) and (x >= sbor[0]) and (x <= sbor[1]) \
                    and (z >= sbor[2]) and (z <= sbor[3])

    if (abs(y) <= 0.5*width) and (z >= floor) and (z <= floor+height) and (x < length):
        if not kp_pass:
            for f in en[0].keys():
                en[0][f][0][0] /= kh
                en[0][f][1][0] /= ke_per[f]
        if not nt_pass:
            for f in en[0].keys():
                en[0][f][0][1] /= kh
                en[0][f][1][1] /= ke_per[f]
        if not up_pass:
            for f in en[0].keys():
                en[0][f][0][2] /= kh
                en[0][f][1][2] /= ke_per[f]
        
        if not kp_sec_pass:
            for f in en[0].keys():
                en[0][f][0][3] /= kh
                en[0][f][1][3] /= ke_per[fr]
        if not nt_sec_pass:
            for f in en[0].keys():
                en[0][f][0][4] /= kh
                en[0][f][1][4] /= ke_per[fr]
        if not up_sec_pass:
            for f in en[0].keys():
                en[0][f][0][5] /= kh
                en[0][f][1][5] /= ke_per[fr]

        if not kp_thd_pass:
            for f in en[0].keys():
                en[0][f][0][6] /= kh
                en[0][f][1][6] /= ke_per[fr]
        if not nt_thd_pass:
            for f in en[0].keys():
                en[0][f][0][7] /= kh
                en[0][f][1][7] /= ke_per[fr]
        if not up_thd_pass:
            for f in en[0].keys():
                en[0][f][0][8] /= kh
                en[0][f][1][8] /= ke_per[fr]
        
        if not ep_pass:
            for f in en[0].keys():
                en[0][f][0][9] /= kh
                en[0][f][1][9] /= ke_per[fr]
        if not ep_sec_pass:
            for f in en[0].keys():
                en[0][f][0][10] /= kh
                en[0][f][1][10] /= ke_per[fr]
        if not ep_thd_pass:
            for f in en[0].keys():
                en[0][f][0][11] /= kh
                en[0][f][1][11] /= ke_per[fr]        
    return en

def ekran_post(ext_en):
    k_h, k_e = 1, 1
    if (ext_en[1][2] > floor-1) and (ext_en[1][2] < floor+height):
        if abs(ext_en[1][1]) <= 0.5*width:
            k_h = kh
            k_e = ke_post
            if ext_en[1][2] > floor:
                k_h *= kh
                k_e *= ke_post

    return [[ext_en[0][0] / k_h, ext_en[0][1] / k_e], ext_en[1]]


def visual_front():
    Ymax = xp_up2 * 1.2 + xp_mid12
    Ymin = xp_up * 1.2
    Zmax = 0.1
    Zmin = max(h_kp, h_nt, h_up) * 1.1

    y = np.linspace(Ymin, Ymax, dis)
    z = np.linspace(Zmin, Zmax, dis)

    every_f = [[({fr: (magnetic_calc(y_, z_, fr), electric_calc(y_, z_, fr)) for fr in harm.keys()},
                 (x_chel, y_, z_)) for y_ in y] for z_ in z]

    return every_f

'''
def ted_field_calc(x_arr, y_arr, I_g, U_g, n, type_='UP'):
    ds = 8

    # разбиваем ТЭД на узлы
    nodes_x = [x_td1_sr + 0.5*r_td * np.cos(ap) for ap in np.linspace(0, 2*pi, ds)]
    nodes_z = [z_td + 0.5*r_td * np.sin(ap) for ap in np.linspace(0, 2*pi, ds)]
    nodes_y = [td-td_p for td in [0] for td_p in np.linspace(-0.5*l_td, 0.5*l_td, 4)]

    points = [[x_, y_, z_] for z_ in nodes_z for y_ in nodes_y for x_ in nodes_x]

    # разбиваем кабину на узлы
    if type_ == 'UP':
        minus = [[x_, y_, z_] for z_ in (floor, floor-1) for y_ in y_arr for x_ in x_arr]
    else:
        x_cab = np.linspace(0, length, 40)
        y_cab = np.linspace(-0.5*width, 0.5*width, 40)
        minus = [[x_, y_, z_] for z_ in (floor, floor-1) for y_ in y_cab for x_ in x_cab]

    def in_point(x_, y_, z_):
        H_ob, E_ob = 0, 0
        for p in points:
            r = ((p[0]-x_)**2 + (p[1]-y_)**2 + (p[2]-z_)**2) ** 0.5
            H_ob += I_g / (pi * l_td) * atan(l_td / 2 / r)
            E_ob += U_g / r / len(points)

        for m in minus:
            r_m = ((m[0] - x_) ** 2 + (m[1] - y_) ** 2 + (m[2] - z_) ** 2) ** 0.5
            if r_m != 0:
                E_ob += U_g / r_m / len(minus)
        return [[H_ob * n / len(points), E_ob], (x_, y_, z_)]

    if type_ == 'UP':
        return [in_point(x_, y_, z_graph) for y_ in y_arr for x_ in x_arr]
    else:
        return [in_point(x_chel, y_, z_) for z_ in y_arr for y_ in x_arr]
'''        

def visual_front_locomotive(ext_f):
    Ymin, Ymax = -0.6*width, 0.6*width
    Zmin, Zmax = floor+height+1, 0.1

    ekran_ = [[ekran(y_el) for y_el in z_list if abs(y_el[1][1]) <= Ymax] for z_list in ext_f
              if z_list[0][1][2] < Zmin]

    summar = [[full_field(x_el) for x_el in y_list] for y_list in ekran_]
    magnetic = [[x_el[0] for x_el in y_list] for y_list in summar]
    electric = [[x_el[1] for x_el in y_list] for y_list in summar]
    energy = [[x_el[2] for x_el in y_list] for y_list in summar]

    y_ln = np.linspace(Ymin, Ymax, len(ekran_[0]))
    z_ln = np.linspace(Zmin, Zmax, len(ekran_))

    def table_out(znach, f=0, t=0, ln=10):
        for y in y_ln:
            print(f'{y:.3f}'.ljust(ln), end='', file=rf)
        print('y / z\n', file=rf)
        for no, y_list in enumerate(znach):
            for dt in y_list:
                if f:
                    print(f'{sum(dt[0][f][t]):.3f}'.ljust(ln), end='', file=rf)
                else:
                    print(f'{dt:.3f}'.ljust(ln), end='', file=rf)
            print(f'| {z_ln[no]:.3f}', file=rf)
        print('\n', file=rf)

    rf = open('peremennoe_pole.txt', 'w')

    print('Верхняя строка - ось y, метры. Крайний правый столбец - ось z, метры. '
          'В ячейках - магнитная или электрическая напряжённость А/м и В/м соответственно.\n', file=rf)

    print('МАГНИТНОЕ ПОЛЕ\n', file=rf)
    print('Общее\n', file=rf)
    table_out(magnetic)
    print('Гармоники\n', file=rf)
    for fr in harm.keys():
        print(f'{fr} Гц\n', file=rf)
        table_out(ekran_, f=fr)

    print('ЭЛЕКТРИЧЕСКОЕ ПОЛЕ\n', file=rf)
    print('Общее\n', file=rf)
    table_out(electric)
    print('Гармоники\n', file=rf)
    for fr in harm.keys():
        print(f'{fr} Гц\n', file=rf)
        table_out(ekran_, f=fr, t=1)

    print('ЭНЕРГИЯ\n', file=rf)
    table_out(energy, ln=12)
    rf.close()


def visual_front_post():
    dis_y, dis_z = 60, 60
    y_ln = np.linspace(-0.6*width, 0.6*width, dis_y, endpoint=True)
    z_ln = np.linspace(floor+height+1, 0.1, dis_z, endpoint=True)

    ted_field = ted_field_calc(y_ln, z_ln, I_ted, U_ted, 5, type_='FRONT')

    kabina = [[ekran_post(el) for el in ted_field if (el[1][2] == z_)] for z_ in reversed(z_ln)]

    magnetic = [[el[0][0] for el in z_list] for z_list in kabina]
    electric = [[el[0][1] for el in z_list] for z_list in kabina]
    energy = [[el[0][0] * el[0][0] for el in z_list] for z_list in kabina]

    def table_out(znach, ln=12):
        for y in y_ln:
            print(f'{y:.3f}'.ljust(ln), end='', file=rf)
        print('y / z\n', file=rf)
        for no, y_list in enumerate(znach):
            for dt in y_list:
                print(f'{dt:.3f}'.ljust(ln), end='', file=rf)
            print(f'| {z_ln[no]:.3f}', file=rf)
        print('\n', file=rf)

    rf = open('postoyannoe_pole.txt', 'w')

    print('Верхняя строка - ось y, метры. Крайний правый столбец - ось z, метры. '
          'В ячейках - магнитная или электрическая напряжённость А/м и В/м соответственно.\n', file=rf)

    print('МАГНИТНОЕ ПОЛЕ\n', file=rf)
    table_out(magnetic)
    print('ЭЛЕКТРИЧЕСКОЕ ПОЛЕ\n', file=rf)
    table_out(electric)
    print('ЭНЕРГИЯ\n', file=rf)
    table_out(energy, ln=14)
    rf.close()


# ВЫВОД ПАРАМЕТРОВ

print('\nПараметры сети')
print(f'Высота КП: {h_kp} м')
print(f'Высота НЧ: {h_nt} м')
print(f'Высота УП: {h_up} м')
print('Первый путь')
print(f'Напряжение: {U1} Вольт')
print(f'Суммарный ток: {I1} Ампер')
print('Второй путь')
print(f'Напряжение: {U2} Вольт')
print(f'Суммарный ток: {I2} Ампер')
print('Третий путь')
print(f'Напряжение: {U3} Вольт')
print(f'Суммарный ток: {I3} Ампер')
print(f'\nНапряжение ТЭД: {U_ted} Вольт')
print(f'Ток ТЭД: {I_ted} Ампер')

# ПОСТРОЕНИЕ ГРАФИКА


print('\nРасчёт поля\n')
cont_f_front = visual_front()
visual_front_locomotive(cont_f_front)
#visual_front_post()

# РАСЧЁТ СТАТИСТИКИ

print('СТАТИСТИКА\n')
S = (a * b / 3600) ** 1 / 2
p = ti / 24  # статистическая вероятность воздействия

chel_f_per = [{fr: (magnetic_calc(y_chel, z_chel, fr), electric_calc(y_chel, z_chel, fr)) for fr in harm.keys()},
              (x_chel, y_chel, z_chel)]
no_ekran_per = full_field(chel_f_per)[2]
print('\nПеременное поле без экрана: %.4f' % no_ekran_per)

ekran_per = full_field(ekran(chel_f_per))[2]
print('\nПерменное поле с экраном %.4f' % ekran_per)
Dco = ekran_per * ti * S * p
Dpo = Dco / b
print('Удельная суточная доза поглощённой энергии: %.4f' % Dpo)

'''
chel_f_post = ted_field_calc([y_chel], [z_chel], I_ted, U_ted, 5, type_='FRONT')[0][0]
ekran_post_ = chel_f_post[0] / kh * chel_f_post[1] / ke_post
print('\nПостоянное поле с экраном %.4f' % ekran_post_)
Dco = ekran_post_ * ti * S * p
Dpo = Dco / b
print('Удельная суточная доза поглощённой энергии: %.8f' % Dpo)
'''
