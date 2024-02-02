# Импорт библиотек
from math import log, exp, pi, atan
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib.colors as colors
import matplotlib.patches as ptch
from shapely.geometry import Polygon, LineString, Point

# цветоваяя схема графиков
plt.style.use('seaborn-white')
cmap = 'YlOrRd'

# РЕЖИМ РАБОТЫ СЕТИ

I = 300  # cуммарная сила тока, А
U = 30000  # cуммарное напряжение, В

I_ted = 880  # сила тока в ТЭД, А
U_ted = 1950  # напряжение в ТЭД, В

# СТАТИСТИЧЕСКИЕ ДАННЫЕ
x_chel = 0.9  # положение человека по оси х
y_chel = 0.9  # положение человека по оси y
floor = 2  # расстояние от земли до дна кабины
gr_floor = 1  # высота самого низа электровоза
z_chair = floor + 1.2  # сидушка стула
z_chel = floor + 1.5  # где находится человек по оси z
a = 1.75  # высота человека метры
b = 80  # масса человека килограммы
ti = 1  # длительность пребывания работника на рабочем месте, часов
z_graph = z_chel  # высота среза

# КОНСТАНТЫ

dis = 100  # дискретизация графиков (меньше - менее точно, но быстрее считает; больше - точнее, но дольше расчёт)
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
xp_kp = 0  # m - расстояние от центра между рельсами до КП (если левее центра - поставить минус)
xp_nt = 0  # m - расстояние от центра между рельсами до НТ (если левее центра - поставить минус)
xp_up = -3.7  # m - расстояние от центра между рельсами до УП
d_kp = 12.81 / 1000  # mm
d_nt = 12.5 / 1000  # mm
d_up = 17.5 / 1000  # mm
h_kp = 6.0  # КП
h_nt = 7.8  # НТ
h_up = 8.0  # УП

xp_mid = 4.2  # расстояние между центрами путей
xp_kp2 = 0  # m - расстояние от центра между рельсами до КП2 (если левее центра - поставить минус)
xp_nt2 = 0  # m - расстояние от центра между рельсами до НТ2 (если левее центра - поставить минус)
xp_up2 = 3.7  # m - расстояние от центра между рельсами до УП2

# ДАННЫЕ О ЛОКОМОТИВЕ

length = 1.3  # длина кабины
all_length = 15.2  # длина всего локомотива
width = 2.8  # ширина кабины
height = 2.6  # высота кабины
# min_x, max_x, min_y, max_y, min_z, max_z
bor = [0.2, 0.6, -1.2, 1.2, floor + 1.5, floor + 2.2]  # узлы окна
# min_x, max_x, min_z, max_z
sbor = [0.3, 1, floor + 1.5, floor + 2.2]  # узлы для бокового окна

# формируем передние окна методом Polygon: составляем список из координат точек по x, y, z каждого угла
frontWindleft = Polygon([(bor[0], bor[2], bor[4]),
                         (bor[1], bor[2], bor[5]),
                         (bor[1], -0.22, bor[5]),
                         (bor[0], -0.22, bor[4])])

frontWindright = Polygon([(bor[0], 0.22, bor[4]),
                          (bor[1], 0.22, bor[5]),
                          (bor[1], bor[3], bor[5]),
                          (bor[0], bor[3], bor[4])])

# расчёт границ теней боковых окон для кажого источника поля
min_nt = Point(0.5 * width, sbor[3]).distance(Point(xp_nt, h_nt))  # луч нижней границы тени от НТ
max_nt = Point(0.5 * width, sbor[2]).distance(Point(xp_nt, h_nt))  # луч верхней границы тени от НТ

min_kp = Point(0.5 * width, sbor[3]).distance(Point(xp_kp, h_kp))  # далее аналогично для остальных проводов
max_kp = Point(0.5 * width, sbor[2]).distance(Point(xp_kp, h_kp))

min_up = Point(-0.5 * width, sbor[3]).distance(Point(xp_up, h_up))
max_up = Point(-0.5 * width, sbor[2]).distance(Point(xp_up, h_up))

min_nt2 = Point(0.5 * width, sbor[3]).distance(Point(xp_nt2 + xp_mid, h_nt))
max_nt2 = Point(0.5 * width, sbor[2]).distance(Point(xp_nt2 + xp_mid, h_nt))

min_kp2 = Point(0.5 * width, sbor[3]).distance(Point(xp_kp2 + xp_mid, h_kp))
max_kp2 = Point(0.5 * width, sbor[2]).distance(Point(xp_kp2 + xp_mid, h_kp))

min_up2 = Point(0.5 * width, sbor[3]).distance(Point(xp_up2 + xp_mid, h_up))
max_up2 = Point(0.5 * width, sbor[2]).distance(Point(xp_up2 + xp_mid, h_up))

# ЭКРАН
# стекло - высчитываем d для подсчёта энергии преломлённой волны
e1 = 1
e2 = 4
mu1 = 1
mu2 = 0.99

n1 = (e1 * mu1) ** 0.5
n2 = (e2 * mu2) ** 0.5
k_glass = ((n1 - n2) / (n1 + n2)) ** 2
d_glass = 1 - k_glass


# РАСЧЁТЫ


# по теореме Пифагора расчёт значения вектора из составляющих х и y
def mix(h_x, h_zz):
    return (h_x ** 2 + h_zz ** 2) ** 0.5


# магнитное поле гармоники f для заданной координаты x и z
def magnetic_calc(x_m, z_m, f_m, reflect=False):
    # общая сила тока гармоники
    I_h = I * harm.get(f_m)[0]

    # сила тока по проводам
    Ikp = 0.41 * I_h
    Int = 0.20 * I_h
    Iup = 0.39 * I_h

    # если отражение идёт от стекла, магнитная составляющая отражается - корректируем координаты
    if reflect:
        if abs(x_m) < bor[3] and z_m > floor + height:  # лобовые
            z_m = 2 * (height + floor) - z_m
        elif z_m > sbor[2] and z_m < sbor[3] and x_m < -.5*width:  # левое боковое
            x_m = -width - x_m
        elif z_m > sbor[2] and z_m < sbor[3] and x_m < .5 * width:  # правое боковое
            x_m = width - x_m
        else:
            return [0, 0, 0, 0, 0, 0]

    # расчёт x и z составляющих магнитного поля от правого рельса для КП
    x = x_m - xp_kp
    h1xkp = Ikp / (4 * pi) * (
            -z_m / ((x + xp) ** 2 + z_m ** 2) + (z_m - h_kp) / (x ** 2 + (h_kp - z_m) ** 2))
    h1zkp = Ikp / (4 * pi) * (x + xp) * (
            1 / ((x + xp) ** 2 + z_m ** 2) - 1 / (x ** 2 + (h_kp - z_m) ** 2))
    # сумма (по т.Пифагора) векторов x и z
    h1kp = mix(h1xkp, h1zkp)
    # расчёт x и z составляющих магнитного поля от левого рельса для КП
    x = x_m - 2 * xp - xp_kp
    h2xkp = Ikp / (4 * pi) * (
            -z_m / ((x + xp) ** 2 + z_m ** 2) + (z_m - h_kp) / ((x + 2 * xp) ** 2 + (h_kp - z_m) ** 2))
    h2zkp = Ikp / (4 * pi) * (x + xp) * (
            1 / ((x + xp) ** 2 + z_m ** 2) - 1 / ((x + 2 * xp) ** 2 + (h_kp - z_m) ** 2))
    # сумма (по т.Пифагора) векторов x и z    
    h2kp = mix(h2xkp, h2zkp)
    # суммарное поле двух рельс    
    hkp = h1kp + h2kp

    # далее аналогично для остальных проводов:
    # НТ
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

    # УП
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

    # КП2
    x = x_m - (xp_kp2 + xp_mid)
    h1xkp_2 = Ikp / (4 * pi) * (
            -z_m / ((x + xp) ** 2 + z_m ** 2) + (z_m - h_kp) / (x ** 2 + (h_kp - z_m) ** 2))
    h1zkp_2 = Ikp / (4 * pi) * (x + xp) * (
            1 / ((x + xp) ** 2 + z_m ** 2) - 1 / (x ** 2 + (h_kp - z_m) ** 2))
    h1kp_2 = mix(h1xkp_2, h1zkp_2)
    x = x_m - 2 * xp - (xp_kp2 + xp_mid)
    h2xkp_2 = Ikp / (4 * pi) * (
            -z_m / ((x + xp) ** 2 + z_m ** 2) + (z_m - h_kp) / ((x + 2 * xp) ** 2 + (h_kp - z_m) ** 2))
    h2zkp_2 = Ikp / (4 * pi) * (x + xp) * (
            1 / ((x + xp) ** 2 + z_m ** 2) - 1 / ((x + 2 * xp) ** 2 + (h_kp - z_m) ** 2))
    h2kp_2 = mix(h2xkp_2, h2zkp_2)
    hkp_scd = h1kp_2 + h2kp_2

    # НТ2
    x = x_m - (xp_nt2 + xp_mid)
    h1xnt_2 = Int / (4 * pi) * (
            -z_m / ((x + xp) ** 2 + z_m ** 2) + (z_m - h_nt) / (x ** 2 + (h_nt - z_m) ** 2))
    h1znt_2 = Int / (4 * pi) * (x + xp) * (
            1 / ((x + xp) ** 2 + z_m ** 2) - 1 / (x ** 2 + (h_nt - z_m) ** 2))
    h1nt_2 = mix(h1xnt_2, h1znt_2)
    x = x_m - 2 * xp - (xp_nt2 + xp_mid)
    h2xnt_2 = Int / (4 * pi) * (
            -z_m / ((x + xp) ** 2 + z_m ** 2) + (z_m - h_nt) / ((x + 2 * xp) ** 2 + (h_nt - z_m) ** 2))
    h2znt_2 = Int / (4 * pi) * (x + xp) * (
            1 / ((x + xp) ** 2 + z_m ** 2) - 1 / ((x + 2 * xp) ** 2 + (h_nt - z_m) ** 2))
    h2nt_2 = mix(h2xnt_2, h2znt_2)
    hnt_scd = h1nt_2 + h2nt_2

    # УП2
    x = x_m - (xp_up2 + xp_mid)
    x2 = -xp + xp_up2
    h1xup_2 = Iup / (4 * pi) * (
            -z_m / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) + (z_m - h_up) / (x ** 2 + (h_up - z_m) ** 2))
    h1zup_2 = Iup / (4 * pi) * (x2 + 2 * xp + x) * (
            1 / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) - 1 / (x ** 2 + (h_up - z_m) ** 2))
    h1up_2 = mix(h1xup_2, h1zup_2)
    x = x_m - (xp_up2 + xp_mid) - 2 * xp
    x2 = -xp + xp_up2
    h2xup_2 = Iup / (4 * pi) * (
            -z_m / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) + (z_m - h_up) / ((x + 2 * xp) ** 2 + (h_up - z_m) ** 2))
    h2zup_2 = Iup / (4 * pi) * (
            (x2 + 2 * xp + x) / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) - (x + 2 * xp) / (
            (x + 2 * xp) ** 2 + (h_up - z_m) ** 2))
    h2up_2 = mix(h2xup_2, h2zup_2)
    hup_sec = h1up_2 + h2up_2

    # результат выполнения этой функции - значения магнитных полей КП, НТ, УП для выбранной гармоники
    return [hkp, hnt, hup, hkp_scd, hnt_scd, hup_sec]


# расчёт электрического поля для гармоники f в точке x, z
def electric_calc(x_e, z_e, f_e, reflect=False):
    U_h = U * harm.get(f_e)[1]

    if reflect:  # если считаем отражённое электрическое поле, корректируем координаты для подсчёта поля мнимого провода
        if abs(x_e) < 0.5 * width and z_e > height + floor:  # отражение вверх
            z_e = 2 * (height + floor) - z_e
        elif abs(x_e) < 0.5 * width and z_e < gr_floor:  # отражение вниз
            z_e = 2 * gr_floor - z_e
        elif x_e < -.5 * width and z_e < height + floor and z_e > gr_floor:  # отражение влево
            x_e = -width - x_e
        elif x_e > .5 * width and z_e < height + floor and z_e > gr_floor:  # отражение вправо
            x_e = width - x_e
        elif x_e > .5 * width and z_e > height + floor:  # верхний правый угол
            z_e = 2 * (height + floor) - z_e
            x_e = width - x_e
        elif x_e > .5 * width and z_e < gr_floor:  # нижний правый угол
            z_e = 2 * gr_floor - z_e
            x_e = width - x_e
        elif x_e < -.5 * width and z_e > height + floor:  # верхний левый угол
            z_e = 2 * (height + floor) - z_e
            x_e = -width - x_e
        elif x_e < -.5 * width and z_e < gr_floor:  # нижний левый угол
            z_e = 2 * gr_floor - z_e
            x_e = -width - x_e
        else:
            return [0, 0, 0, 0, 0, 0]

    ekp = U_h * log(1 + 4 * h_nt * z_e / ((x_e - xp_nt) ** 2 + (h_nt - z_e) ** 2)) / (2 * z_e * log(2 * h_nt / d_nt))
    ent = U_h * log(1 + 4 * h_kp * z_e / ((x_e - xp_kp) ** 2 + (h_kp - z_e) ** 2)) / (2 * z_e * log(2 * h_kp / d_kp))
    eup = U_h * log(1 + 4 * h_up * z_e / ((x_e - xp_up) ** 2 + (h_up - z_e) ** 2)) / (2 * z_e * log(2 * h_up / d_up))

    ekp_scd = U_h * log(1 + 4 * h_nt * z_e / ((x_e - xp_nt2 - xp_mid) ** 2 + (h_nt - z_e) ** 2)) / (
                2 * z_e * log(2 * h_nt / d_nt))
    ent_scd = U_h * log(1 + 4 * h_kp * z_e / ((x_e - xp_kp2 - xp_mid) ** 2 + (h_kp - z_e) ** 2)) / (
                2 * z_e * log(2 * h_kp / d_kp))
    eup_scd = U_h * log(1 + 4 * h_up * z_e / ((x_e - xp_up2 - xp_mid) ** 2 + (h_up - z_e) ** 2)) / (
                2 * z_e * log(2 * h_up / d_up))

    return [ekp, ent, eup, ekp_scd, ent_scd, eup_scd]


# суммироввание всех полей для каждой точки:
def full_field(res_en):
    sum_h, sum_e, sum_g = 0, 0, 0
    for en in res_en[0].values():
        sum_h += sum(en[0])  # магнитная составляющая
        sum_e += sum(en[1])  # электрическая составляющая
        # энергия
        sum_g += en[0][0] * en[1][0] + en[0][1] * en[1][1] + en[0][2] * en[1][2] + \
                 en[0][3] * en[1][3] + en[0][4] * en[1][4] + en[0][5] * en[1][5]
    return [sum_h, sum_e, sum_g]


#  расчёт экрана переменного поля
def ekran(en, reflect=False):
    x, y, z = en[1]  # координаты точки

    if reflect:  # расчёт для отражённого поля: где отразилось от стекла, поле имеет меньшую интенсивность
        if (abs(y) < bor[3] and z > floor + height) or \
                (z > sbor[2] and z < sbor[3] and abs(y) > .5 * width):
            for f in en[0].keys():
                en[0][f][0][0] *= k_glass
                en[0][f][1][0] *= k_glass
        return en

    # расстояние от текущей точки до КТ и НТ - для расчёта лобовых окон
    kppth = LineString([(x, y, z), (x, xp_kp, h_kp)])
    ntpth = LineString([(x, y, z), (x, xp_nt, h_nt)])
    # проверяем, попадает ли лобовое окно по направлению от текущей точки до КТ, НТ
    kp_pass = kppth.intersects(frontWindleft) or kppth.intersects(frontWindright)
    nt_pass = ntpth.intersects(frontWindleft) or ntpth.intersects(frontWindright)

    # для каждого провода проверяем, попадает ли текущая точка в тень от бокового окна или нет
    kp_dist = Point(y, z).distance(Point(xp_kp, h_kp))  # направление от точки до провода
    # есть ли на пути этого направления окно
    # для КП и НТ - учитываем значение для лобового стекла логическим сложением
    kp_pass |= (kp_dist >= min_kp) and (kp_dist <= max_kp) and (x >= sbor[0]) and (x <= sbor[1])

    nt_dist = Point(y, z).distance(Point(xp_nt, h_nt))
    nt_pass |= (nt_dist >= min_nt) and (nt_dist <= max_nt) and (x >= sbor[0]) and (x <= sbor[1])

    up_dist = Point(y, z).distance(Point(xp_up, h_up))
    up_pass = (up_dist >= min_up) and (up_dist <= max_up) and (x >= sbor[0]) and (x <= sbor[1])

    kp_sec_d = Point(y, z).distance(Point(xp_kp2 + xp_mid, h_kp))
    kp_sec_p = (kp_sec_d >= min_kp2) and (kp_sec_d <= max_kp2) and (x >= sbor[0]) and (x <= sbor[1])

    nt_sec_d = Point(y, z).distance(Point(xp_nt2 + xp_mid, h_nt))
    nt_sec_p = (nt_sec_d >= min_nt2) and (nt_sec_d <= max_nt2) and (x >= sbor[0]) and (x <= sbor[1])

    up_sec_d = Point(y, z).distance(Point(xp_up2 + xp_mid, h_up))
    up_sec_p = (up_sec_d >= min_up2) and (up_sec_d <= max_up2) and (x >= sbor[0]) and (x <= sbor[1])

    # для каждой точки внутри кабины проверяем, проходит ли для неё какое-либо поле через стекло
    # сталь: электрическое поле полностью отражается, магнитное полностью затухает
    # стекло: и электрическое, и магнитное домножаются на d_glass по формуле:
    # Эпрел = Эпад*d = (ExH)*d = E*d x H*d
    if (abs(y) <= 0.5 * width) and (z >= gr_floor) and (z <= floor + height):
        # внутри кабины
        if kp_pass:
            # поле КП через стекло
            for f in en[0].keys():
                en[0][f][0][0] *= d_glass
                en[0][f][1][0] *= d_glass
        if nt_pass:
            # поле НТ через стекло
            for f in en[0].keys():
                en[0][f][0][1] *= d_glass
                en[0][f][1][1] *= d_glass
        if up_pass:
            # поле УП через стекло
            for f in en[0].keys():
                en[0][f][0][2] *= d_glass
                en[0][f][1][2] *= d_glass
        if kp_sec_p:
            # поле КП второго пути через стекло
            for f in en[0].keys():
                en[0][f][0][3] *= d_glass
                en[0][f][1][3] *= d_glass
        if nt_sec_p:
            # поле НТ второго пути через стекло
            for f in en[0].keys():
                en[0][f][0][4] *= d_glass
                en[0][f][1][4] *= d_glass
        if up_sec_p:
            # поле УП второго пути через стекло
            for f in en[0].keys():
                en[0][f][0][5] *= d_glass
                en[0][f][1][5] *= d_glass
        if not (kp_pass or nt_pass or up_pass or kp_sec_p or nt_sec_p or up_sec_p):
            # если ни через одно стекло не проходит, значит тут сталь, т.е. поле нулевое
            for f in en[0].keys():
                en[0][f][0] = [1, 1, 1, 1, 1, 1]
                en[0][f][1] = [1, 1, 1, 1, 1, 1]

    return en


# ГРАФИКА И ВЫВОД

# сохранение картинки в файл
def show(name):
    mng = plt.get_current_fig_manager()  # захват изображения 
    # mng.window.state('zoomed')  # вывод изображения на весь экран если граф.оболочка это поддерживает
    plt.savefig(f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{name}.png")
    # сохранение картинки в файл дата_время_название.png в папку со скриптом


# рисование линий кабины вид спереди
def fr_kab_lines(star=False):
    ln_ = '--'  # стиль линии

    cl_ = 'royalblue'  # окна
    plt.hlines(bor[4], bor[2], bor[3], colors=cl_, linestyles=ln_)
    plt.hlines(bor[5], bor[2], bor[3], colors=cl_, linestyles=ln_)
    plt.vlines(bor[2], bor[4], bor[5], colors=cl_, linestyles=ln_)
    plt.vlines(bor[3], bor[4], bor[5], colors=cl_, linestyles=ln_)
    plt.vlines(0, bor[4], bor[5], colors=cl_, linestyles=ln_)
    # дворники
    plt.plot(np.array([bor[2] + .1, bor[2] + 0.6]),
             np.array([bor[4] + .1, bor[5] + .1]), c=cl_, linestyle=ln_)
    plt.plot(np.array([bor[3] - .1, bor[3] - 0.6]),
             np.array([bor[4] + .1, bor[5] + .1]), c=cl_, linestyle=ln_)

    if star:
        cl_ = 'red'  # полосы и зезда
        plt.hlines(bor[4] - .3, -0.5 * width, 0.5 * width, colors=cl_, linestyles='solid', lw=3)
        plt.hlines(floor + .3, -0.5 * width, 0.5 * width, colors=cl_, linestyles='solid', lw=3)
        plt.hlines(1.4, -0.5 * width, 0.5 * width, colors=cl_, linestyles='solid', lw=4)
        plt.scatter(0, 2.7, s=200, marker='*', color=cl_)

        # пантограф
        cl_ = 'forestgreen'
        for i in range(1, 6):
            plt.hlines(height + floor + .3 + i * .08, -.4, -.7, colors=cl_, linestyles='solid')
            plt.hlines(height + floor + .3 + i * .08, .4, .7, colors=cl_, linestyles='solid')
        h = height + floor + .3 + 6 * .08
        plt.plot(np.array([-.8, -.8, .8, .8, -.8]),
                 np.array([h, h_kp - .1, h_kp - .1, h, h]),
                 c=cl_, linestyle=ln_)

    cl_ = 'forestgreen'  # очертания кабины
    plt.hlines(height + floor, -0.5 * width, 0.5 * width, colors=cl_, linestyles=ln_)
    plt.hlines(floor + gr_floor, -0.5 * width, 0.5 * width, colors=cl_, linestyles=ln_)
    plt.hlines(gr_floor, -0.5 * width, 0.5 * width, colors=cl_, linestyles=ln_)
    plt.vlines(-0.5 * width, gr_floor, height + floor, colors=cl_, linestyles=ln_)
    plt.vlines(0.5 * width, gr_floor, height + floor, colors=cl_, linestyles=ln_)

    # низ
    plt.plot(np.array([-.5 * width + .1, -.5 * width + .4, .5 * width - .4, .5 * width - .1]),
             np.array([gr_floor, .4, .4, gr_floor]), c=cl_, linestyle=ln_)
    delta = (width - .2) / 6
    for i in range(1, 6):
        plt.vlines(-.5 * width + .1 + delta * i, gr_floor, .6, colors=cl_, linestyles=ln_)

    # головная фара и крыша
    bj1 = ptch.Arc((0, floor + height), width, .8, theta1=0, theta2=180, color=cl_, linestyle=ln_)
    bj2 = ptch.Circle((0, floor + height + 0.3), 0.2, color=cl_, linestyle=ln_, fill=None)

    for bj in [bj1, bj2]:
        plt.gca().add_artist(bj)


# рисование линий внутри кабины
def kab_lines_front():
    d = 0.13
    cl = 'blue'
    plt.hlines(z_chair, y_chel - d, y_chel + d, colors=cl, linestyles='--')
    plt.hlines(z_chair, -y_chel - d, -y_chel + d, colors=cl, linestyles='--')
    plt.hlines(z_chair - 0.05, y_chel - d, y_chel + d, colors=cl, linestyles='--')
    plt.hlines(z_chair - 0.05, -y_chel - d, -y_chel + d, colors=cl, linestyles='--')

    plt.vlines(y_chel - d, z_chair, z_chair - 0.05, colors=cl, linestyles='--')
    plt.vlines(y_chel + d, z_chair, z_chair - 0.05, colors=cl, linestyles='--')
    plt.vlines(-y_chel - d, z_chair, z_chair - 0.05, colors=cl, linestyles='--')
    plt.vlines(-y_chel + d, z_chair, z_chair - 0.05, colors=cl, linestyles='--')

    d = 0.12
    plt.hlines(z_chair + 0.05, y_chel - d, y_chel + d, colors=cl, linestyles='--')
    plt.hlines(z_chair + 0.05 + 2 * d, y_chel - d, y_chel + d, colors=cl, linestyles='--')
    plt.hlines(z_chair + 0.05, -y_chel - d, -y_chel + d, colors=cl, linestyles='--')
    plt.hlines(z_chair + 0.05 + 2 * d, -y_chel - d, -y_chel + d, colors=cl, linestyles='--')

    plt.vlines(y_chel - d, z_chair + 0.05, z_chair + 0.05 + 2 * d, colors=cl, linestyles='--')
    plt.vlines(y_chel + d, z_chair + 0.05, z_chair + 0.05 + 2 * d, colors=cl, linestyles='--')
    plt.vlines(-y_chel - d, z_chair + 0.05, z_chair + 0.05 + 2 * d, colors=cl, linestyles='--')
    plt.vlines(-y_chel + d, z_chair + 0.05, z_chair + 0.05 + 2 * d, colors=cl, linestyles='--')


# рисование линий кабины вид сверху
def kab_lines_up():
    d = 0.12
    cl = 'blue'
    plt.hlines(y_chel - d, x_chel - d, x_chel + d, colors=cl, linestyles='--')
    plt.hlines(y_chel + d, x_chel - d, x_chel + d, colors=cl, linestyles='--')
    plt.hlines(-y_chel - d, x_chel - d, x_chel + d, colors=cl, linestyles='--')
    plt.hlines(-y_chel + d, x_chel - d, x_chel + d, colors=cl, linestyles='--')
    plt.vlines(x_chel - d, y_chel - d, y_chel + d, colors=cl, linestyles='--')
    plt.vlines(x_chel + d, y_chel - d, y_chel + d, colors=cl, linestyles='--')
    plt.vlines(x_chel - d, -y_chel - d, -y_chel + d, colors=cl, linestyles='--')
    plt.vlines(x_chel + d, -y_chel - d, -y_chel + d, colors=cl, linestyles='--')

    plt.hlines(y_chel - d, x_chel + d + 0.05, x_chel + d + 0.10, colors=cl, linestyles='--')
    plt.hlines(y_chel + d, x_chel + d + 0.05, x_chel + d + 0.10, colors=cl, linestyles='--')
    plt.hlines(-y_chel - d, x_chel + d + 0.05, x_chel + d + 0.10, colors=cl, linestyles='--')
    plt.hlines(-y_chel + d, x_chel + d + 0.05, x_chel + d + 0.10, colors=cl, linestyles='--')
    plt.vlines(x_chel + d + 0.05, y_chel - d, y_chel + d, colors=cl, linestyles='--')
    plt.vlines(x_chel + d + 0.10, y_chel - d, y_chel + d, colors=cl, linestyles='--')
    plt.vlines(x_chel + d + 0.05, -y_chel - d, -y_chel + d, colors=cl, linestyles='--')
    plt.vlines(x_chel + d + 0.10, -y_chel - d, -y_chel + d, colors=cl, linestyles='--')

    plt.vlines(bor[0], bor[2], -0.22, colors='white', linestyles='--')
    plt.vlines(bor[1], bor[2], -0.22, colors='white', linestyles='--')
    plt.hlines(bor[2], bor[0], bor[1], colors='white', linestyles='--')
    plt.hlines(-0.22, bor[0], bor[1], colors='white', linestyles='--')

    plt.vlines(bor[0], 0.22, bor[3], colors='white', linestyles='--')
    plt.vlines(bor[1], 0.22, bor[3], colors='white', linestyles='--')
    plt.hlines(0.22, bor[0], bor[1], colors='white', linestyles='--')
    plt.hlines(bor[3], bor[0], bor[1], colors='white', linestyles='--')

    cl = 'black'
    plt.plot(np.array([0.01, bor[0]]), np.array([0, bor[2]]), c=cl, linestyle='--')
    plt.plot(np.array([0.01, bor[0]]), np.array([0, bor[3]]), c=cl, linestyle='--')

    plt.hlines(0.5 * width - 0.01, 0, length, colors=cl, linestyles='--')
    plt.hlines(-0.5 * width + 0.01, 0, length, colors=cl, linestyles='--')
    plt.vlines(0.01, 0.5 * width, -0.5 * width, colors=cl, linestyles='--')
    plt.vlines(length - 0.01, 0.5 * width, -0.5 * width, colors=cl, linestyles='--')


# построение вида сверху без электровоза
def visual_up():
    print('График строится..................')

    # границы графика
    Xmin = -0.5
    Xmax = length + 0.5
    Ymin = xp_up * 1.15
    Ymax = xp_mid + abs(Ymin)

    # разбиение по точкам
    x = np.linspace(Xmin, Xmax, dis)
    y = np.linspace(Ymin, Ymax, dis)

    # расчёт значений полей для каждой точки графика
    every_f = [[[{fr: [magnetic_calc(y_, z_graph, fr), electric_calc(y_, z_graph, fr)] for fr in harm.keys()},
                 [x_, y_, z_graph]] for x_ in x] for y_ in y]

    # применяем экран и считаем итоговое значение для каждой точки
    summar = [[full_field(x_el) for x_el in y_list] for y_list in every_f]

    # формируем массив значений на магнитную, электрическую составляющую и энергию
    magnetic = [[x_el[0] for x_el in y_list] for y_list in summar]
    electric = [[x_el[1] for x_el in y_list] for y_list in summar]
    energy = [[x_el[2] for x_el in y_list] for y_list in summar]

    # общая функция отрисовки графика
    def do_graph(content, name_, x_lb='Ось x, метры', y_lb='Ось y, метры'):
        # создаём объект точек графика
        ct = plt.contour(x, y, content, alpha=0.75, colors='black', linestyles='dotted', levels=5)
        # создаём линии уровней из объекта точек        
        plt.clabel(ct, fontsize=10)
        # отрисовка
        plt.imshow(content, extent=[Xmin, Xmax, Ymax, Ymin], cmap='YlOrRd', alpha=0.95)
        # раскраска        
        plt.colorbar()

        # рисование и подпись проводов
        for delta_y in [xp_kp, xp_up, xp_nt, xp_kp2 + xp_mid, xp_nt2 + xp_mid, xp_up2 + xp_mid]:
            plt.hlines(delta_y, Xmin, Xmax, color='black', linewidth=2)
        plt.text(0.1, xp_kp + 0.05, 'КП', color='black')
        plt.text(1, xp_nt - 0.3, 'НТ', color='black')
        plt.text(0.1, xp_up + 0.05, 'УП', color='black')
        plt.text(0.1, xp_kp2 + xp_mid + 0.05, 'КП2', color='black')
        plt.text(1, xp_nt2 + xp_mid - 0.3, 'НТ2', color='black')
        plt.text(0.1, xp_up2 + xp_mid + 0.05, 'УП2', color='black')

        # рисование очертания поезда
        plt.hlines(0.5 * width, 0, length, colors='red', linestyles='--')
        plt.hlines(-0.5 * width, 0, length, colors='red', linestyles='--')
        plt.vlines(0, -0.5 * width, 0.5 * width, colors='red', linestyles='--')
        plt.vlines(length, -0.5 * width, 0.5 * width, colors='red', linestyles='--')

        # название осей 
        plt.xlabel(x_lb)
        plt.ylabel(y_lb)

        plt.title(name_)

    # отрисовка по очереди магнитного, электрического и энергии
    global gph_num
    gph_num += 1
    plt.figure(gph_num)
    name = 'Контактная сеть вид сверху (без электровоза)'
    plt.subplot(1, 3, 1)
    do_graph(magnetic, 'Магнитное', x_lb='Ось x, метры', y_lb='Ось y, метры')
    plt.subplot(1, 3, 2)
    do_graph(electric, 'Электрическое', x_lb='Ось x, метры', y_lb='Ось y, метры')
    plt.subplot(1, 3, 3)
    do_graph(energy, 'Энергия', x_lb='Ось x, метры', y_lb='Ось y, метры')
    plt.suptitle(name)
    show(name)

    print('График построен.')

    return every_f  # возвращаем поле для перерасчёта с локомотивом


# вывод вида спереди без электровоза
def visual_front():
    print('График строится..................')

    # границы графика
    Ymax = xp_up2 * 1.2 + xp_mid
    Ymin = xp_up * 1.2
    Zmax = 0.1
    Zmin = max(h_kp, h_nt, h_up) * 1.1

    # разбиение на точки
    y = np.linspace(Ymin, Ymax, dis)
    z = np.linspace(Zmin, Zmax, dis)

    # расчёт значений полей для каждой точки графика
    every_f = [[[{fr: [magnetic_calc(y_, z_, fr), electric_calc(y_, z_, fr)] for fr in harm.keys()},
                 [x_chel, y_, z_]] for y_ in y] for z_ in z]

    # считаем итоговое значение для каждой точки
    summar = [[full_field(x_el)[2] for x_el in y_list] for y_list in every_f]

    def graph_do(znach, name_):
        # задаём уровни
        b = len(str(round(np.amax(znach))))  # высчитываем диапазон графика для правильного отображения линий уровня
        levels = [i * (10 ** j) for j in range(0, b) for i in [1, 2, 5, 7]]
        # создаём объект точек графика
        ct = plt.contour(y, z, znach, alpha=0.75, colors='black', linestyles='dotted',
                         levels=levels)
        # создаём линии уровней из объекта точек
        plt.clabel(ct, fontsize=10)
        # отрисовка
        plt.imshow(znach, extent=[Ymin, Ymax, Zmax, Zmin], cmap=cmap, alpha=0.95, norm=colors.LogNorm())
        # раскраска
        plt.colorbar()

        # названия проводов
        plt.text(xp_kp, h_kp, 'КП', color='black', fontsize=14)
        plt.text(xp_up, h_up, 'УП', color='black', fontsize=14)
        plt.text(xp_nt, h_nt, 'НТ', color='black', fontsize=14)
        plt.text(xp_kp2 + xp_mid, h_kp, 'КП2', color='black', fontsize=14)
        plt.text(xp_up2 + xp_mid, h_up, 'УП2', color='black', fontsize=14)
        plt.text(xp_nt2 + xp_mid, h_nt, 'НТ2', color='black', fontsize=14)

        # очертания кабины
        fr_kab_lines(star=True)

        # название осей
        plt.xlabel('Ось y, метры')
        plt.ylabel('Ось z, метры')

        plt.title(name_)  # подпись названия
        show(name_)  # вывести и сохранить

    # вывод общей КС
    global gph_num
    gph_num += 1
    plt.figure(gph_num)
    graph_do(summar, 'Контактная сеть вид спереди (без электровоза) - Энергия')

    print('График построен.')
    return every_f  # возвращаем поле для перерасчёта с локомотивом


# вывод вида сверху с электровозом
def visual_up_locomotive(ext_f):
    print('График строится..................')
    # границы графика
    Xmin = 0
    Xmax = length
    Ymax = -0.5 * width
    Ymin = -Ymax

    # выборка области для отрисовки из уже посчитанного поля
    inside = [[full_field(ekran(el)) for el in y_list if (el[1][0] >= Xmin) and (el[1][0] <= Xmax)]
              for y_list in ext_f if abs(y_list[0][1][1]) <= 0.5 * width]

    # формируем массивы значений на магнитную, электрическую составляющую и энергию
    energy = [[x_el[2] for x_el in y_list] for y_list in inside]

    # общая функция отрисовки графика
    def graph_do(znach, name_, x_lb='', y_lb=''):
        # отрисовка
        plt.imshow(znach, extent=[Xmin, Xmax, Ymax, Ymin], cmap='YlOrRd', alpha=0.95, norm=colors.LogNorm())
        # раскраска          
        plt.colorbar()

        # название осей 
        plt.xlabel(x_lb)
        plt.ylabel(y_lb)

        plt.title(name_)  # подпись названия

    # отрисовка энергии
    global gph_num
    gph_num += 1
    plt.figure(gph_num)
    name = 'Кабина вид сверху (c экраном) - энергия'
    graph_do(energy, name, x_lb='Ось x, метры', )
    kab_lines_up()
    show(name)


def visual_front_locomotive(ext_f):
    print('График строится..................')

    # границы графика
    Ymin, Ymax = -0.6 * width, 0.6 * width
    Zmin, Zmax = floor + height + 1, 0.1

    # применяем экран
    ekran_ = [[ekran(y_el) for y_el in z_list if abs(y_el[1][1]) <= Ymax] for z_list in ext_f
              if z_list[0][1][2] < Zmin]

    # перевод значений посчитанных для каждой гармоники каждого провода в одно значение
    summar = [[full_field(x_el) for x_el in y_list] for y_list in ekran_]
    # выбор значений только энергии
    energy = [[x_el[2] for x_el in y_list] for y_list in summar]

    # разбиение по точкам
    y_ln = np.linspace(Ymin, Ymax, len(ekran_[0]))
    z_ln = np.linspace(Zmin, Zmax, len(ekran_))
    # находим координаты человека в массиве точек
    chel_y = np.where(y_ln == max([y_ for y_ in y_ln if y_ <= y_chel]))[0][0]
    chel_z = np.where(z_ln == max([z_ for z_ in z_ln if z_ <= z_chel]))[0][0]

    # общая функция отрисовки графика
    def graph_do(znach, name_, x_lb='', y_lb=''):
        # отрисовка        
        plt.imshow(znach, extent=[Ymin, Ymax, Zmax, Zmin], cmap=cmap, alpha=0.95, norm=colors.LogNorm())
        # раскраска    
        plt.colorbar()

        # очертания кабины
        fr_kab_lines()
        # название осей 
        plt.xlabel(x_lb)
        plt.ylabel(y_lb)

        plt.title(name_)  # подпись названия

    # отрисовка энергии поля
    global gph_num
    gph_num += 1
    plt.figure(gph_num)
    name = 'Кабина вид спереди (c экраном) - энергия'
    graph_do(energy, name, x_lb='Ось y, метры', )
    kab_lines_front()
    show(name)

    # отрисовка отрисовка энергии по гармоникам
    gph_num += 1
    plt.figure(gph_num)
    name = 'Гармоники вид спереди (экран и отражённое поле) - энергия'
    plt.title(name)
    i = 0
    chel_harm = {}
    # для каждой гармоники формируем массив точек на отрисовку + считаем воздействие в положении человека
    for fr in harm.keys():
        i += 1
        plt.subplot(3, 3, i)
        # считаем энергию для конкретной гармоники
        data = [[el[0][fr][0][0] * el[0][fr][1][0] +
                 el[0][fr][0][1] * el[0][fr][1][1] +
                 el[0][fr][0][2] * el[0][fr][1][2] +
                 el[0][fr][0][3] * el[0][fr][1][3] +
                 el[0][fr][0][4] * el[0][fr][1][4] +
                 el[0][fr][0][5] * el[0][fr][1][5]
                 for el in lst] for lst in ekran_]
        chel_harm[fr] = data[chel_z][chel_y]
        graph_do(data, '', y_lb=str(fr))
        kab_lines_front()
    plt.subplot(3, 3, 9)
    plt.bar(range(0, len(harm.keys())), chel_harm.values())
    plt.suptitle(name)
    show(name)

    # возвращаем значения для гармоник в координатах человека чтобы вывести в блоке статистики
    return chel_harm


# визуализируем вид спереди с учётом отражённой от экрана
def visual_front_ekran(ext_f):
    # чтобы не возникло проблем при вычитании поля КС и поля отражений,
    # получаем список точек графика из уже рассчитанного ранее поля КС
    y_ln = [el[1][1] for el in ext_f[0]]
    z_ln = [el[0][1][2] for el in ext_f]
    Ymin, Ymax = y_ln[0], y_ln[-1]
    Zmin, Zmax = z_ln[0], z_ln[-1]

    # посчёт отражённого поля
    reflect_f = [[[{fr: [magnetic_calc(y_, z_graph, fr, reflect=True),
                         electric_calc(y_, z_, fr, reflect=True)
                         ] for fr in harm.keys()},
                   [x_chel, y_, z_]] for y_ in y_ln] for z_ in z_ln]
    summar_reflect = np.array([[full_field(ekran(x_el, reflect=True))[2] for x_el in y_list] for y_list in reflect_f])
    # перевод в конечные значения внешнего поля с экраном
    summar_ext = np.array([[full_field(ekran(x_el))[2] for x_el in y_list] for y_list in ext_f])
    # вычитаем из поля внешнего поле отражённое
    summar = summar_ext - summar_reflect

    global gph_num
    gph_num += 1
    plt.figure(gph_num)
    name_ = 'Вид спереди (экран и отражённое поле) - Энергия'

    # задаём уровни
    b = len(str(round(np.amax(summar))))  # ручной подсчёт порядка диапазона для отображения линий уровня
    levels = [i*(10**j) for j in range(4, b) for i in [1, 2, 5, 7]]  # ограничиваем 4-ой степенью чтобы не было
                                                                     # артефактов на границе с экраном
    # создаём объект точек графика
    ct = plt.contour(y_ln, z_ln, summar, alpha=0.75, colors='black', linestyles='dotted',
                     levels=levels)
    # создаём линии уровней из объекта точек
    plt.clabel(ct, fontsize=10)
    # отрисовка
    plt.imshow(summar, extent=[Ymin, Ymax, Zmax, Zmin], cmap=cmap, alpha=0.95, norm=colors.LogNorm())
    # раскраска
    plt.colorbar()

    # названия проводов
    plt.text(xp_kp, h_kp, 'КП', color='black', fontsize=14)
    plt.text(xp_up, h_up, 'УП', color='black', fontsize=14)
    plt.text(xp_nt, h_nt, 'НТ', color='black', fontsize=14)
    plt.text(xp_kp2 + xp_mid, h_kp, 'КП2', color='black', fontsize=14)
    plt.text(xp_up2 + xp_mid, h_up, 'УП2', color='black', fontsize=14)
    plt.text(xp_nt2 + xp_mid, h_nt, 'НТ2', color='black', fontsize=14)

    # очертания кабины
    fr_kab_lines(star=True)

    # название осей
    plt.xlabel('Ось y, метры')
    plt.ylabel('Ось z, метры')

    plt.title(name_)  # подпись названия

    show(name_)  # вывести и сохранить


# ВЫВОД ПАРАМЕТРОВ
print('\nПараметры сети')
print(f'Высота КП: {h_kp} м')
print(f'Высота НЧ: {h_nt} м')
print(f'Высота УП: {h_up} м')
print(f'Напряжение: {U} Вольт')
print(f'Суммарный ток: {I} Ампер')
print(f'Высота среза: {z_graph} метров')

# ПОСТРОЕНИЕ ГРАФИКА

gph_num = 0
print('\nБез электровоза:')
print('\nВид сверху')
cont_f_up = visual_up()

print('\nВид спереди')
cont_f_front = visual_front()

print('\nКабина электровоза:')
print('\nВид сверху')
visual_up_locomotive(cont_f_up)

print('\nВид спереди')
chel_harm = visual_front_locomotive(cont_f_front)

print('\nВид спереди для отражённого поля')
visual_front_ekran(cont_f_front)


# РАСЧЁТ СТАТИСТИКИ

print('СТАТИСТИКА\n')

print('Гармоники энергии поля для человека:')
for f, znach in chel_harm.items():
    print(f, ': %.4f' % znach)

S = (a * b / 3600) ** 1 / 2
p = ti / 24  # статистическая вероятность воздействия

chel_f_per = [{fr: (magnetic_calc(y_chel, z_chel, fr), electric_calc(y_chel, z_chel, fr)) for fr in harm.keys()},
              (x_chel, y_chel, z_chel)]
no_ekran_per = full_field(chel_f_per)[2]
print('\nПеременное поле без экрана: %.4f' % no_ekran_per)

ekran_per = full_field(ekran(chel_f_per))[2]
print('Переменное поле с экраном %.4f' % ekran_per)
Dco = ekran_per * ti * S * p
Dpo = Dco / b
print('Удельная суточная доза поглощённой энергии: %.4f' % Dpo)

plt.show()
