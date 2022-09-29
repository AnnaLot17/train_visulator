from math import pi, log, exp, atan
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri
from datetime import datetime
import matplotlib.colors as colors
plt.style.use('seaborn-white')

'''
ОПИСАНИЕ

выполнение модуля начинается в 700-х строках, вызовом
visual_up_per() - построение вида сверху для перменного поля
visual_up_post() - построение вида сверху для постоянного поля
visual_front() - построение вида сбоку по срезам. В срезах перменное и постоянное поля считаются отдельно, но строятся
смешанно, поэтому разделения нет.
Читать код рекомендую начинать с вызова этих функций. Если читаете через Pycharm, то зажав Ctrl кликните по функции
чтобы перейти к её коду.
В начале кода приведены все константы и координаты в системе СИ (метры, амперы, вольты).
Затем - функции для вычисления поля: расчёт шин и оборудования, суммирование поля, расчёт экрана.
После - функции построения графиков: разбиение на теругольники, отрисовка линий, параметры вывода графиков.
После них - функции, формирующие отрисовку графиков по видам и срезам. В самом конце - рачёт статистики.
'''


# СТАТИСТИЧЕСКИЕ ДАННЫЕ
x_chel = 1  # положение человека по оси х
y_chel = 0.8  # положение человека по оси y
a = 1.75  # высота человека метры
b = 80  # масса человека килограммы
ti = 1  # длительность пребывания работника на рабочем месте, часов

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

# ДАННЫЕ О ЛОКОМОТИВЕ

length = 1.3  # длина кабины
all_length = 15.2  # длина всего локомотива
width = 2.8  # ширина кабины
height = 2.6  # высота кабины
floor = 2  # расстояние от земли до дна кабины
chel = floor + 0.7  # где находится человек

metal_mu = 1000  # относительная магнитная проницаемость стали
metal_t = 0.0025  # толщина стали
metal_sigma = 10 ** 7  # удельная проводимость стали
v_kab = all_length * width * height
metal_r = (v_kab * 3 / 4 / pi) ** 1 / 3
glass_r = (2.86 * 3 / 4 / pi) ** 1 / 3
kh_metal = {str(frq): 10 * log(1 + (metal_sigma * 2 * pi * frq * metal_mu * metal_r * metal_t / 2) ** 2, 10)
                 for frq in harm.keys()}
ke_metal = 20 * log(60 * pi * metal_t * metal_sigma, 10)


# УЗЛЫ ОБОРУДОВАНИЯ ИХ ТОКИ И НАПРЯЖЕНИЯ
# x - от стенки кабины
# y - от нижнежнего края
# z - от пола

x_tt = 4.500  # тяговый трансформатор
y_tt = 0.600 + 0.600 - 0.265
l_tt = 1.420
l_sh_tt = 1.1
w_tt = 0.7
h_tt = 1.265
d_tt = 0.3

x_gk = x_tt + 0.132  # главный контроллер
y_gk = 0.600 + 0.600
z_gk = 0.5
l_gk = 1.155
w_gk = 0.880
h_gk = 0.735

x_mt = x_tt + 1.420  # медная труба
y_mt = y_tt

x_vu1 = 3.700  # выпрямительная установка
x_vu2 = 6.700
y_vu1 = 0.600 + 0.15 - 0.2
y_vu2 = y_vu1 + 1.4
l_vu = 1.120
h_vu = 0.4472
z_vu = 0.630
d_vu = 1.1
w_vu = 0.5

x_cp1 = x_vu1 + 1 - d_vu  # сглаживающий реактор
x_cp2 = x_vu2 - 1 + d_vu
y_cp = y_vu1 + 0.9
l_cp = 0.8
h_cp = 0.8
z_cp = 0.6
w_cp = -0.6

x_td1 = 0.9  # тяговый двигатель
x_td2 = x_td1 + 8
dy_td = 0.8
r_td = 0.604
l_td = 0.66
z_td = 1 - floor
kol_par = 1.5

I_mt = 550
U_mt = 27000

I_tt_gk = 1750
U_tt_gk = 1218

I_gk_vu = 3150
U_gk_vu = 1400

I_vu_cp = 3150
U_vu_cp = 1400

I_cp_td = 880
U_cp_td = 950

I_tt = 750
U_tt = 1218

I_gk = 1300
U_gk = 3000

I_vu = 1850
U_vu = 1500
n_vu = 7

I_cp = 3150
U_cp = 1400
n_cp = 1

I_td = 880
U_td = 1950
n_td = 5

# РАЗМЕРЫ И ДАННЫЕ ЭКРАНА

b_v = 2.6  # ширина экрана камеры, м
l_v = 12.7  # длина экрана камеры, м
h_v = 2.2  # высота экрана камеры, м
d_v = 0.001  # толщина сетки, м

b_z = 3.5  # ширина экрана кузова, м
l_z = 15.2  # длина экрана кузова, м
h_z = 2.8  # высота экрана кузова, м
d_z = 0.025  # толщина экрана кузова, м

Z0 = 377  # волновое сопротивление поля, Ом
ds = 0.001  # диаметр провода сетки, м
ro = 0.15  # удельное сопротивление материала сетки, Ом*м
s_ = 0.015  # шаг сетки, м

re_v = 0.62 * (b_v * l_v * h_v) ** (1 / 3)  # эквивалетныый радиус экрана камеры
re_z = 0.62 * (b_z * l_z * h_z) ** (1 / 3)  # эквивалетныый радиус экрана кабины

koef_ekr_h_setka, koef_ekr_e_setka = {}, {}
for fr in harm.keys():
    lam = 300000000 / fr  # длина волны
    Zh = Z0 * 2 * pi * re_v / lam
    ekr_h = 0.012 * (d_v * Zh / ro) ** 0.5 * (lam / re_v) ** (1 / 3) * exp(pi * ds / (s_ - ds))
    koef_ekr_h_setka[fr] = ekr_h

    delta = 0.016 / (fr ** 0.5)
    ekr_e = 60 * pi * 1 * delta / (ro * s_ * 2.83 * (ds ** 0.5)) * exp(ds / delta)
    koef_ekr_e_setka[fr] = ekr_e

koef_ekr_h_splosh_v = 1 + (0.66 * metal_mu * d_v / re_v)
koef_ekr_h_splosh_z = 1 + (0.66 * metal_mu * d_z / re_z)
koef_ekr_e_splosh = ke_metal

k_post_ekr_e_setka = 55.45 + 20 * log(ds ** 2 * metal_sigma / s_, 10)
k_post_ekr_h_setka = exp(pi * d_v / s_)


# ШИНЫ И ОБОРУДОВАНИЕ ДЛЯ РАСЧЁТОВ

# формат шин: [(координаты начала), (смещение конца относительно начала)]
# формат оборудования: [[x_нач, x_кон, y_нач, y_кон, z_нач, z_кон]]

mednaya_truba = [[(x_mt, y_mt, 0), (0, 0, height)]]

sh_tt_gk = [[(x_tt+l_tt/6*i, y_tt, 0), (0, 0, -1.5)] for i in range(0, 7)]

sh_gk_vu = [[(x_gk, y_gk + h_gk, z_gk), (0, 0.5, 0)],
            [(x_gk, y_gk + h_gk + 0.5, z_gk), (0, 0, 0.8)],
            [(x_gk, y_gk + h_gk + 0.5, z_gk + 0.8), (-0.8, 0, 0)],
            [(x_gk-0.8, y_gk + h_gk + 0.5, z_gk + 0.8), (0, -0.8, 0)],
            [(x_gk-0.8, y_gk + h_gk + 0.5 - 0.8, z_gk + 0.8), (0, 0, -1.6)],
            [(x_gk-0.8, y_gk + h_gk + 0.5 - 0.3, z_gk + 0.8 - 1.1), (-0.15, 0, 0)],
            [(x_gk-0.8, y_gk + h_gk + 0.5 - 0.3, z_gk + 0.8 - 1.6), (0, -1.4, 0)],
            [(x_gk-0.8, y_gk + h_gk + 0.5 - 0.3 - 1.4, z_gk + 0.8 - 1.6), (-0.15, 0, 0)],

            [(x_gk+l_gk, y_gk + h_gk, z_gk), (0, 0.5, 0)],
            [(x_gk+l_gk, y_gk + h_gk + 0.5, z_gk), (0, 0, 0.8)],
            [(x_gk+l_gk, y_gk + h_gk + 0.5, z_gk+0.8), (0.8, 0, 0)],
            [(x_gk+l_gk+0.8, y_gk + h_gk + 0.5, z_gk+0.8), (0, -0.8, 0)],
            [(x_gk+l_gk+0.8, y_gk + h_gk + 0.5 - 0.8, z_gk+0.8), (0, 0, -1.6)],
            [(x_gk+l_gk+0.8, y_gk + h_gk + 0.5 - 0.3, z_gk+0.8-1.1), (0.15, 0, 0)],
            [(x_gk+l_gk+0.8, y_gk + h_gk + 0.5 - 0.3, z_gk+0.8-1.6), (0, -1.4, 0)],
            [(x_gk+l_gk+0.8, y_gk + h_gk + 0.5 - 0.3-1.4, z_gk+0.8-1.6), (0.15, 0, 0)],
            ]

tt = [[x_tt, x_tt + l_tt, y_tt, y_tt + w_tt, 0.5, 0.5 - h_tt]]

gk = [[x_gk, x_gk + l_gk, y_gk, y_gk + w_gk, z_gk, z_gk + h_gk]]

# ПОСТОЯННОЕ

sh_vu_cp = [
            [(x_vu1, y_vu1 + 0.2, z_vu), (1, 0, 0)],
            [(x_vu1, y_vu2 + 0.2, z_vu), (1, 0, 0)],
            [(x_vu1 + 1, y_vu1 + 0.2, z_vu), (0, 1.4, 0)],
            [(x_vu1+1, y_vu1 + 0.83, z_vu), (-0.6, 0, 0)],

            [(x_vu2, y_vu1 + 0.2, z_vu), (-1, 0, 0)],
            [(x_vu2, y_vu2 + 0.2, z_vu), (-1, 0, 0)],
            [(x_vu2 - 1, y_vu1 + 0.2, z_vu), (0, 1.4, 0)],
            [(x_vu2-1, y_vu1 + 0.83, z_vu), (0.6, 0, 0)]]

sh_cp_td = [
            [(x_cp2 + l_cp, y_cp, 0), (0, 0, 1.9)],
            [(x_cp2 + l_cp, y_cp, 1.9), (0, 0, -0.8)],
            [(x_cp2 + l_cp, y_cp, 1.1), (2.5, 0, 0)],
            [(x_cp2 + l_cp+2.5, y_cp, 1.1), (0, 0, -1.7-0.5)],
            [(x_cp1 - l_cp, y_cp, 0), (0, 0, 1.9)],
            [(x_cp1 - l_cp, y_cp, 1.9), (0, 0, -0.8)],
            [(x_cp1 - l_cp, y_cp, 1.1), (-2.5, 0, 0)],
            [(x_cp1 - l_cp-2.5, y_cp, 1.1), (0, 0, -1.7-0.5)],
            ]

vu = [[x_vu1, x_vu1 - l_vu, y_vu1, y_vu1 + h_vu, z_vu, z_vu+w_vu],
      [x_vu1, x_vu1 - l_vu, y_vu2, y_vu2 + h_vu, z_vu, z_vu+w_vu],
      [x_vu2, x_vu2 + l_vu, y_vu1, y_vu1 + h_vu, z_vu, z_vu+w_vu],
      [x_vu2, x_vu2 + l_vu, y_vu2, y_vu2 + h_vu, z_vu, z_vu+w_vu]]

cp = [[x_cp1, x_cp1 - l_cp, y_cp - 0.5 * h_cp, y_cp + 0.5 * h_cp, z_cp, z_cp+w_cp],
      [x_cp2, x_cp2 + l_cp, y_cp - 0.5 * h_cp, y_cp + 0.5 * h_cp, z_cp, z_cp+w_cp]]


def radius(st, ed):
    return ((st[0] - ed[0]) ** 2 + (st[1] - ed[1]) ** 2 + (st[2] - ed[2]) ** 2) ** 0.5


def shina(shinas, v1arr, v2arr, v3, I, U, type_='FRONT', ver_='PER'):
    dc = 10

    sh_p = []
    for sh in shinas:
        if sh[1][0]:
            arr = np.linspace(sh[0][0], sh[0][0]+sh[1][0], dc)
            sh_p.extend([(x, sh[0][1], sh[0][2]) for x in arr])
        elif sh[1][1]:
            arr = np.linspace(sh[0][1], sh[0][1]+sh[1][1], dc)
            sh_p.extend([(sh[0][0], y, sh[0][2]) for y in arr])
        elif sh[1][2]:
            arr = np.linspace(sh[0][2], sh[0][2]+sh[1][2], dc)
            sh_p.extend([(sh[0][0], sh[0][1], z) for z in arr])

    sh_points = [(length+pp[0], -0.5*width+pp[1], floor+pp[2]) for pp in sh_p]

    def in_point(x_, y_, z_):
        r = 0
        for point in sh_points:
            r += 1 / radius((x_, y_, z_), point)

        if ver_ == 'PER':
            return [{f: [I * harm[f][0] * r / (2 * pi * len(sh_points)), U * harm[f][0] * r / len(sh_points)]
                    for f in harm.keys()}, (x_, y_, z_)]
        else:
            return [[I * r / (2 * pi * len(sh_points)), U * r / len(sh_points)], (x_, y_, z_)]

    if type_ == 'FRONT':
        res = [in_point(v3, y, z) for z in v2arr for y in v1arr]
    else:
        res = [in_point(x, y, v3) for y in v2arr for x in v1arr]

    return res


def oborud_per(element, v1arr, v2arr, v3, I, U, type_='FRONT'):
    ds = 6
    points = []
    for el in element:
        nodes_x = np.linspace(el[0], el[1], ds)
        nodes_y = np.linspace(el[2], el[3], ds)
        nodes_z = np.linspace(el[4], el[5], ds)
        points.extend([[length+x_, -0.5*width+y_, floor+z_] for z_ in nodes_z for y_ in nodes_y for x_ in nodes_x])

    x_cab = np.linspace(length, all_length, 40)
    y_cab = np.linspace(-0.5 * width, 0.5 * width, 40)
    minus = [[x_, y_] for y_ in y_cab for x_ in x_cab]

    l_ob = abs(element[0][1] - element[0][0])

    def in_point(x_, y_, z_):
        H_ob, E_ob = 0, 0
        for p in points:
            r = ((p[0] - x_) ** 2 + (p[1] - y_) ** 2 + (p[2] - z_) ** 2) ** 0.5
            H_ob += I / (pi * l_ob) * atan(l_ob / 2 / r)
            E_ob += U / r / len(points)

        for m in minus:
            r_m = ((m[0] - x_) ** 2 + (m[1] - y_) ** 2 + (floor - z_) ** 2) ** 0.5
            if r_m != 0:
                E_ob += U / r_m / len(minus)
        return [{f: [harm[f][0] * H_ob / len(points), harm[f][1] * E_ob] for f in harm.keys()}, (x_, y_, z_)]

    if type_ == 'FRONT':
        return [in_point(v3, y, z) for z in v2arr for y in v1arr]
    else:
        return [in_point(x, y, v3) for y in v2arr for x in v1arr]


def oborud_post(element, v1arr, v2arr, v3, I, U, n=1, type_='FRONT'):
    ds = 6
    points = []
    for el in element:
        nodes_x = np.linspace(el[0], el[1], ds)
        nodes_y = np.linspace(el[2], el[3], ds)
        nodes_z = np.linspace(el[4], el[5], ds)
        points.extend([[length+x_, -0.5*width+y_, floor+z_] for z_ in nodes_z for y_ in nodes_y for x_ in nodes_x])

    x_cab = np.linspace(length, all_length, 40)
    y_cab = np.linspace(-0.5 * width, 0.5 * width, 40)
    minus = [[x_, y_] for y_ in y_cab for x_ in x_cab]

    l_ob = abs(element[0][1] - element[0][0])

    def in_point(x_, y_, z_):
        H_ob, E_ob = 0, 0
        for p in points:
            r = ((p[0] - x_) ** 2 + (p[1] - y_) ** 2 + (p[2] - z_) ** 2) ** 0.5
            H_ob += I / (pi * l_ob) * atan(l_ob / 2 / r)
            E_ob += U / r / len(points)

        for m in minus:
            r_m = ((m[0] - x_) ** 2 + (m[1] - y_) ** 2 + (floor - z_) ** 2) ** 0.5
            if r_m != 0:
                E_ob += U / r_m / len(minus)
        return [[H_ob * n / len(points), E_ob], (x_, y_, z_)]

    if type_ == 'FRONT':
        return [in_point(v3, y, z) for z in v2arr for y in v1arr]
    else:
        return [in_point(x, y, v3) for y in v2arr for x in v1arr]


def oborud_ted(v1arr, v2arr, v3, I, U, n=n_td, type_='FRONT'):
    ds = 8
    nodes_x = [dx + 0.5 * r_td * np.cos(ap) for dx in [x_td1, x_td1+kol_par, x_td2, x_td2+kol_par]
               for ap in np.linspace(0, 2 * pi, ds)]
    nodes_z = [z_td + 0.5 * r_td * np.sin(ap) for ap in np.linspace(0, 2 * pi, ds)]
    nodes_y = [td - td_p for td in [dy_td, -dy_td] for td_p in np.linspace(-0.5 * l_td, 0.5 * l_td, 4)]
    points = [[x_, y_, z_] for z_ in nodes_z for y_ in nodes_y for x_ in nodes_x]

    x_cab = np.linspace(length, all_length, 40)
    y_cab = np.linspace(-0.5 * width, 0.5 * width, 40)
    minus = [[x_, y_] for y_ in y_cab for x_ in x_cab]

    def in_point(x_, y_, z_):
            H_ob, E_ob = 0, 0
            for p in points:
                r = ((p[0] - x_) ** 2 + (p[1] - y_) ** 2 + (p[2] - z_) ** 2) ** 0.5
                H_ob += I / (2 * pi)
                E_ob += U / r / len(points)

            for m in minus:
                r_m = ((m[0] - x_) ** 2 + (m[1] - y_) ** 2 + (floor - z_) ** 2) ** 0.5
                if r_m != 0:
                    E_ob += U / r_m / len(minus)

            h, e = H_ob * n / len(points), E_ob
            if z_ > floor:
                h /= koef_ekr_h_splosh_v
                e /= koef_ekr_e_splosh
            return [[h, e], (x_, y_, z_)]

    if type_ == 'FRONT':
        return [in_point(v3, y, z) for z in v2arr for y in v1arr]
    else:
        return [in_point(x, y, v3) for y in v2arr for x in v1arr]


def field_sum_per(*arg):
    def summ(f, i):
        sum_h, sum_e = 0, 0
        for el in arg:
            sum_h += el[i][0][f][0]
            sum_e += el[i][0][f][1]
        return [sum_h, sum_e]
    return [[{frq: summ(frq, i) for frq in harm.keys()}, arg[0][i][1]] for i in range(0, len(arg[0]))]


def field_sum_post(*arg):
    def summ(i):
        sum_h, sum_e = 0, 0
        for el in arg:
            sum_h += el[i][0][0]
            sum_e += el[i][0][1]
        return [sum_h, sum_e]
    return [[summ(i), arg[0][i][1]] for i in range(0, len(arg[0]))]


def full_energy(en):
    sum_h, sum_e = 0, 0
    for e in en.values():
        sum_h += e[0]
        sum_e += e[1]
    return [sum_h, sum_e]


def ekran(elm, tp='SETKA', ver='PER'):
    if elm[1][0] > length + 0.4:
        if ver == 'PER':
            eg = full_energy(elm[0])
            return eg[0] * eg[1]
        else:
            return elm[0][0] * elm[0][1]
    else:
        if tp == 'SETKA':
            if ver == 'PER':
                k_h = koef_ekr_h_setka
                k_e = koef_ekr_e_setka
            else:
                k_h = k_post_ekr_h_setka
                k_e = k_post_ekr_e_setka
        else:
            k_e = koef_ekr_e_splosh
            k_h = koef_ekr_h_splosh_v
        if elm[1][0] < length:
            k_e = {f: k_e[f]*koef_ekr_e_splosh for f in k_e.keys()}\
                if type(k_h) == dict else k_e * koef_ekr_e_splosh
            k_h = {f: k_h[f] * koef_ekr_h_splosh_z for f in k_h.keys()} \
                if type(k_h) == dict else k_h * koef_ekr_h_splosh_z

    if ver == 'PER':
        sum_h, sum_e = 0, 0
        for fq in harm.keys():
            sum_h += elm[0][fq][0] / k_h[fq] if type(k_h) == dict else elm[0][fq][0] / k_h
            sum_e += elm[0][fq][1] / k_e[fq] if type(k_e) == dict else elm[0][fq][1] / k_e
        return sum_h * sum_e
    else:
        return elm[0][0] / k_h * elm[0][1] / k_e


def do_draw(h_lines, v_lines, c, type_, nm=''):
    li = '--'
    if type_ == 'FRONT':
        for h in h_lines:
            plt.hlines(floor + h[0], -0.5 * width + h[1], -0.5 * width + h[2], colors=c, linestyles=li)
        for v in v_lines:
            plt.vlines(-0.5 * width + v[0], floor + v[1], floor + v[2], colors=c, linestyles=li)
    else:
        for h in h_lines:
            plt.hlines(-0.5 * width + h[0], length + h[1], length + h[2], colors=c, linestyles=li)
        for v in v_lines:
            plt.vlines(length + v[0], -0.5 * width + v[1], -0.5 * width + v[2], colors=c, linestyles=li)


def lines_oborud(oborud_, color, nm, type_='FRONT'):
    h_lines = []
    v_lines = []
    if type_ == 'FRONT':
        for ob in oborud_:
            h_lines.append([ob[4], ob[2], ob[3]])
            h_lines.append([ob[5], ob[2], ob[3]])
            v_lines.append([ob[2], ob[4], ob[5]])
            v_lines.append([ob[3], ob[4], ob[5]])
    else:
        for ob in oborud_:
            h_lines.append([ob[2], ob[0], ob[1]])
            h_lines.append([ob[3], ob[0], ob[1]])
            v_lines.append([ob[0], ob[2], ob[3]])
            v_lines.append([ob[1], ob[2], ob[3]])

    do_draw(h_lines, v_lines, color, type_, nm=nm)


def lines_shina(shina_, color, type_='FRONT'):
    h_lines = []
    v_lines = []
    if type_ == 'FRONT':
        for sh in shina_:
            if sh[1][1] != 0:
                h_lines.append([sh[0][2], sh[0][1], sh[0][1]+sh[1][1]])
            if sh[1][2] != 0:
                v_lines.append([sh[0][1], sh[0][2], sh[0][2]+sh[1][2]])
    else:
        for sh in shina_:
            if sh[1][0] != 0:
                h_lines.append([sh[0][1], sh[0][0], sh[0][0] + sh[1][0]])
            if sh[1][1] != 0:
                v_lines.append([sh[0][0], sh[0][1], sh[0][1] + sh[1][1]])

    do_draw(h_lines, v_lines, color, type_)


def lines_ted(color, type_='FRONT'):
    l = 0.5*l_td
    r = 0.5*r_td
    w = 0.5*width
    if type_ == 'FRONT':
        h_lines = [[z, y-l, y+l]
                   for z in [z_td-r, z_td+r] for y in [w-dy_td, w+dy_td]]
        v_lines = [[y+dy, z_td-r,  z_td+r] for dy in [l, -l]
                   for y in [w-dy_td, w+dy_td]]
    else:
        h_lines = [[y, x-r, x+r] for dy in [-l, l] for y in [w-dy_td+dy, w+dy_td+dy]
                   for x in [x_td1, x_td1 + kol_par, x_td2, x_td2+kol_par]]
        v_lines = [[x+dx, y-l, y+l] for y in [w-dy_td, w+dy_td] for dx in [r, -r]
                   for x in [x_td1, x_td1 + kol_par, x_td2, x_td2+kol_par]]

    do_draw(h_lines, v_lines, color, type_)


def make_triang(x_arr, y_arr):
    nodes_x = [x_ for _ in y_arr for x_ in x_arr]
    nodes_y = [y_ for y_ in y_arr for _ in x_arr]
    elements = [[i + j * dis, i + 1 + j * dis, (j + 1) * dis + i + 1] for j in range(0, dis - 1) for i in
                range(0, dis - 1)]
    elements.extend(
        [[i + j * dis, (j + 1) * dis + i, (j + 1) * dis + i + 1] for j in range(0, dis - 1) for i in range(0, dis - 1)])
    return tri.Triangulation(nodes_x, nodes_y, elements)


def triang_draw(triangulation, scalar_, name_, x_lb='Ось x, метры', y_lb='Ось y, метры', lv=5):
    plt.axis('equal')
    plt.tricontourf(triangulation, scalar_, cmap='YlOrRd')
    plt.colorbar()
    tcf = plt.tricontour(triangulation, scalar_, alpha=0.75, colors='black', linestyles='dotted', levels=lv)
    plt.clabel(tcf, fontsize=10)

    plt.xlabel(x_lb)
    plt.ylabel(y_lb)

    plt.title(name_)


def show(name):
    # mng = plt.get_current_fig_manager()
    # mng.window.state('zoomed')
    file_name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{name}.png"
    plt.savefig(file_name)


def visual_up_per():
    print('График строится..................')

    Xmin = 0
    Xmax = all_length
    Ymax = 0.5 * width
    Ymin = -Ymax

    x_ln = np.linspace(Xmin, Xmax, dis, endpoint=True)
    y_ln = np.linspace(Ymin, Ymax, dis, endpoint=True)

    tr = make_triang(x_ln, y_ln)

    def figure_draw(znach, name_):
        triang_draw(tr, znach, name_)
        lines_shina(sh_tt_gk, 'turquoise', type_='UP')
        lines_shina(sh_gk_vu, 'c', type_='UP')
        lines_oborud(gk, 'lime', 'ГК', type_='UP')
        lines_oborud(tt, 'white', 'ТТ', type_='UP')

    print('Расчёт поля переменного тока.....')
    print('Расчёт поля шин...')
    tt_gk = shina(sh_tt_gk, x_ln, y_ln, z_graph, I_tt_gk, U_tt_gk, type_='UP', ver_='PER')
    gk_vu = shina(sh_gk_vu, x_ln, y_ln, z_graph, I_gk_vu, U_gk_vu, type_='UP', ver_='PER')
    print('Расчёт поля оборудования...')
    gk_f = oborud_per(gk, x_ln, y_ln, z_graph, I_gk, U_gk, type_='UP')
    tt_f = oborud_per(tt, x_ln, y_ln, z_graph, I_tt, U_tt, type_='UP')

    field = field_sum_per(tt_gk, gk_vu, gk_f, tt_f)

    summar = [full_energy(el[0]) for el in field]
    magnetic = [el[0] for el in summar]
    electric = [el[1] for el in summar]
    energy = [el[0]*el[1] for el in summar]

    global gph_num
    gph_num += 1
    plt.figure(gph_num)
    plt.subplot(3, 1, 1)
    figure_draw(magnetic, 'Магнетизм')
    plt.subplot(3, 1, 2)
    figure_draw(electric, 'Электричество')
    plt.subplot(3, 1, 3)
    figure_draw(energy, 'Электричество')
    plt.suptitle('Переменный вид сверху')
    show('пер_верх')

    gph_num += 1
    plt.figure(gph_num)
    name = 'Гармоники магнитное вид сверху'
    j = 0
    for f in harm.keys():
        j += 1
        plt.subplot(4, 2, j)
        data = [dt[0][f][0] for dt in field]
        triang_draw(tr, data, '', y_lb=str(fr), lv=3)
    plt.suptitle(name)
    show('гарм_маг_верх')

    gph_num += 1
    plt.figure(gph_num)
    name = 'Гармоники электричество вид сверху'
    j = 0
    for f in harm.keys():
        j += 1
        plt.subplot(4, 2, j)
        data = [dt[0][f][1] for dt in field]
        triang_draw(tr, data, '', y_lb=str(fr), lv=3)
    plt.suptitle(name)
    show('гарм_эл_верх')


def visual_up_post(z):

    Xmin = 0
    Xmax = all_length
    Ymax = 0.5 * width
    Ymin = -Ymax

    x_ln = np.linspace(Xmin, Xmax, dis, endpoint=True)
    y_ln = np.linspace(Ymin, Ymax, dis, endpoint=True)

    tr = make_triang(x_ln, y_ln)

    def figure_draw(znach, name_):
        triang_draw(tr, znach, name_)
        lines_shina(sh_vu_cp, 'darkcyan', type_='UP')
        lines_shina(sh_cp_td, 'royalblue', type_='UP')
        lines_oborud(vu, 'aqua', 'ВУ', type_='UP')
        lines_oborud(cp, 'indigo', 'СР', type_='UP')
        lines_ted('darkblue', type_='UP')

    print('Расчёт поля постоянного тока.....')
    print('Расчёт поля шин...')
    vu_cp = shina(sh_vu_cp, x_ln, y_ln, z, I_vu_cp, U_vu_cp, type_='UP', ver_='POST')
    cp_td = shina(sh_cp_td, x_ln, y_ln, z, I_cp_td, U_cp_td, type_='UP', ver_='POST')
    print('Расчёт поля оборудования...')
    vu_f = oborud_post(vu, x_ln, y_ln, z, I_vu, U_vu, n_vu, type_='UP')
    cp_f = oborud_post(cp, x_ln, y_ln, z, I_cp, U_cp, n_cp, type_='UP')
    ted_f = oborud_ted(x_ln, y_ln, z, I_td, U_td, n_td, type_='UP')

    summar = field_sum_post(vu_cp, cp_td, vu_f, cp_f, ted_f)
    magnetic = [el[0][0] for el in summar]
    electric = [el[0][1] for el in summar]
    energy = [el[0][0]*el[0][1] for el in summar]

    global gph_num
    gph_num += 1
    plt.figure(gph_num)
    plt.subplot(3, 1, 1)
    figure_draw(magnetic, 'Магнетизм')
    plt.subplot(3, 1, 2)
    figure_draw(electric, 'Электричество')
    plt.subplot(3, 1, 3)
    figure_draw(energy, 'Энергия')
    if z == floor+z_td:
        plt.suptitle('Постоянный, вид сверху ТЭД')
    else:
        plt.suptitle('Постоянный, вид сверху')

    show(f'пост_верх_{z}_м')


def visual_front():
    print('График строится..................')

    Ymax = -0.5 * width
    Ymin = -Ymax
    Zmax = 0.1
    Zmin = height+floor

    y_ln = np.linspace(Ymin, Ymax, dis, endpoint=True)
    z_ln = np.linspace(Zmin, Zmax, dis, endpoint=True)

    tr = make_triang(y_ln, z_ln)

    global gph_num

    for no in SZ.keys():
        print(f"Построение среза {no} м")
        print('Расчёт поля переменного тока...')

        print('Расчёт поля шин...')
        tt_gk = shina(sh_tt_gk, y_ln, z_ln, no, I_tt_gk, U_tt_gk, ver_='PER')
        gk_vu = shina(sh_gk_vu, y_ln, z_ln, no, I_gk_vu, U_gk_vu, ver_='PER')
        print('Расчёт поля оборудования...')
        gk_f = oborud_per(gk, y_ln, z_ln, no, I_gk, U_gk)
        tt_f = oborud_per(tt, y_ln, z_ln, no, I_tt, U_tt)
        field_per = field_sum_per(tt_gk, gk_vu, gk_f, tt_f)

        if no == x_chel+0.01:
            energy_per = [ekran(elm, tp='SPLOSH') for elm in field_per]
        else:
            energy_per = [ekran(elm) for elm in field_per]

        print('Расчёт поля постоянного тока...')
        print('Расчёт поля шин...')
        vu_cp = shina(sh_vu_cp, y_ln, z_ln, no, I_vu_cp, U_vu_cp, ver_='POST')
        cp_td = shina(sh_cp_td, y_ln, z_ln, no, I_cp_td, U_cp_td, ver_='POST')
        print('Расчёт поля оборудования...')
        vu_f = oborud_post(vu, y_ln, z_ln, no, I_vu, U_vu, n_vu)
        cp_f = oborud_post(cp, y_ln, z_ln, no, I_cp, U_cp, n_cp)
        ted_f = oborud_ted(y_ln, z_ln, no, I_td, U_td, n_td)
        field_post = field_sum_post(vu_cp, cp_td, vu_f, cp_f, ted_f)

        if no == x_chel+0.01:
            energy_post = [ekran(elm, tp='SPLOSH', ver='POST') for elm in field_post]
        else:
            energy_post = [ekran(elm, ver='POST') for elm in field_post]

        gph_num += 1
        plt.figure(gph_num)

        sb = f' Экран сетка' if no == x_chel else f' Экран слошной' if no == x_chel+0.01 else ''
        name = f'Энергия. Вид сбоку. Срез {SZ[no]}.{sb}'

        plt.subplot(1, 2, 1)
        triang_draw(tr, energy_per, 'Переменный', y_lb='Ось z, метры')
        lines_shina(sh_tt_gk, 'turquoise')
        lines_shina(sh_gk_vu, 'c')
        lines_oborud(gk, 'lime', 'ГК')
        lines_oborud(tt, 'white', 'ТТ')

        plt.subplot(1, 2, 2)
        triang_draw(tr, energy_post, 'Постоянный', y_lb='Ось z, метры')
        lines_shina(sh_vu_cp, 'darkcyan')
        lines_shina(sh_cp_td, 'royalblue')
        lines_oborud(vu, 'aqua', 'ВУ')
        lines_oborud(cp, 'indigo', 'СР')
        lines_ted('darkblue')

        plt.suptitle(name)
        show(f'энерг_{no}_м')

        gph_num += 1
        plt.figure(gph_num)

        name = f'Гармоники вид сбоку {SZ[no]} м.{sb}'
        j = 0
        for f in harm.keys():
            j += 1
            plt.subplot(3, 3, j)
            data = [dt[0][f][0] * dt[0][f][1] for dt in field_per]
            triang_draw(tr, data, '', y_lb=str(f))
        plt.suptitle(name)
        show(f'гарм_{SZ[no]}_м')

# ПОСТРОЕНИЕ ГРАФИКА


SZ = {x_chel: 'кабина',
      x_chel+0.01: 'кабина', # эта строка добавлена для того, чтобы была возможность расчитать два варианта экрана
      x_vu1-0.5*l_vu: 'ВУ/СР ближние',
      x_tt+0.5*x_tt: 'ТТ/ГК',
      x_vu2+0.5*l_vu: 'ВУ/СР дальние'
      }

z_graph = 3

gph_num = 0
print('\nВид сверху.')
# visual_up_per()
# visual_up_post(z_graph)  # построение среза на высоте пола
# visual_up_post(floor+z_td)  # построение среза на высоте ТЭД
print('\nВид сбоку')
visual_front()

# РАСЧЁТ СТАТИСТИКИ

S = (a * b / 3600) ** 1 / 2
p = ti / 24  # статистическая вероятность воздействия

print('\nпеременное \t| постоянное\t | общее')

x_ln, y_ln = [x_chel], [y_chel]
z_chel = floor + 1
ch_per = field_sum_per(
    shina(sh_tt_gk, x_ln, y_ln, z_chel, I_tt_gk, U_tt_gk, type_='UP', ver_='PER'),
    shina(sh_gk_vu, x_ln, y_ln, z_chel, I_gk_vu, U_gk_vu, type_='UP', ver_='PER'),
    oborud_per(gk, x_ln, y_ln, z_chel, I_gk, U_gk, type_='UP'),
    oborud_per(tt, x_ln, y_ln, z_chel, I_tt, U_tt, type_='UP'))[0][0]
ch_post = field_sum_post(
    shina(sh_vu_cp, x_ln, y_ln, z_chel, I_vu_cp, U_vu_cp, type_='UP', ver_='POST'),
    shina(sh_cp_td, x_ln, y_ln, z_chel, I_cp_td, U_cp_td, type_='UP', ver_='POST'),
    oborud_post(vu, x_ln, y_ln, z_chel, I_vu, U_vu, n_vu, type_='UP'),
    oborud_post(cp, x_ln, y_ln, z_chel, I_cp, U_cp, n_cp, type_='UP'),
    oborud_ted(x_ln, y_ln, z_chel, I_td, U_td, n_td, type_='UP'))[0][0]

e_per = 0
for el in ch_per.values():
    e_per += el[0]*el[1]
e_post = ch_post[0] * ch_post[1]

print(f'Поле без экрана:\n{e_per:.3f}\t| {e_post:.3f}\t| {(e_per+e_post):.3f}')

e_per = 0
for fq in harm.keys():
    e_per += ch_per[fq][0] / (koef_ekr_h_setka[fq] * koef_ekr_h_splosh_z) *\
               ch_per[fq][1] / (koef_ekr_e_setka[fq] * koef_ekr_e_splosh)
e_post = ch_post[0] / (k_post_ekr_h_setka * koef_ekr_h_splosh_z) *\
           ch_post[1] / (k_post_ekr_e_setka * koef_ekr_e_splosh)
print(f'\nЭкран сетка:\n{e_per:.3f}\t| {e_post:.3f}\t| {(e_per+e_post):.3f}')
Dco = (e_per+e_post) * ti * S * p
Dpo = Dco / b
print('Удельная суточная доза поглощённой энергии: %.4f' % Dpo)

f_per = full_energy(ch_per)
e_per = f_per[0] / (koef_ekr_h_splosh_v * koef_ekr_h_splosh_z) * \
          f_per[1] / (koef_ekr_e_splosh * koef_ekr_e_splosh)
e_post = ch_post[0] / (koef_ekr_h_splosh_v * koef_ekr_h_splosh_z) *\
           ch_post[1] / (koef_ekr_e_splosh * koef_ekr_e_splosh)

print(f'\nЭкран сплошной:\n{e_per:.3f}\t| {e_post:.3f}\t| {(e_per+e_post):.3f}')
Dco = (e_per+e_post) * ti * S * p
Dpo = Dco / b
print('Удельная суточная доза поглощённой энергии: %.4f' % Dpo)

plt.show()
