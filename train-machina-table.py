from math import pi, log, exp, atan
import numpy as np


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
x_chel = 0.9  # положение человека по оси х
y_chel = 1.2 - 0.3  # положение человека по оси y
z_chel = 2 + 1.5  # положение человека по оси z
z_chair = 2 + 1.2  # сидушка стула
a = 1.75  # высота человека метры
b = 80  # масса человека килограммы
ti = 1  # длительность пребывания работника на рабочем месте, часов
chel = {}

f_per = open('machina_peremennoe_pole.txt', 'w')
f_post = open('machina_postoyannoe_pole.txt', 'w')

# КОНСТАНТЫ
#todo dis = 60  # дискретизация графиков
dis = 20  # дискретизация графиков
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
kor_w = 0.5  # ширина коридора

# УЗЛЫ ОБОРУДОВАНИЯ ИХ ТОКИ И НАПРЯЖЕНИЯ
# x - от стенки кабины
# y - от нижнежнего края
# z - от пола

x_tt = 4.500  # тяговый трансформатор
y_tt = 0.600 + 0.800
l_tt = 1.420
l_sh_tt = 1.1
w_tt = 0.7
h_tt = 1.265
d_tt = 0.3

x_pr = 4.8  # переходной реактор
l_pr = 0.835
y_pr = 1.4
w_pr = 0.940
z_pr = 1.25
h_pr = 0.955

x_gk = x_tt + 0.132  # главный контроллер
y_gk = 0.600 + 0.8
z_gk = 0.5
l_gk = 1.155
w_gk = 0.880
h_gk = 0.735

x_mt = x_tt + 1.420  # медная труба
y_mt = y_tt

x_vu1 = 3.700  # выпрямительная установка
x_vu2 = 6.700
y_vu1 = 0.600 + 0.15
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


I_pr = 1270
U_pr = 1500

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
d_v = 1  # толщина сетки мм

b_z = 3.5  # ширина экрана кузова, м
l_z = 15.2  # длина экрана кузова, м
h_z = 2.8  # высота экрана кузова, м
d = 0.0025  # толщина экрана кузова, м

Z0 = 377  # волновое сопротивление поля, Ом
ro = 0.15  # удельное сопротивление материала сетки, Ом*м
rs = 15 / 1000  # шаг сетки, м
sc = 0.5 / 1000  # диаметр провода сетки, м

sigma = 10 ** 7  # удельная проводимость стали, см/м
v_kab = 1.3 * 2.8 * 2.6  # объём кабины, м
r_kab = (v_kab * 3 / (4 * pi)) ** (1 / 3)  # эквивалентный радиус кабины, м
v_mash = 13.9 * 2.8 * 2.6  # объём машинного отделения, м
r_mash = (v_mash * 3 / (4 * pi)) ** (1 / 3)  # эквивалентный радиус машинного отделения, м
v_vvk = 12.7 * 2.2 * 2.6  # объём камеры, м
r_vvk = (v_vvk * 3 / (4 * pi)) ** (1 / 3)  # эквивалентный радиус камеры, м
e0 = 8.85 * (10 ** -12)  # диэлектрическая постоянная

ke_post_splosh = (60 * pi * d * sigma)

kh_splosh_kab = (1 + 1000 * d / (2 * r_kab)) ** 2
kh_splosh_mash = (1 + 1000 * d / (2 * r_mash)) ** 2
kh_splosh_vvk = (1 + 400 * d / (2 * r_vvk)) ** 2


y = rs / (2*pi*r_vvk) * (log(rs/sc) - 1.25)
ke_per_setka = 1 / (3*y / (1 + 3*y))


kh_per_setka, ke_per_splosh_vvk, ke_per_splosh_kab = {}, {}, {}
Ch = exp(2 * pi * sc / (rs - 2 * sc))
Ce = exp(2 * pi * 0.0025 / 0.01)

for fr in harm.keys():
    f = fr / 1000000
    lam = 300000000 / f  # длина волны

    Zh = Z0 * 2 * r_vvk * pi / lam
    Ze_vvk = Z0 * lam / (2 * pi * r_vvk)
    Ze_kab = Z0 * lam / (2 * pi * r_kab)

    Ah = (d * sigma * Zh) ** 0.5
    delta = 0.03 * ((10 ** -7) * lam) ** 0.5
    Ae_vvk = (delta * Ze_vvk / (10 ** -7)) ** 0.5
    Ae_kab = (delta * Ze_kab / (10 ** -7)) ** 0.5
    B_vvk = (lam / r_vvk) ** (1 / 3)
    B_kab = (lam / r_kab) ** (1 / 3)

    kh_per_setka[fr] = 0.024 * Ah * B_vvk * Ch
    ke_per_splosh_vvk[fr] = 0.024 * Ae_vvk * B_vvk * Ce * 0.001
    ke_per_splosh_kab[fr] = 0.024 * Ae_kab * B_kab * Ce * 0.001


# ШИНЫ И ОБОРУДОВАНИЕ ДЛЯ РАСЧЁТОВ

# формат шин: [(координаты начала), (смещение конца относительно начала)]
# формат оборудования: [[x_нач, x_кон, y_нач, y_кон, z_нач, z_кон]]

mednaya_truba = [[(x_mt, y_mt, 0), (0, 0, height)]]

sh_tt_gk = [[(x_tt + l_tt / 6 * i, y_tt, 0.5), (0, 0, -1.0)] for i in range(0, 7)]

sh_gk_vu = [[(x_gk, y_gk + h_gk, z_gk), (0, 0.5, 0)],
            [(x_gk, y_gk + h_gk + 0.5, z_gk), (0, 0, 0.8)],
            [(x_gk, y_gk + h_gk + 0.5, z_gk + 0.8), (-0.8, 0, 0)],
            [(x_gk - 0.8, y_gk + h_gk + 0.5, z_gk + 0.8), (0, -0.8, 0)],
            [(x_gk - 0.8, y_gk + h_gk + 0.5 - 0.8, z_gk + 0.8), (0, 0, -1.6)],
            [(x_gk - 0.8, y_gk + h_gk + 0.5 - 0.3, z_gk + 0.8 - 1.1), (-0.15, 0, 0)],
            [(x_gk - 0.8, y_gk + h_gk + 0.5 - 0.3, z_gk + 0.8 - 1.6), (0, -1.4, 0)],
            [(x_gk - 0.8, y_gk + h_gk + 0.5 - 0.3 - 1.4, z_gk + 0.8 - 1.6), (-0.15, 0, 0)],

            [(x_gk + l_gk, y_gk + h_gk, z_gk), (0, 0.5, 0)],
            [(x_gk + l_gk, y_gk + h_gk + 0.5, z_gk), (0, 0, 0.8)],
            [(x_gk + l_gk, y_gk + h_gk + 0.5, z_gk + 0.8), (0.8, 0, 0)],
            [(x_gk + l_gk + 0.8, y_gk + h_gk + 0.5, z_gk + 0.8), (0, -0.8, 0)],
            [(x_gk + l_gk + 0.8, y_gk + h_gk + 0.5 - 0.8, z_gk + 0.8), (0, 0, -1.6)],
            [(x_gk + l_gk + 0.8, y_gk + h_gk + 0.5 - 0.3, z_gk + 0.8 - 1.1), (0.15, 0, 0)],
            [(x_gk + l_gk + 0.8, y_gk + h_gk + 0.5 - 0.3, z_gk + 0.8 - 1.6), (0, -1.4, 0)],
            [(x_gk + l_gk + 0.8, y_gk + h_gk + 0.5 - 0.3 - 1.4, z_gk + 0.8 - 1.6), (0.15, 0, 0)],
            ]

tt = [[x_tt, x_tt + l_tt, y_tt, y_tt + w_tt, 0.5, 0.5 - h_tt]]

pr = [[x_pr, x_pr+l_pr, y_pr, y_pr+w_pr, z_pr, z_pr+h_pr]]

gk = [[x_gk, x_gk + l_gk, y_gk, y_gk + w_gk, z_gk, z_gk + h_gk]]

# ПОСТОЯННОЕ

sh_vu_cp = [
    [(x_vu1, y_vu1 + 0.2, z_vu), (1, 0, 0)],
    [(x_vu1, y_vu2 + 0.2, z_vu), (1, 0, 0)],
    [(x_vu1 + 1, y_vu1 + 0.2, z_vu), (0, 1.4, 0)],
    [(x_vu1 + 1, y_vu1 + 0.83, z_vu), (-1, 0, 0)],

    [(x_vu2, y_vu1 + 0.2, z_vu), (-1, 0, 0)],
    [(x_vu2, y_vu2 + 0.2, z_vu), (-1, 0, 0)],
    [(x_vu2 - 1, y_vu1 + 0.2, z_vu), (0, 1.4, 0)],
    [(x_vu2 - 1, y_vu1 + 0.83, z_vu), (1, 0, 0)]]

sh_cp_td = [
    [(x_cp2 + l_cp, y_cp, 0), (0, 0, 1.9)],
    [(x_cp2 + l_cp, y_cp, 1.9), (0, 0, -0.8)],
    [(x_cp2 + l_cp, y_cp, 1.1), (2.5, 0, 0)],
    [(x_cp2 + l_cp + 2.5, y_cp, 1.1), (0, 0, -1.7 - 0.5)],
    [(x_cp1 - l_cp, y_cp, 0), (0, 0, 1.9)],
    [(x_cp1 - l_cp, y_cp, 1.9), (0, 0, -0.8)],
    [(x_cp1 - l_cp, y_cp, 1.1), (-2.5, 0, 0)],
    [(x_cp1 - l_cp - 2.5, y_cp, 1.1), (0, 0, -1.7 - 0.5)],
]

vu = [[x_vu1, x_vu1 - l_vu, y_vu1, y_vu1 + h_vu, z_vu, z_vu + w_vu],
      [x_vu1, x_vu1 - l_vu, y_vu2, y_vu2 + h_vu, z_vu, z_vu + w_vu],
      [x_vu2, x_vu2 + l_vu, y_vu1, y_vu1 + h_vu, z_vu, z_vu + w_vu],
      [x_vu2, x_vu2 + l_vu, y_vu2, y_vu2 + h_vu, z_vu, z_vu + w_vu]]

cp = [[x_cp1, x_cp1 - l_cp, y_cp - 0.5 * h_cp, y_cp + 0.5 * h_cp, z_cp, z_cp + w_cp],
      [x_cp2, x_cp2 + l_cp, y_cp - 0.5 * h_cp, y_cp + 0.5 * h_cp, z_cp, z_cp + w_cp]]


def radius(st, ed):
    return ((st[0] - ed[0]) ** 2 + (st[1] - ed[1]) ** 2 + (st[2] - ed[2]) ** 2) ** 0.5


def shina(shinas, v1arr, v2arr, v3, I, U, type_='FRONT', ver_='PER'):
    dc = 10

    sh_p = []
    for sh in shinas:
        if sh[1][0]:
            arr = np.linspace(sh[0][0], sh[0][0] + sh[1][0], dc)
            sh_p.extend([(x, sh[0][1], sh[0][2]) for x in arr])
        elif sh[1][1]:
            arr = np.linspace(sh[0][1], sh[0][1] + sh[1][1], dc)
            sh_p.extend([(sh[0][0], y, sh[0][2]) for y in arr])
        elif sh[1][2]:
            arr = np.linspace(sh[0][2], sh[0][2] + sh[1][2], 3*dc)
            sh_p.extend([(sh[0][0], sh[0][1], z) for z in arr])

    sh_points = [(length + pp[0], -0.5 * width + pp[1], floor + pp[2]) for pp in sh_p]

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
        points.extend(
            [[length + x_, -0.5 * width + y_, floor + z_] for z_ in nodes_z for y_ in nodes_y for x_ in nodes_x])

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
        return [{f: [harm[f][0] * H_ob / len(points), harm[f][1] * E_ob, ] for f in harm.keys()}, (x_, y_, z_)]

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
        points.extend(
            [[length + x_, -0.5 * width + y_, floor + z_] for z_ in nodes_z for y_ in nodes_y for x_ in nodes_x])

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
    nodes_x = [length + dx + 0.5 * r_td * np.cos(ap) for dx in [x_td1, x_td1 + kol_par, x_td2, x_td2 + kol_par]
               for ap in np.linspace(0, 2 * pi, ds)]
    nodes_z = [floor + z_td + 0.5 * r_td * np.sin(ap) for ap in np.linspace(0, 2 * pi, ds)]
    nodes_y = [td_p for td_p in np.linspace(-0.5 * l_td, 0.5 * l_td, 4)]
    points = [[x_, y_, z_] for z_ in nodes_z for y_ in nodes_y for x_ in nodes_x]

    x_cab = np.linspace(length, all_length, 40)
    y_cab = np.linspace(-0.5 * width, 0.5 * width, 40)
    minus = [[x_, y_] for y_ in y_cab for x_ in x_cab]

    def in_point(x_, y_, z_):
        H_ob, E_ob = 0, 0
        for p in points:
            r = ((p[0] - x_) ** 2 + (p[1] - y_) ** 2 + (p[2] - z_) ** 2) ** 0.5
            H_ob += I / (pi * l_td) * atan(l_td / 2 / r)
            E_ob += U / r / len(points)

        for m in minus:
            r_m = ((m[0] - x_) ** 2 + (m[1] - y_) ** 2 + (floor - z_) ** 2) ** 0.5
            if r_m != 0:
                E_ob += U / r_m / len(minus)

        h, e = H_ob * n / len(points), E_ob
        if z_ > floor - 1:
            h /= kh_splosh_kab
            e /= ke_post_splosh
            if z_ > floor:
                h /= kh_splosh_kab
                e /= ke_post_splosh
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
    for e in en[0].values():
        sum_h += e[0]
        sum_e += e[1]
    return [sum_h, sum_e], en[1]


def ekran(elm, tp='SETKA', ver='PER', viz='FRONT'):
    y_flag = elm[1][1] < 0.5 * width - kor_w if viz == 'FRONT' else elm[1][1] > -0.5 * width + kor_w

    if (elm[1][0] > length + 0.4) and y_flag and (elm[1][2] > 2):  # внутри ВВК
        return elm
    else:  # коридор кузова
        if tp == 'SETKA':
            if ver == 'PER':
                k_h = kh_per_setka
                k_e = ke_per_setka
            else:
                k_h = kh_splosh_vvk
                k_e = ke_post_splosh
        else:
            if ver == 'PER':
                k_e = ke_per_splosh_vvk
            else:
                k_e = ke_post_splosh
            k_h = kh_splosh_vvk

    if elm[1][0] < length:  # кабина
        if tp == 'SETKA':
            if ver == 'PER':
                k_h = {frq: kh_per_setka[frq] * kh_splosh_kab for frq in harm.keys()}
                k_e = {frq: ke_per_setka * ke_per_splosh_kab[frq] for frq in harm.keys()}
            else:
                k_h *= kh_splosh_kab
                k_e *= ke_post_splosh
        else:
            if ver == 'PER':
                k_e = {frq: ke_per_splosh_vvk[frq] * ke_per_splosh_kab[frq] for frq in harm.keys()}
            else:
                k_e *= ke_post_splosh
            k_h *= kh_splosh_kab

    if ver == 'PER':
        if tp == 'SETKA':
            if type(k_e) == dict:
                return [{frq: [elm[0][frq][0] / k_h[frq], elm[0][frq][1] / k_e[frq]] for frq in harm.keys()}, elm[1]]
            else:
                return [{frq: [elm[0][frq][0] / k_h[frq], elm[0][frq][1] / k_e] for frq in harm.keys()}, elm[1]]
        else:
            return [{frq: [elm[0][frq][0] / k_h, elm[0][frq][1] / k_e[frq]] for frq in harm.keys()}, elm[1]] # тут может быть и словарь и нет
    else:
        return [(elm[0][0] / k_h, elm[0][1] / k_e), elm[1]]


def table_out(scalar, h_ln, v_ln, rf, ln=12):
    znach = np.array(scalar).reshape(len(v_ln), len(h_ln))

    for pt in h_ln:
        print(f'{pt:.3f}'.ljust(ln), end='', file=rf)
    for no, y_list in enumerate(znach):
        for dt in y_list:
            print(f'{dt:.3f}'.ljust(ln), end='', file=rf)
        print(f'| {v_ln[no]:.3f}', file=rf)
    print('\n', file=rf)


def visual_up_per():
    print('Таблица рассчитывается..................')
    print('ВИД СВЕРХУ', file=f_per)

    Xmin = 0
    Xmax = all_length
    Ymax = -0.5 * width
    Ymin = -Ymax

    x_ln = np.linspace(Xmin, Xmax, dis, endpoint=True)
    y_ln = np.linspace(Ymin, Ymax, dis, endpoint=True)

    print('Расчёт поля переменного тока.....')
    print('Расчёт поля шин...')
    tt_gk = shina(sh_tt_gk, x_ln, y_ln, z_graph, I_tt_gk, U_tt_gk, type_='UP', ver_='PER')
    gk_vu = shina(sh_gk_vu, x_ln, y_ln, z_graph, I_gk_vu, U_gk_vu, type_='UP', ver_='PER')
    mt = shina(mednaya_truba, x_ln, y_ln, z_graph, I_mt, U_mt, type_='UP', ver_='PER')
    print('Расчёт поля оборудования...')
    gk_f = oborud_per(gk, x_ln, y_ln, z_graph, I_gk, U_gk, type_='UP')
    pr_f = oborud_per(pr, y_ln, y_ln, z_graph, I_pr, U_pr, type_='UP')

    field = [ekran(el, viz='UP') for el in field_sum_per(tt_gk, gk_vu, gk_f, mt, pr_f)]

    summar = [full_energy(el) for el in field]
    magnetic = [el[0][0] for el in summar]
    electric = [el[0][1] for el in summar]
    energy = [el[0][0] * el[0][1] for el in summar]

    print('\nМагнетизм', file=f_per)
    table_out(magnetic, x_ln, y_ln, f_per)
    print('\nЭлектричество', file=f_per)
    table_out(electric, x_ln, y_ln, f_per)
    print('\nЭнергия', file=f_per)
    table_out(energy, x_ln, y_ln, f_per)

    print('\nГармоники магнитное вид сверху', file=f_per)
    for f in harm.keys():
        data = [dt[0][f][0] for dt in field]
        print(f"{fr} Гц", file=f_per)
        table_out(data, x_ln, y_ln, f_per)

    print('Гармоники электричество вид сверху', file=f_per)
    for f in harm.keys():
        data = [dt[0][f][1] for dt in field]
        print(f"{fr} Гц", file=f_per)
        table_out(data, x_ln, y_ln, f_per)


def visual_up_post(z):
    if z == floor + z_td:
        print('\n\nВИД СВЕРХУ ТЭД', file=f_post)
    else:
        print('\n\nВИД СВЕРХУ', file=f_post)

    Xmin = 0
    Xmax = all_length
    Ymax = -0.5 * width
    Ymin = -Ymax

    x_ln = np.linspace(Xmin, Xmax, dis, endpoint=True)
    y_ln = np.linspace(Ymin, Ymax, dis, endpoint=True)

    print('Расчёт поля постоянного тока.....')
    print('Расчёт поля шин...')
    vu_cp = shina(sh_vu_cp, x_ln, y_ln, z, I_vu_cp, U_vu_cp, type_='UP', ver_='POST')
    cp_td = shina(sh_cp_td, x_ln, y_ln, z, I_cp_td, U_cp_td, type_='UP', ver_='POST')
    print('Расчёт поля оборудования...')
    vu_f = oborud_post(vu, x_ln, y_ln, z, I_vu, U_vu, n_vu, type_='UP')
    cp_f = oborud_post(cp, x_ln, y_ln, z, I_cp, U_cp, n_cp, type_='UP')
    ted_f = oborud_ted(x_ln, y_ln, z, I_td, U_td, n_td, type_='UP')

    if z < floor:
        summ_inside = field_sum_post(vu_cp, cp_td, vu_f, cp_f)
        ekran_f = [[[el[0][0] / kh_splosh_vvk, el[0][1] / ke_post_splosh], el[1]] for el in summ_inside]
        summar = field_sum_post(ekran_f, ted_f)
    else:
        summar = field_sum_post(vu_cp, cp_td, vu_f, cp_f, ted_f)
    magnetic = [el[0][0] for el in summar]
    electric = [el[0][1] for el in summar]
    energy = [el[0][0] * el[0][1] for el in summar]

    print('\nМагнетизм', file=f_post)
    table_out(magnetic, x_ln, y_ln, f_post)
    print('\nЭлектричество', file=f_post)
    table_out(electric, x_ln, y_ln, f_post)
    print('\nЭнергия', file=f_post)
    table_out(energy, x_ln, y_ln, f_post)


def energy_pass(x, y, z, okna=False):
    # на данном этапе считаем что внешнее поле не проходит в кабину и кузов
    def fr_make():
        return {f: [0, 0] for f in harm.keys()}
    return [fr_make(), (x, y, z)]


def visual_front():
    print('График строится..................')

    Ymax = 0.5 * width
    Ymin = -Ymax
    Zmax = floor + 0.1
    Zmin = height + floor

    y_ln = np.linspace(Ymin, Ymax, dis)
    z_ln = np.linspace(Zmin, Zmax, dis)

    def eng(en):
        fe = full_energy(en)
        return fe[0][0] * fe[0][1]

    for no in SZ.keys():

        print(f"Построение среза {no} м")
        print('Расчёт поля переменного тока...')

        print('Расчёт поля шин...')
        tt_gk = shina(sh_tt_gk, y_ln, z_ln, no, I_tt_gk, U_tt_gk, ver_='PER')
        gk_vu = shina(sh_gk_vu, y_ln, z_ln, no, I_gk_vu, U_gk_vu, ver_='PER')
        mt = shina(mednaya_truba, y_ln, z_ln, no, I_mt, U_mt, ver_='PER')
        print('Расчёт поля оборудования...')
        gk_f = oborud_per(gk, y_ln, z_ln, no, I_gk, U_gk)
        pr_f = oborud_per(pr, y_ln, z_ln, no, I_pr, U_pr)

        field_per = field_sum_per(tt_gk, gk_vu, gk_f, mt, pr_f)

        print('Расчёт поля постоянного тока...')
        print('Расчёт поля шин...')
        vu_cp = shina(sh_vu_cp, y_ln, z_ln, no, I_vu_cp, U_vu_cp, ver_='POST')
        cp_td = shina(sh_cp_td, y_ln, z_ln, no, I_cp_td, U_cp_td, ver_='POST')
        print('Расчёт поля оборудования...')
        vu_f = oborud_post(vu, y_ln, z_ln, no, I_vu, U_vu, n_vu)
        cp_f = oborud_post(cp, y_ln, z_ln, no, I_cp, U_cp, n_cp)
        field_post = field_sum_post(vu_cp, cp_td, vu_f, cp_f)

        ted_f = oborud_ted(y_ln, z_ln, no, I_td, U_td, n_td)

        print('Расчёт экрана...')

        ekran_per_setka = [ekran(elm) for elm in field_per]
        ekran_per_splosh = [ekran(elm, tp='SPLOSH') for elm in field_per]

        ekran_post_setka = field_sum_post([ekran(elm, ver='POST') for elm in field_post], ted_f)
        ekran_post_splosh = field_sum_post([ekran(elm, tp='SPLOSH', ver='POST') for elm in field_post], ted_f)

        koridor = {'сетка_переменный': np.array([eng(en) for en in ekran_per_setka]),
                   'сетка_постоянный': np.array([en[0][0] * en[0][1] for en in ekran_post_setka]),
                   'сплошной_переменный': np.array([eng(en) for en in ekran_per_splosh]),
                   'сплошной_постоянный': np.array([en[0][0] * en[0][1] for en in ekran_post_splosh])
                   }

        for nam, val in koridor.items():

            table_out(val, y_ln, z_ln, f_per)
            table_out(val, y_ln, z_ln, f_post)

            if 'переменный' in nam:
                tp = 'сетка' if 'сетка' in nam else 'сплошной'
                print(f"Энергия срез {SZ[no]} {tp}", file=f_per)
                table_out(val, y_ln, z_ln, f_per)
            else:
                tp = 'сетка' if 'сетка' in nam else 'сплошной'
                print(f"Энергия срез {SZ[no]} {tp}", file=f_post)
                table_out(val, y_ln, z_ln, f_post)

        print(f'Гармоники вид сбоку {SZ[no]}.', file=f_per)
        for f in harm.keys():
            data = [dt[0][f][0] * dt[0][f][1] for dt in ekran_per_setka]
            print(f"{fr} Гц", file=f_per)
            table_out(data, y_ln, z_ln, f_per)


# ПОСТРОЕНИЕ ГРАФИКА

SZ = {1: 'кабина',
      x_vu1 - 0.5 * l_vu: 'ВУ_СР ближние',
      x_tt + 0.5 * x_tt: 'ТТ_ГК',
      x_vu2 + 0.5 * l_vu: 'ВУ_СР дальние'
      }

z_graph = 3

print('\nВид сверху.')

visual_up_per()
visual_up_post(z_graph)  # построение среза на высоте пола
visual_up_post(floor + z_td)  # построение среза на высоте ТЭД

print('\nВид сбоку')
print('\n\nВИД СБОКУ', file=f_post)
visual_front()

f_post.close()
f_per.close()
