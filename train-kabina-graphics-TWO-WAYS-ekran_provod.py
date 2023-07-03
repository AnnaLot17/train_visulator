# Импорт библиотек
from math import log, exp, pi, atan
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib.colors as colors
from shapely.geometry import Polygon, LineString, Point

# цветоваяя схема графиков
plt.style.use('seaborn-white')
cmap = 'YlOrRd'

# РЕЖИМ РАБОТЫ СЕТИ

I = 300  # cуммарная сила тока, А
U = 25000  # cуммарное напряжение, В
U_ep = U

I_ted = 880  # сила тока в ТЭД, А
U_ted = 1950  # напряжение в ТЭД, В

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
xp_kp = 0  # m - расстояние от центра между рельсами до КП (если левее центра - поставить минус)
xp_nt = 0  # m - расстояние от центра между рельсами до НТ (если левее центра - поставить минус)
xp_up = -3.7  # m - расстояние от центра между рельсами до УП
xp_ep = -2.7   # m - расстояние от центра между рельсами до ЭП
d_kp = 12.81 / 1000  # mm диаметр провода КП
d_nt = 12.5 / 1000  # mm диаметр провода НТ
d_up = 17.5 / 1000  # mm диаметр провода УП
d_ep = 12.5 / 1000  # mm диаметр провода КП
h_kp = 6.0  # КП высота, м
h_nt = 7.8  # НТ высота, м
h_up = 8.0  # УП высота, м
h_ep = 8.4  # ЕП высота, м

xp_mid = 4.2  # расстояние между центрами путей
xp_kp2 = 0  # m - расстояние от центра между вторыми рельсами до КП2 (если левее центра - поставить минус)
xp_nt2 = 0  # m - расстояние от центра между вторыми рельсами до НТ2 (если левее центра - поставить минус)
xp_up2 = 3.7  # m - расстояние от центра между вторыми рельсами до УП2
xp_ep2 = 2.7  # m - расстояние от центра между вторыми рельсами до ЭП2

# ДАННЫЕ О ЛОКОМОТИВЕ

length = 1.3  # длина кабины
all_length = 15.2  # длина всего локомотива
width = 2.8  # ширина кабины
height = 2.6  # высота кабины
# узлы лобового окна min_x, max_x, min_y, max_y, min_z, max_z
bor = [0.2, 0.6, -1.2, 1.2, floor+1.5, floor+2.2]
# узлы для бокового окна: min_x, max_x, min_z, max_z
sbor = [0.3, 1, floor+1.5, floor+2.2]

# формируем передние окна методом Polygon: составляем список из координат точек по x, y, z каждого угла
frontWindleft = Polygon([(bor[0], bor[2], bor[4]),
                         (bor[1], bor[2], bor[5]),
                         (bor[1], -0.22, bor[5]),
                         (bor[0], -0.22, bor[4])])  # левое
frontWindright = Polygon([(bor[0], 0.22, bor[4]),
                          (bor[1], 0.22, bor[5]),
                          (bor[1], bor[3], bor[5]),
                          (bor[0], bor[3], bor[4])])  # правиое

# расчёт границ теней боковых окон для кажого источника поля
min_nt = Point(0.5*width, sbor[3]).distance(Point(xp_nt, h_nt))  # луч нижней границы тени от НТ
max_nt = Point(0.5*width, sbor[2]).distance(Point(xp_nt, h_nt))  # луч нижней границы тени от НТ

min_kp = Point(0.5*width, sbor[3]).distance(Point(xp_kp, h_kp))  # далее аналогично
max_kp = Point(0.5*width, sbor[2]).distance(Point(xp_kp, h_kp))

min_up = Point(-0.5*width, sbor[3]).distance(Point(xp_up, h_up))
max_up = Point(-0.5*width, sbor[2]).distance(Point(xp_up, h_up))

min_ep = Point(-0.5*width, sbor[3]).distance(Point(xp_ep, h_ep))
max_ep = Point(-0.5*width, sbor[2]).distance(Point(xp_ep, h_ep))

min_nt2 = Point(0.5*width, sbor[3]).distance(Point(xp_nt2+xp_mid, h_nt))
max_nt2 = Point(0.5*width, sbor[2]).distance(Point(xp_nt2+xp_mid, h_nt))

min_kp2 = Point(0.5*width, sbor[3]).distance(Point(xp_kp2+xp_mid, h_kp))
max_kp2 = Point(0.5*width, sbor[2]).distance(Point(xp_kp2+xp_mid, h_kp))

min_up2 = Point(0.5*width, sbor[3]).distance(Point(xp_up2+xp_mid, h_up))
max_up2 = Point(0.5*width, sbor[2]).distance(Point(xp_up2+xp_mid, h_up))

min_ep2 = Point(0.5*width, sbor[3]).distance(Point(xp_ep2+xp_mid, h_ep))
max_ep2 = Point(0.5*width, sbor[2]).distance(Point(xp_ep2+xp_mid, h_ep))

# ЭКРАН

Z0 = 377  # волновое сопротивление поля, Ом
mu = 1000  # относительная магнитная проницаемость стали
dst = 0.0025  # толщина стали м
sigma = 10 ** 7  # удельная проводимость стали

v_kab = 1.3 * 2.8 * 2.6  # объём кабины, м
r_kab = (v_kab * 3 / (4 * pi)) ** (1 / 3)  # эквивалентный радиус кабины, м

# расчёт коэффициентов экранирования
kh = (1 + 1000 * dst / (2 * r_kab)) ** 2  # магнитный коэффициент
ke_post = 60 * pi * dst * sigma  # электрический постоянный ток
ke_per = {}  # для переменного электрического поля формируем словарь коэффициентов {частота: значение}
# рассчитали коэффициент который не зависит от гармоники
Ce = exp(2 * pi * 0.0025 / 0.01)
#  для каждый гармоники считаем коэффициент электрического поля и заносим в словарь
for fr in harm.keys():
    lam = 300000000 / (fr / 1000000)  # длина волны
    Ze = Z0 * lam / (2 * pi * r_kab)  # волновое сопротивление
    delta = 0.03 * ((10 ** -7) * lam) ** 0.5  # дельта
    A = (delta * Ze / (10 ** -7)) ** 0.5  # первое слагаемое
    B = (lam / r_kab) ** (1 / 3)  # второе слагаемое
    ke_per[fr] = 0.024 * A * B * Ce * 0.001  # итоговый коэффициент


# ОБОРУДОВАНИЕ

x_td1_sr = 0.9  # тяговый двигатель - положение по оси х
dy_td = 0.8  # расстояние от центра симметрии по оси у
r_td = 0.604  # радиус
l_td = 0.66  # длина
z_td = 1  # положение по оси z


# значение вектора магнитного поля из составляющих х и y
def mix(h_x, h_zz):
    return (h_x ** 2 + h_zz ** 2) ** 0.5

# магнитное поле
def magnetic_calc(x_m, z_m, f_m):

    I_h = I * harm.get(f_m)[0]

    Ikp = 0.41 * I_h
    Int = 0.20 * I_h
    Iup = 0.39 * I_h
    Iep = 0.4 * I_h

    # расчёт x и z составляющих магнитного поля от правого рельса для КП
    x = x_m - xp_kp
    h1xkp = Ikp / (4 * pi) * (
                -z_m / ((x + xp) ** 2 + z_m**2) + (z_m - h_kp)/(x ** 2 + (h_kp - z_m)**2))
    h1zkp = Ikp / (4 * pi) * (x + xp) * (
                1 / ((x + xp) ** 2 + z_m ** 2) - 1/(x ** 2 + (h_kp - z_m) ** 2))
    # сумма векторов x и z
    h1kp = mix(h1xkp, h1zkp)
    # расчёт x и z составляющих магнитного поля от левого рельса для КП
    x = x_m - 2*xp - xp_kp
    h2xkp = Ikp / (4 * pi) * (
                -z_m / ((x + xp) ** 2 + z_m ** 2) + (z_m - h_kp) / ((x + 2*xp) ** 2 + (h_kp - z_m) ** 2))
    h2zkp = Ikp / (4 * pi) * (x + xp) * (
                1 / ((x + xp) ** 2 + z_m ** 2) - 1 / ((x + 2*xp) ** 2 + (h_kp - z_m) ** 2))
    # сумма векторов x и z
    h2kp = mix(h2xkp, h2zkp)
    # суммарное поле двух рельс
    hkp = h1kp + h2kp

    # далее аналогично
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

    # ЭП
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

    # КП2
    x = x_m - (xp_kp2 + xp_mid)
    h1xkp_2 = Ikp / (4 * pi) * (
                -z_m / ((x + xp) ** 2 + z_m**2) + (z_m - h_kp)/(x ** 2 + (h_kp - z_m)**2))
    h1zkp_2 = Ikp / (4 * pi) * (x + xp) * (
                1 / ((x + xp) ** 2 + z_m ** 2) - 1/(x ** 2 + (h_kp - z_m) ** 2))
    h1kp_2 = mix(h1xkp_2, h1zkp_2)
    x = x_m - 2*xp - (xp_kp2 + xp_mid)
    h2xkp_2 = Ikp / (4 * pi) * (
                -z_m / ((x + xp) ** 2 + z_m ** 2) + (z_m - h_kp) / ((x + 2*xp) ** 2 + (h_kp - z_m) ** 2))
    h2zkp_2 = Ikp / (4 * pi) * (x + xp) * (
                1 / ((x + xp) ** 2 + z_m ** 2) - 1 / ((x + 2*xp) ** 2 + (h_kp - z_m) ** 2))
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

    # УП2
    x = x_m - (xp_ep2 + xp_mid)
    x2 = -xp + xp_ep2
    h1xep_2 = Iep / (4 * pi) * (
            -z_m / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) + (z_m - h_ep) / (x ** 2 + (h_ep - z_m) ** 2))
    h1zep_2 = Iep / (4 * pi) * (x2 + 2 * xp + x) * (
            1 / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) - 1 / (x ** 2 + (h_ep - z_m) ** 2))
    h1ep_2 = mix(h1xep_2, h1zep_2)
    x = x_m - (xp_ep2 + xp_mid) - 2 * xp
    x2 = -xp + xp_ep2
    h2xep_2 = Iep / (4 * pi) * (
            -z_m / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) + (z_m - h_ep) / ((x + 2 * xp) ** 2 + (h_ep - z_m) ** 2))
    h2zep_2 = Iep / (4 * pi) * (
            (x2 + 2 * xp + x) / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) - (x + 2 * xp) / (
            (x + 2 * xp) ** 2 + (h_ep - z_m) ** 2))
    h2ep_2 = mix(h2xep_2, h2zep_2)
    hep_sec = h1ep_2 + h2ep_2

    # рассчитав модуль поля в точке для каждого провода, составляем список из полученных значений.
    # так как поля НТ, КП, УП складываются, их берём со знаком плюс, а ЕП вычитается - его со знаком минус
    return [hkp, hnt, hup, hkp_scd, hnt_scd, hup_sec, -hep, -hep_sec]


# расчёт магнитного поля
def electric_calc(x_e, z_e, f_e):

    U_h = U * harm.get(f_e)[1]
    U_hep = U_ep * harm.get(f_e)[1]

    ekp = U_h * log(1 + 4 * h_nt * z_e / ((x_e - xp_nt) ** 2 + (h_nt - z_e) ** 2)) / (2 * z_e * log(2 * h_nt / d_nt))
    ent = U_h * log(1 + 4 * h_kp * z_e / ((x_e - xp_kp) ** 2 + (h_kp - z_e) ** 2)) / (2 * z_e * log(2 * h_kp / d_kp))
    eup = U_h * log(1 + 4 * h_up * z_e / ((x_e - xp_up) ** 2 + (h_up - z_e) ** 2)) / (2 * z_e * log(2 * h_up / d_up))
    eep = U_hep * log(1 + 4 * h_ep * z_e / ((x_e - xp_ep) ** 2 + (h_ep - z_e) ** 2)) / (2 * z_e * log(2 * h_ep / d_ep))

    ekp_scd = U_h * log(1 + 4 * h_nt * z_e / ((x_e - xp_nt2 - xp_mid) ** 2 + (h_nt - z_e) ** 2)) / (2 * z_e * log(2 * h_nt / d_nt))
    ent_scd = U_h * log(1 + 4 * h_kp * z_e / ((x_e - xp_kp2 - xp_mid) ** 2 + (h_kp - z_e) ** 2)) / (2 * z_e * log(2 * h_kp / d_kp))
    eup_scd = U_h * log(1 + 4 * h_up * z_e / ((x_e - xp_up2 - xp_mid) ** 2 + (h_up - z_e) ** 2)) / (2 * z_e * log(2 * h_up / d_up))
    eep_scd = U_hep * log(1 + 4 * h_ep * z_e / ((x_e - xp_ep2 - xp_mid) ** 2 + (h_ep - z_e) ** 2)) / (2 * z_e * log(2 * h_ep / d_ep))

    return [ekp, ent, eup, ekp_scd, ent_scd, eup_scd, -eep, -eep_scd]

#  суммирование поля: рассчитывали для каждой точки графика значения для каждой гармоники
def full_field(res_en):
    sum_h, sum_e, sum_g = 0, 0, 0
    # для получения итогового поля, суммируем значения от всех проводов в каждой точке
    # так как ЕП уже записано со знаком минус, они вычитаются
    for en in res_en[0].values():
        sum_h += abs(sum(en[0]))  # магнитная составляющая
        sum_e += abs(sum(en[1]))  # электрическая составляющая
        # для расчёта энергии, перемножаем значение магнитного и электрического поля, затем складываем полученные значения
        # так как ЕП имеют минус и по электричеству, и по магнитному, мы вычитаем значения их энергии
        sum_g += abs(
                 en[0][0] * en[1][0] + en[0][1] * en[1][1] + en[0][2] * en[1][2] + \
                 en[0][3] * en[1][3] + en[0][4] * en[1][4] + en[0][5] * en[1][5] - \
                 (en[0][6] * en[1][6] + en[0][7] * en[1][7]))
    return [sum_h, sum_e, sum_g]


#  расчёт экрана переменного поля
def ekran(en):

    x, y, z = en[1]   # координаты точки

    # расстояние от текущей точки до КТ и НТ - для расчёта лобовых окон
    kppth = LineString([(x, y, z), (x, xp_kp, h_kp)])
    ntpth = LineString([(x, y, z), (x, xp_nt, h_nt)])
    kp_pass = kppth.intersects(frontWindleft) or kppth.intersects(frontWindright)
    nt_pass = ntpth.intersects(frontWindleft) or ntpth.intersects(frontWindright)

    # для каждого провода проверяем, попадает ли текущая точка в тень от бокового окна или нет
    kp_dist = Point(y, z).distance(Point(xp_kp, h_kp))  # направление от точки до провода
    kp_pass |= (kp_dist >= min_kp) and (kp_dist <= max_kp) and (x >= sbor[0]) and (x <= sbor[1])  # пересекает ли окно

    nt_dist = Point(y, z).distance(Point(xp_nt, h_nt))
    nt_pass |= (nt_dist >= min_nt) and (nt_dist <= max_nt) and (x >= sbor[0]) and (x <= sbor[1])

    up_dist = Point(y, z).distance(Point(xp_up, h_up))
    up_pass = (up_dist >= min_up) and (up_dist <= max_up) and (x >= sbor[0]) and (x <= sbor[1])

    ep_dist = Point(y, z).distance(Point(xp_ep, h_ep))
    ep_pass = (ep_dist >= min_ep) and (ep_dist <= max_ep) and (x >= sbor[0]) and (x <= sbor[1])

    kp_sec_d = Point(y, z).distance(Point(xp_kp2+xp_mid, h_kp))
    kp_sec_p = (kp_sec_d >= min_kp2) and (kp_sec_d <= max_kp2) and (x >= sbor[0]) and (x <= sbor[1])

    nt_sec_d = Point(y, z).distance(Point(xp_nt2+xp_mid, h_nt))
    nt_sec_p = (nt_sec_d >= min_nt2) and (nt_sec_d <= max_nt2) and (x >= sbor[0]) and (x <= sbor[1])

    up_sec_d = Point(y, z).distance(Point(xp_up2+xp_mid, h_up))
    up_sec_p = (up_sec_d >= min_up2) and (up_sec_d <= max_up2) and (x >= sbor[0]) and (x <= sbor[1])

    ep_sec_d = Point(y, z).distance(Point(xp_ep2 + xp_mid, h_ep))
    ep_sec_p = (ep_sec_d >= min_ep2) and (ep_sec_d <= max_ep2) and (x >= sbor[0]) and (x <= sbor[1])

    # для каждой точки внутри кабины делим на коэффициент экранирования, если точка в тени окна для поля каждого провода
    if (abs(y) <= 0.5*width) and (z >= floor) and (z <= floor+height):
        if not kp_pass:
            for f in en[0].keys():
                en[0][f][0][0] /= kh
                en[0][f][1][0] /= ke_per[fr]
        if not nt_pass:
            for f in en[0].keys():
                en[0][f][0][1] /= kh
                en[0][f][1][1] /= ke_per[fr]
        if not up_pass:
            for f in en[0].keys():
                en[0][f][0][2] /= kh
                en[0][f][1][2] /= ke_per[fr]
        if not kp_sec_p:
            for f in en[0].keys():
                en[0][f][0][3] /= kh
                en[0][f][1][3] /= ke_per[fr]
        if not nt_sec_p:
            for f in en[0].keys():
                en[0][f][0][4] /= kh
                en[0][f][1][4] /= ke_per[fr]
        if not up_sec_p:
            for f in en[0].keys():
                en[0][f][0][5] /= kh
                en[0][f][1][5] /= ke_per[fr]
        if not ep_pass:
            for f in en[0].keys():
                en[0][f][0][6] /= kh
                en[0][f][1][6] /= ke_per[fr]
        if not ep_sec_p:
            for f in en[0].keys():
                en[0][f][0][7] /= kh
                en[0][f][1][7] /= ke_per[fr]

    return en


# расчёт экрана постоянного поля
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


# сохранение файла с картинкой
def show(name):
    mng = plt.get_current_fig_manager()
    # mng.window.state('zoomed')
    plt.savefig(f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{name}.png")


# построение вида сверху без электровоза
def visual_up(z_srez=z_graph):
    print('График строится..................')
    print(f'Высота среза: {z_srez} метров')

    # границы графика
    Xmin = -0.5
    Xmax = length + 0.5
    Ymin = xp_up * 1.15
    Ymax = xp_mid + abs(Ymin)

    # разбиение по точкам
    x = np.linspace(Xmin, Xmax, dis)
    y = np.linspace(Ymin, Ymax, dis)

    # расчёт значений для каждой точки графика
    every_f = [[[{fr: [magnetic_calc(y_, z_srez, fr), electric_calc(y_, z_srez, fr)] for fr in harm.keys()},
                 (x_, y_, z_srez)] for x_ in x] for y_ in y]

    # перевод значений посчитанных для каждой гармоники каждого провода в одно значение
    summar = [[full_field(x_el) for x_el in y_list] for y_list in every_f]

    # составление графика на магнитную, электрическую составляющую и энергию
    magnetic = [[x_el[0] for x_el in y_list] for y_list in summar]
    electric = [[x_el[1] for x_el in y_list] for y_list in summar]
    energy = [[x_el[2] for x_el in y_list] for y_list in summar]

    # вывод точек, рисование уровней
    def do_graph(content, name_, x_lb='Ось x, метры', y_lb='Ось y, метры'):
        ct = plt.contour(x, y, content, alpha=0.75, colors='black', linestyles='dotted', levels=5)
        plt.clabel(ct, fontsize=10)
        plt.imshow(content, extent=[Xmin, Xmax, Ymax, Ymin], cmap='YlOrRd', alpha=0.95)
        plt.colorbar()

        # рисование и подпись проводов
        for delta_y in [xp_kp, xp_up, xp_nt, xp_kp2+xp_mid, xp_nt2+xp_mid, xp_up2+xp_mid, xp_ep, xp_ep2+xp_mid]:
            plt.hlines(delta_y, Xmin, Xmax, color='black', linewidth=2)
        plt.text(0.1, xp_kp+0.05, 'КП', color='black')
        plt.text(1, xp_nt-0.3, 'НТ', color='black')
        plt.text(0.1, xp_up+0.05, 'УП', color='black')
        plt.text(0.1, xp_ep+0.05, 'ЕП', color='black')
        plt.text(0.1, xp_kp2+xp_mid+0.05, 'КП2', color='black')
        plt.text(1, xp_nt2+xp_mid-0.3, 'НТ2', color='black')
        plt.text(0.1, xp_up2+xp_mid+0.05, 'УП2', color='black')
        plt.text(0.1, xp_ep2+xp_mid+0.05, 'ЕП2', color='black')


        # рисование очертания поезда
        plt.hlines(0.5 * width, 0, length, colors='red', linestyles='--')
        plt.hlines(-0.5 * width, 0, length, colors='red', linestyles='--')
        plt.vlines(0, -0.5 * width, 0.5 * width, colors='red', linestyles='--')
        plt.vlines(length, -0.5 * width, 0.5 * width, colors='red', linestyles='--')
        plt.xlabel(x_lb)
        plt.ylabel(y_lb)

        plt.title(name_)

    # вывод значений магнитного, электрического поля, энергии
    global gph_num
    gph_num += 1
    plt.figure(gph_num)
    name = f'Контактная сеть вид сверху срез {z_srez} м'
    plt.subplot(1, 3, 1)
    do_graph(magnetic, 'Магнитное', x_lb='Ось x, метры', y_lb='Ось y, метры')
    plt.subplot(1, 3, 2)
    do_graph(electric, 'Электрическое', x_lb='Ось x, метры', y_lb='Ось y, метры')
    plt.subplot(1, 3, 3)
    do_graph(energy, 'Энергия', x_lb='Ось x, метры', y_lb='Ось y, метры')
    plt.suptitle(name)
    show(name)

    print('График построен.')

    return every_f


# рисование линий кабины
def fr_kab_lines():
    plt.hlines(height + floor, -0.5 * width, 0.5 * width, colors='red', linestyles='--')
    plt.hlines(floor, -0.5 * width, 0.5 * width, colors='red', linestyles='--')
    plt.hlines(1, -0.5 * width, 0.5 * width, colors='red', linestyles='--')
    plt.vlines(-0.5 * width, 1, height+floor, colors='red', linestyles='--')
    plt.vlines(0.5 * width, 1, height+floor, colors='red', linestyles='--')


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

    # расчёт по точкам
    every_f = [[({fr: (magnetic_calc(y_, z_, fr), electric_calc(y_, z_, fr)) for fr in harm.keys()},
                 (x_chel, y_, z_)) for y_ in y] for z_ in z]

    # суммирование гармоник
    all_field = [[full_field(x_el) for x_el in y_list] for y_list in every_f]
    magnetic = [[x_el[0] for x_el in y_list] for y_list in all_field]
    electric = [[x_el[1] for x_el in y_list] for y_list in all_field]
    energy = [[x_el[2] for x_el in y_list] for y_list in all_field]

    def do_graph(content, name_, x_lb='Ось x, метры', y_lb='Ось y, метры'):

        if name_ == 'Магнитное':
            levels = [10, 20, 30, 50, 100]
        elif name_ == 'Энергия':
            levels = [10000, 20000, 30000, 50000]
        elif name_ == 'Электрическое':
            levels = [5000, 6000, 7000, 8000]
        else:
            levels = 5

        ct = plt.contour(y, z, content, alpha=0.75, colors='black', linestyles='dotted', levels=levels)
        plt.clabel(ct, fontsize=10)
        plt.imshow(content, extent=[Ymin, Ymax, Zmax, Zmin], cmap=cmap, alpha=0.95, norm=colors.LogNorm())

        plt.colorbar()

        # названия проводов
        plt.text(xp_kp, h_kp, 'КП', color='black', fontsize=11)
        plt.text(xp_up, h_up, 'УП', color='black', fontsize=11)
        plt.text(xp_nt, h_nt, 'НТ', color='black', fontsize=11)
        plt.text(xp_ep+0.1, h_ep, 'ЕП', color='black', fontsize=11)
        plt.text(xp_kp2 + xp_mid, h_kp, 'КП2', color='black', fontsize=11)
        plt.text(xp_up2 + xp_mid, h_up, 'УП2', color='black', fontsize=11)
        plt.text(xp_ep2-0.1 + xp_mid, h_ep, 'ЕП2', color='black', fontsize=11)
        plt.text(xp_nt2 + xp_mid, h_nt, 'НТ2', color='black', fontsize=11)

        fr_kab_lines()

        plt.xlabel('Ось y, метры')
        plt.ylabel('Ось z, метры')

        plt.title(name_)


    # вывод графика, рисование уровней
    global gph_num
    gph_num += 1
    plt.figure(gph_num)

    name = 'Контактная сеть вид сбоку'
    plt.subplot(221)
    do_graph(magnetic, 'Магнитное', x_lb='Ось x, метры', y_lb='Ось y, метры')
    plt.subplot(222)
    do_graph(electric, 'Электрическое', x_lb='Ось x, метры', y_lb='Ось y, метры')
    plt.subplot(212)
    do_graph(energy, 'Энергия', x_lb='Ось x, метры', y_lb='Ось y, метры')
    plt.suptitle(name)
    show(name)

    print('График построен.')

    return every_f


# расчёт постоянного поля
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

    # расчёт постоянного поля для каждой точки кабины
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


# рисование окон и стульев в кабине, вид сверху
def kab_lines_up():
    d = 0.12
    cl = 'blue'
    plt.hlines(y_chel-d, x_chel-d, x_chel+d, colors=cl, linestyles='--')
    plt.hlines(y_chel+d, x_chel-d, x_chel+d, colors=cl, linestyles='--')
    plt.hlines(-y_chel-d, x_chel-d, x_chel+d, colors=cl, linestyles='--')
    plt.hlines(-y_chel+d, x_chel-d, x_chel+d, colors=cl, linestyles='--')
    plt.vlines(x_chel-d, y_chel-d, y_chel+d, colors=cl, linestyles='--')
    plt.vlines(x_chel+d, y_chel-d, y_chel+d, colors=cl, linestyles='--')
    plt.vlines(x_chel-d, -y_chel-d, -y_chel+d, colors=cl, linestyles='--')
    plt.vlines(x_chel+d, -y_chel-d, -y_chel+d, colors=cl, linestyles='--')

    plt.hlines(y_chel-d, x_chel+d+0.05, x_chel+d+0.10, colors=cl, linestyles='--')
    plt.hlines(y_chel+d, x_chel+d+0.05, x_chel+d+0.10, colors=cl, linestyles='--')
    plt.hlines(-y_chel-d, x_chel+d+0.05, x_chel+d+0.10, colors=cl, linestyles='--')
    plt.hlines(-y_chel+d, x_chel+d+0.05, x_chel+d+0.10, colors=cl, linestyles='--')
    plt.vlines(x_chel+d+0.05, y_chel-d, y_chel+d, colors=cl, linestyles='--')
    plt.vlines(x_chel+d+0.10, y_chel-d, y_chel+d, colors=cl, linestyles='--')
    plt.vlines(x_chel+d+0.05, -y_chel-d, -y_chel+d, colors=cl, linestyles='--')
    plt.vlines(x_chel+d+0.10, -y_chel-d, -y_chel+d, colors=cl, linestyles='--')

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

    plt.hlines(0.5*width-0.01, 0, length, colors=cl, linestyles='--')
    plt.hlines(-0.5*width+0.01, 0, length, colors=cl, linestyles='--')
    plt.vlines(0.01, 0.5*width, -0.5*width, colors=cl, linestyles='--')
    plt.vlines(length-0.01, 0.5*width, -0.5*width, colors=cl, linestyles='--')


# рисование стульев в кабине вид спереди
def kab_lines_front():
    d = 0.13
    cl = 'blue'
    plt.hlines(z_chair, y_chel-d, y_chel+d, colors=cl, linestyles='--')
    plt.hlines(z_chair, -y_chel-d, -y_chel+d, colors=cl, linestyles='--')
    plt.hlines(z_chair-0.05, y_chel-d, y_chel+d, colors=cl, linestyles='--')
    plt.hlines(z_chair-0.05, -y_chel-d, -y_chel+d, colors=cl, linestyles='--')

    plt.vlines(y_chel-d, z_chair, z_chair-0.05, colors=cl, linestyles='--')
    plt.vlines(y_chel+d, z_chair, z_chair-0.05, colors=cl, linestyles='--')
    plt.vlines(-y_chel-d, z_chair, z_chair-0.05, colors=cl, linestyles='--')
    plt.vlines(-y_chel+d, z_chair, z_chair-0.05, colors=cl, linestyles='--')

    d = 0.12
    plt.hlines(z_chair+0.05, y_chel-d, y_chel+d, colors=cl, linestyles='--')
    plt.hlines(z_chair+0.05+2*d, y_chel-d, y_chel+d, colors=cl, linestyles='--')
    plt.hlines(z_chair+0.05, -y_chel-d, -y_chel+d, colors=cl, linestyles='--')
    plt.hlines(z_chair+0.05+2*d, -y_chel-d, -y_chel+d, colors=cl, linestyles='--')

    plt.vlines(y_chel-d, z_chair+0.05, z_chair+0.05+2*d, colors=cl, linestyles='--')
    plt.vlines(y_chel+d, z_chair+0.05, z_chair+0.05+2*d, colors=cl, linestyles='--')
    plt.vlines(-y_chel-d, z_chair+0.05, z_chair+0.05+2*d, colors=cl, linestyles='--')
    plt.vlines(-y_chel+d, z_chair+0.05, z_chair+0.05+2*d, colors=cl, linestyles='--')


# рисование очертаний ТЭД
def ted_lines_front():
    plt.hlines(z_td + 0.5*r_td, dy_td - 0.5*l_td, dy_td + 0.5*l_td, colors='blue', linestyles='--')
    plt.hlines(z_td - 0.5*r_td, dy_td - 0.5*l_td, dy_td + 0.5*l_td, colors='blue', linestyles='--')
    plt.hlines(z_td + 0.5*r_td, -dy_td - 0.5*l_td, -dy_td + 0.5*l_td, colors='blue', linestyles='--')
    plt.hlines(z_td - 0.5*r_td, -dy_td - 0.5*l_td, -dy_td + 0.5*l_td, colors='blue', linestyles='--')

    plt.vlines(dy_td - 0.5*l_td, z_td - 0.5*r_td, z_td + 0.5*r_td, colors='blue', linestyles='--')
    plt.vlines(dy_td + 0.5*l_td, z_td - 0.5*r_td, z_td + 0.5*r_td, colors='blue', linestyles='--')
    plt.vlines(-dy_td - 0.5*l_td, z_td - 0.5*r_td, z_td + 0.5*r_td, colors='blue', linestyles='--')
    plt.vlines(-dy_td + 0.5*l_td, z_td - 0.5*r_td, z_td + 0.5*r_td, colors='blue', linestyles='--')


# формирование треугольников
def triang_do(triangulation, scalar_, name_, x_lb='Ось x, метры', y_lb='Ось y, метры', lev=5):
    plt.axis('equal')
    plt.tricontourf(triangulation, scalar_, cmap=cmap)
    plt.colorbar()
    tcf = plt.tricontour(triangulation, scalar_, alpha=0.75, colors='black', linestyles='dotted', levels=lev)
    plt.clabel(tcf, fontsize=10)

    plt.xlabel(x_lb)
    plt.ylabel(y_lb)

    plt.title(name_)


# вид сверху внутри кабины
def visual_up_locomotive(ext_f):
    print('График строится..................')

    # границы графика
    Xmin = 0
    Xmax = length
    Ymax = -0.5 * width
    Ymin = -Ymax

    # расчёт
    inside = [[full_field(ekran(el)) for el in y_list if (el[1][0] >= Xmin) and (el[1][0] <= Xmax)]
              for y_list in ext_f if abs(y_list[0][1][1]) <= 0.5 * width]

    # x_ln = np.linspace(Xmin, Xmax, len(inside[0]))
    # y_ln = np.linspace(Ymin, Ymax, len(inside))

    # выделение отдельно магнитной, электрической, энергетической составляющей
    magnetic = [[x_el[0] for x_el in y_list] for y_list in inside]
    electric = [[x_el[1] for x_el in y_list] for y_list in inside]
    energy = [[x_el[2] for x_el in y_list] for y_list in inside]

    # функция вывода графика
    def graph_do(znach, name_, x_lb='', y_lb=''):
        # ct = plt.contour(x_ln, y_ln, znach, alpha=0.95, colors='black', linestyles='dotted', levels=5)
        # plt.clabel(ct, fontsize=10)
        plt.imshow(znach, extent=[Xmin, Xmax, Ymax, Ymin], cmap='YlOrRd', alpha=0.95, norm=colors.LogNorm())
        plt.colorbar()

        plt.xlabel(x_lb)
        plt.ylabel(y_lb)

        plt.title(name_)

    # вывод графика три раза
    global gph_num
    gph_num += 1
    plt.figure(gph_num)
    name = 'Вид сверху кабина переменное экран'
    plt.subplot(1, 3, 1)
    kab_lines_up()
    graph_do(magnetic, 'Магнитное', x_lb='Ось x, метры', y_lb='Ось y, метры')
    plt.subplot(1, 3, 2)
    kab_lines_up()
    graph_do(electric, 'Электрическое', x_lb='Ось x, метры')
    plt.subplot(1, 3, 3)
    kab_lines_up()
    graph_do(energy, 'Общее', x_lb='Ось x, метры',)

    show(name)


# вид внутри кабины сверху для постоянного поля
def visual_up_post():
    print('Расчёт поля от тяговых двигателей....')
    # граница графика
    Xmin = 0
    Xmax = length
    Ymax = -0.5 * width
    Ymin = -Ymax

    # разбиение на узлы
    dis = 60
    x_ln = np.linspace(Xmin, Xmax, dis, endpoint=True)
    y_ln = np.linspace(Ymin, Ymax, dis, endpoint=True)

    # функция отрисовки
    def graph_do(znach, name_, x_lb='', y_lb=''):
        ct = plt.contour(x_ln, y_ln, znach, alpha=0.95, colors='black', linestyles='dotted', levels=5)
        plt.clabel(ct, fontsize=10)
        plt.imshow(znach, extent=[Xmin, Xmax, Ymax, Ymin], cmap='YlOrRd', alpha=0.95, norm=colors.LogNorm())
        plt.colorbar()

        plt.xlabel(x_lb)
        plt.ylabel(y_lb)

        plt.title(name_)

    # расчёт поля
    ted_field = ted_field_calc(x_ln, y_ln, I_ted, U_ted, 5)

    # выделение составляющих поля
    magnetic = np.array([el[0][0]/kh for el in ted_field]).reshape(len(y_ln), len(x_ln))
    electric = np.array([el[0][1]/ke_post for el in ted_field]).reshape(len(y_ln), len(x_ln))
    energy = np.array([el[0][0]/kh * el[0][1]/ke_post for el in ted_field]).reshape(len(y_ln), len(x_ln))

    # вывод графика три раза
    global gph_num
    gph_num += 1
    plt.figure(gph_num)
    name = 'Вид сверху кабина постоянное'
    plt.subplot(1, 3, 1)
    graph_do(magnetic, 'Магнитное', x_lb='Ось x, метры', y_lb='Ось y, метры')
    kab_lines_up()
    plt.subplot(1, 3, 2)
    graph_do(electric, 'Электрическое', x_lb='Ось x, метры')
    kab_lines_up()
    plt.subplot(1, 3, 3)
    graph_do(energy, 'Общее', x_lb='Ось x, метры')
    kab_lines_up()

    show(name)

    print('График построен.')


# вид спереди на кабину, переменное поле
def visual_front_locomotive(ext_f):
    # граница графика
    Ymin, Ymax = -0.6*width, 0.6*width
    Zmin, Zmax = floor+height+1, 0.1

    # расчёт
    ekran_ = [[ekran(y_el) for y_el in z_list if abs(y_el[1][1]) <= Ymax] for z_list in ext_f
              if z_list[0][1][2] < Zmin]

    # суммирование значений и выделение магнитной, электрической и энергетической составляющей
    summar = [[full_field(x_el) for x_el in y_list] for y_list in ekran_]
    magnetic = [[x_el[0] for x_el in y_list] for y_list in summar]
    electric = [[x_el[1] for x_el in y_list] for y_list in summar]
    energy = [[x_el[2] for x_el in y_list] for y_list in summar]

    # формирование расчёта гармоник для положения человека
    y_ln = np.linspace(Ymin, Ymax, len(ekran_[0]))
    z_ln = np.linspace(Zmin, Zmax, len(ekran_))
    chel_y = np.where(y_ln == max([y_ for y_ in y_ln if y_ <= y_chel]))[0][0]
    chel_z = np.where(z_ln == max([z_ for z_ in z_ln if z_ <= z_chel]))[0][0]

    # отрисовка графика
    def graph_do(znach, name_, x_lb='', y_lb=''):
        # ct = plt.contour(y_ln, z_ln, znach, alpha=0.95, colors='black', linestyles='dotted', levels=5)
        # plt.clabel(ct, fontsize=10)
        plt.imshow(znach, extent=[Ymin, Ymax, Zmax, Zmin],  cmap=cmap,  alpha=0.95, norm=colors.LogNorm())
        plt.colorbar()
        fr_kab_lines()

        plt.xlabel(x_lb)
        plt.ylabel(y_lb)
        plt.title(name_)

    # вывод графика три раза
    global gph_num
    gph_num += 1
    plt.figure(gph_num)
    name = 'Cпереди кабина переменное с экраном'
    plt.subplot(1, 3, 1)
    graph_do(magnetic, 'Магнитное', x_lb='Ось x, метры', y_lb='Ось y, метры')
    kab_lines_front()
    plt.subplot(1, 3, 2)
    graph_do(electric, 'Электрическое', x_lb='Ось x, метры')
    kab_lines_front()
    plt.subplot(1, 3, 3)
    graph_do(energy, 'Общее', x_lb='Ось x, метры', )
    kab_lines_front()
    plt.suptitle(name)
    show(name)

    # отрисовка поля по гарминикам, магнитного
    gph_num += 1
    plt.figure(gph_num)
    name = 'Гармоники магнитное вид спереди'
    i = 0
    chel_harm_h = []
    for fr in harm.keys():
        i += 1
        plt.subplot(3, 3, i)
        data = [[sum(el[0][fr][0]) for el in lst]for lst in ekran_]
        chel_harm_h.append(data[chel_z][chel_y])
        graph_do(data, '', y_lb=str(fr))
        kab_lines_front()
    plt.subplot(3, 3, 9)
    plt.bar(range(0, len(harm.keys())), chel_harm_h)
    plt.suptitle(name)
    show(name)

    # отрисовка поля по гармоникам, электрического
    gph_num += 1
    plt.figure(gph_num)
    name = 'Гармоники электрическое вид спереди'
    i = 0
    chel_harm_e = []
    for fr in harm.keys():
        i += 1
        plt.subplot(3, 3, i)
        data = [[sum(el[0][fr][1]) for el in lst]for lst in ekran_]
        chel_harm_e.append(data[chel_z][chel_y])
        graph_do(data, '', y_lb=str(fr))
        kab_lines_front()
    plt.subplot(3, 3, 9)
    plt.bar(range(0, len(harm.keys())), chel_harm_e)
    plt.suptitle(name)
    show(name)

    print('Гармоники магнитного поля для человека', chel_harm_h,
          'Гармоники электрического поля для человека', chel_harm_e,
          sep='\n')


# вид спереди постоянное поле
def visual_front_post():
    print('Расчёт поля от тяговых двигателей')
    # разбиение на узлы, границы графика, разбиение на точки
    dis_y, dis_z = 60, 60
    Ymin, Ymax = -0.6*width, 0.6*width
    Zmin, Zmax = floor+height+1, 0.1
    y_ln = np.linspace(Ymin, Ymax, dis_y, endpoint=True)
    z_ln = np.linspace(Zmin, Zmax, dis_z, endpoint=True)

    # функция отрисовки
    def graph_do(znach, name_, x_lb='', y_lb='', lev=5):
        if lev:
            ct = plt.contour(y_ln, z_ln, znach, alpha=0.95, colors='black', linestyles='dotted', levels=lev)
            plt.clabel(ct, fontsize=10)
        plt.imshow(znach, extent=[Ymin, Ymax, Zmax, Zmin],  cmap=cmap,  alpha=0.95, norm=colors.LogNorm())
        plt.colorbar()

        plt.xlabel(x_lb)
        plt.ylabel(y_lb)
        plt.title(name_)

    # расчёт поля
    ted_field = ted_field_calc(y_ln, z_ln, I_ted, U_ted, 5, type_='FRONT')

    # вызов отрисовки вида спереди как с экраном, так и без
    global gph_num
    gph_num += 1
    plt.figure(gph_num)
    name = 'Вид спереди постоянное'
    all_f_list = [el[0][0] * el[0][1] for el in ted_field]
    print(len(all_f_list))
    all_f = np.array(all_f_list).reshape(len(z_ln), len(y_ln))
    plt.subplot(1, 2, 1)
    graph_do(all_f, 'Энергия', x_lb='Ось y, метры', y_lb='Ось z, метры')
    fr_kab_lines()
    plt.title('Без экрана')
    front_ekran = [ekran_post(el) for el in ted_field]

    plt.subplot(1, 2, 2)

    # формирование графика только внутри кабины
    summar = np.array([el[0][0] * el[0][1] for el in front_ekran]).reshape(len(z_ln), len(y_ln))
    graph_do(summar, 'Энергия', x_lb='Ось y, метры', y_lb='Ось z, метры', lev=0)
    fr_kab_lines()
    plt.title('С экраном')

    show(name)

    # границы графика, пересборка точек
    Ymin, Ymax = -0.5*width, 0.5*width
    Zmax, Zmin = floor, floor+height
    z_points = [el[1][2] for el in ted_field if (el[1][2] > Zmax) and (el[1][2] < Zmin)]
    z_kab = list(sorted(set(z_points), reverse=True))
    y_points = [el[1][1] for el in ted_field if abs(el[1][1]) < Ymax]
    y_kab = list(sorted(set(y_points)))
    kabina = [[el for el in ted_field if (el[1][2] == z_) and (abs(el[1][1]) < Ymax)] for z_ in z_kab]

    # формирование магнитной, электрической, энергетической составляющей
    magnetic = [[el[0][0] for el in z_list] for z_list in kabina]
    electric = [[el[0][1] for el in z_list] for z_list in kabina]
    energy = [[el[0][0] * el[0][0] for el in z_list] for z_list in kabina]

    y_ln, z_ln = y_kab, z_kab
    # отрисовка графиков
    gph_num += 1
    plt.figure(gph_num)
    name = 'Вид спереди кабина постоянное экран'
    plt.subplot(1, 3, 1)
    kab_lines_front()
    graph_do(magnetic, 'Магнитное', x_lb='Ось y, метры', lev=5)
    plt.subplot(1, 3, 2)
    kab_lines_front()
    graph_do(electric, 'Электрическое', x_lb='Ось y, метры', lev=5)
    plt.subplot(1, 3, 3)
    kab_lines_front()
    graph_do(energy, 'Общее', x_lb='Ось y, метры', y_lb='Ось z, метры', lev=5)
    plt.suptitle(name)
    show(name)

    print('График построен.')


# ВЫВОД ПАРАМЕТРОВ

print('\nПараметры сети')
print(f'Высота КП: {h_kp} м')
print(f'Высота НЧ: {h_nt} м')
print(f'Высота УП: {h_up} м')
print(f'Напряжение: {U} Вольт')
print(f'Суммарный ток: {I} Ампер')
print(f'Напряжение ТЭД: {U_ted} Вольт')
print(f'Ток ТЭД: {I_ted} Ампер')

# ПОСТРОЕНИЕ ГРАФИКА

gph_num = 0
print('\nБез электровоза')
cont_f_up = visual_up()
visual_up(1.8)
visual_up(7.0)

print('\nВид спереди')
cont_f_front = visual_front()

print('\nПоле в кабине сверху')
visual_up_locomotive(cont_f_up)
visual_up_post()

print('\nПоле в кабине спереди')
visual_front_locomotive(cont_f_front)
visual_front_post()

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

chel_f_post = ted_field_calc([y_chel], [z_chel], I_ted, U_ted, 5, type_='FRONT')[0][0]
ekran_post_ = chel_f_post[0] / kh * chel_f_post[1] / ke_post
print('\nПостоянное поле с экраном %.4f' % ekran_post_)
Dco = ekran_post_ * ti * S * p
Dpo = Dco / b
print('Удельная суточная доза поглощённой энергии: %.8f' % Dpo)

plt.show()
