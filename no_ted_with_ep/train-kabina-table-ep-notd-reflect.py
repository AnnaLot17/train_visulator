from math import log, pi, atan, exp

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, LineString, Point

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

dis = 100  # дискретизация расчётов (размерность таблицы)
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
d_kp = 12.81 / 1000  # mm
d_nt = 12.5 / 1000  # mm
d_up = 17.5 / 1000  # mm
d_ep = 12.5 / 1000
h_kp = 6.0  # КП
h_nt = 7.8  # НТ
h_up = 8.0  # УП
h_ep = 8.4  # ЕП

# ДАННЫЕ О ЛОКОМОТИВЕ

length = 1.3  # длина кабины
all_length = 15.2  # длина всего локомотива
width = 2.8  # ширина кабины
height = 2.6  # высота кабины
# min_x, max_x, min_y, max_y, min_z, max_z
bor = [0.2, 0.6, -1.2, 1.2, floor+1.5, floor+2.2]  # узлы окна
# min_x, max_x, min_z, max_z
sbor = [0.3, 1, floor + 1, floor + 2.2]  # узлы для бокового окна

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
min_nt = Point(0.5*width, sbor[3]).distance(Point(xp_nt, h_nt)) # луч нижней границы тени от НТ
max_nt = Point(0.5*width, sbor[2]).distance(Point(xp_nt, h_nt)) # луч верхней границы тени от НТ

min_kp = Point(0.5*width, sbor[3]).distance(Point(xp_kp, h_kp)) # далее аналогично для остальных проводов
max_kp = Point(0.5*width, sbor[2]).distance(Point(xp_kp, h_kp))

min_up_l = Point(-0.5 * width, sbor[3]).distance(Point(xp_up, h_up))
max_up_l = Point(-0.5 * width, sbor[2]).distance(Point(xp_up, h_up))
min_up_r = Point(0.5 * width, sbor[3]).distance(Point(xp_up, h_up))
max_up_r = Point(0.5 * width, sbor[2]).distance(Point(xp_up, h_up))

min_ep_l = Point(-0.5*width, sbor[3]).distance(Point(xp_ep, h_ep))
max_ep_l = Point(-0.5*width, sbor[2]).distance(Point(xp_ep, h_ep))
min_ep_r = Point(0.5*width, sbor[3]).distance(Point(xp_ep, h_ep))
max_ep_r = Point(0.5*width, sbor[2]).distance(Point(xp_ep, h_ep))

# ЭКРАН
# стекло - высчитываем d для подсчёта энергии преломлённой волны
e1 = 1
e2 = 4
mu1 = 1
mu2 = 0.99

n1 = (e1*mu1) ** 0.5
n2 = (e2*mu2) ** 0.5
k_glass = ((n1-n2)/(n1+n2)) ** 2
d_glass = 1 - k_glass

# РАСЧЁТЫ


# по теореме Пифагора расчёт значения вектора из составляющих х и y
def mix(h_x, h_zz):
    return (h_x ** 2 + h_zz ** 2) ** 0.5


# магнитное поле гармоники f для заданной координаты x и z
def magnetic_calc(x_m, z_m, f_m):
    # общая сила тока гармоники
    I_h = I * harm.get(f_m)[0]

    # сила тока по проводам
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
    # сумма (по т.Пифагора) векторов x и z
    h1kp = mix(h1xkp, h1zkp)
    # расчёт x и z составляющих магнитного поля от левого рельса для КП
    x = x_m - 2*xp - xp_kp
    h2xkp = Ikp / (4 * pi) * (
                -z_m / ((x + xp) ** 2 + z_m ** 2) + (z_m - h_kp) / ((x + 2*xp) ** 2 + (h_kp - z_m) ** 2))
    h2zkp = Ikp / (4 * pi) * (x + 2 * xp) * (
                1 / ((x + xp) ** 2 + z_m ** 2) - 1 / ((x + 2*xp) ** 2 + (h_kp - z_m) ** 2))
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
    h2znt = Int / (4 * pi) * (x + 2 * xp) * (
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
            (x2 + 2 * xp + x) / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) - (x + 2 * xp) / ((x + 2 * xp) ** 2 + (h_up - z_m) ** 2))
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

    # результат выполнения этой функции - значения магнитных полей КП, НТ, УП для выбранной гармоники
    return [hkp, hnt, hup, -hep]


# расчёт электрического поля для гармоники f в точке x, z
def electric_calc(x_e, z_e, f_e):
    # напряжение гармоники
    U_h = U * harm.get(f_e)[1]

    # электрическое поле от каждого провода
    ekp = U_h * log(1 + 4 * h_nt * z_e / ((x_e - xp_nt) ** 2 + (h_nt - z_e) ** 2)) / (2 * z_e * log(2 * h_nt / d_nt))
    ent = U_h * log(1 + 4 * h_kp * z_e / ((x_e - xp_kp) ** 2 + (h_kp - z_e) ** 2)) / (2 * z_e * log(2 * h_kp / d_kp))
    eup = U_h * log(1 + 4 * h_up * z_e / ((x_e - xp_up) ** 2 + (h_up - z_e) ** 2)) / (2 * z_e * log(2 * h_up / d_up))
    eep = U_h * log(1 + 4 * h_ep * z_e / ((x_e - xp_ep) ** 2 + (h_ep - z_e) ** 2)) / (2 * z_e * log(2 * h_ep / d_ep))

    # результат - список значений полей от каждого провода
    return [ekp, ent, eup, -eep]


# суммироввание всех полей для каждой точки:
def full_field(res_en):
    sum_h, sum_e, sum_g = 0, 0, 0
    for en in res_en[0].values():
        sum_h += sum(en[0])  # магнитная составляющая
        sum_e += sum(en[1])  # электрическая составляющая
        # для расчёта энергии, перемножаем значения магнитного и электрического поля для каждого провода,
        # затем складываем полученные значения
        sum_g += en[0][0] * en[1][0] + en[0][1] * en[1][1] + en[0][2] * en[1][2] - en[0][3] * en[1][3]
    # возвращаем значения магнитной, электрической и энергетической составляющей
    return [sum_h, sum_e, sum_g]


#  расчёт экрана переменного поля
def ekran(en):

    x, y, z = en[1]  # координаты точки

    # направление от текущей точки до проводов - для расчёта лобовых окон
    kppth = LineString([(x, y, z), (x, xp_kp, h_kp)])
    ntpth = LineString([(x, y, z), (x, xp_nt, h_nt)])
    uppth = LineString([(x, y, z), (x, xp_up, h_up)])
    eppth = LineString([(x, y, z), (x, xp_ep, h_ep)])
    # проверяем, попадает ли лобовое окно по направлению от текущей точки до проводов
    kp_pass = kppth.intersects(frontWindleft) or kppth.intersects(frontWindright)
    nt_pass = ntpth.intersects(frontWindleft) or ntpth.intersects(frontWindright)
    up_pass = uppth.intersects(frontWindleft) or uppth.intersects(frontWindright)
    ep_pass = eppth.intersects(frontWindleft) or eppth.intersects(frontWindright)

    # для каждого провода проверяем, попадает ли текущая точка в тень от бокового окна или нет
    kp_dist = Point(y, z).distance(Point(xp_kp, h_kp))  # направление от точки до провода
    # есть ли на пути этого направления окно
    # учитываем значение для лобового стекла логическим сложением
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

    # для каждой точки внутри кабины проверяем, проходит ли для неё какое-либо поле через стекло
    # сталь: электрическое поле полностью отражается, магнитное полностью затухает
    # стекло: и электрическое, и магнитное домножаются на d_glass по формуле:
    # Эпрел = Эпад*d = (ExH)*d = E*d x H*d
    if (abs(y) <= 0.5*width) and (z >= floor) and (z <= floor+height):
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
        if ep_pass:
            # поле ЕП через стекло
            for f in en[0].keys():
                en[0][f][0][3] *= d_glass
                en[0][f][1][3] *= d_glass
        if not (kp_pass or nt_pass or up_pass or ep_pass):
            # если ни через одно стекло не проходит, значит тут сталь, т.е. поле близко нулю
            # принимаем поле от всех проводов равным 1 в этих точках для удобства отображения на графике
            for f in en[0].keys():
                en[0][f][0] = [0, 0, 0, 0]
                en[0][f][1] = [0, 0, 0, 0]

    return en


# СОСТАВЛЕНИЕ ТАБЛИЦ

# вид спереди без локомотива
def visual_front():
    # границы таблицы
    Ymax = 1 * max(xp, width) * 1.15
    Ymin = xp_up * 1.2
    Zmax = 0.1
    Zmin = max(h_kp, h_nt, h_up) * 1.1

    # разбиение по точкам
    y = np.linspace(Ymin, Ymax, dis)
    z = np.linspace(Zmin, Zmax, dis)

    # расчёт значений полей для каждой точки
    every_f = [[[{fr: [magnetic_calc(y_, z_, fr), electric_calc(y_, z_, fr)] for fr in harm.keys()},
                 [x_chel, y_, z_]] for y_ in y] for z_ in z]

    return every_f


# вид спереди 
def visual_front_locomotive(ext_f):
    # границы
    Ymin, Ymax = -0.5*width, 0.5*width
    Zmin, Zmax = floor+height, floor

    # выборка из общего поля фрагмента с электровозом и применение экрана
    ekran_ = [[ekran(y_el) for y_el in z_list if abs(y_el[1][1]) <= Ymax] for z_list in ext_f
              if z_list[0][1][2] <= Zmin and z_list[0][1][2] >= Zmax]

    # суммирование для получения конечного значения в каждой точке
    summar = [[full_field(y_el) for y_el in z_list] for z_list in ekran_]
    energy = np.array([[y_el[2] for y_el in z_list] for z_list in summar])

    # разбиение по точкам
    y_ln = np.linspace(Ymin, Ymax, len(ekran_[0]))
    z_ln = np.linspace(Zmin, Zmax, len(ekran_))

    # составление таблицы
    def table_out(znach, f=0, t=0, ln=10):
        # вывод шапки значений y
        for y in y_ln:
            print(f'{y:.3f}'.ljust(ln), end='', file=rf)
        print('y / z\n', file=rf)
        # вывод построчно значений
        for no, y_list in enumerate(znach):
            for dt in y_list:
                if f:
                    E = dt[0][f][0][0]*dt[0][f][1][0] +\
                        dt[0][f][0][1]*dt[0][f][1][1] +\
                        dt[0][f][0][2]*dt[0][f][1][2] -\
                        dt[0][f][0][3]*dt[0][f][1][3]
                    print(f'{E:.3f}'.ljust(ln), end='', file=rf)
                else:
                    print(f'{dt:.3f}'.ljust(ln), end='', file=rf)
            print(f'| {z_ln[no]:.3f}', file=rf)
        print('\n', file=rf)

    # открываем файл на запись
    rf = open('peremennoe_pole.txt', 'w')

    # шапка
    print('Верхняя строка - ось y, метры. Крайний правый столбец - ось z, метры\n', file=rf)

    # вывод значений энергии в таблицу
    print('ЭНЕРГИЯ вид спереди кабина\n', file=rf)
    print('Общее\n', file=rf)
    table_out(energy, ln=12)
    print('Гармоники\n', file=rf)
    for fr in harm.keys():
        print(f'{fr} Гц\n', file=rf)
        table_out(ekran_, f=fr, t=1)

    rf.close()  # закрываем файл


# ВЫВОД ПАРАМЕТРОВ

print('\nПараметры сети')
print(f'Высота КП: {h_kp} м')
print(f'Высота НЧ: {h_nt} м')
print(f'Высота УП: {h_up} м')
print(f'Высота ЕП: {h_ep} м')
print(f'Напряжение: {U} Вольт')
print(f'Суммарный ток: {I} Ампер')
print(f'Высота среза: {z_graph} метров')

# РАСЧЁТ ТАБЛИЦ

print('\nРасчёт поля........\n')
cont_f_front = visual_front()
visual_front_locomotive(cont_f_front)

# РАСЧЁТ СТАТИСТИКИ

print('СТАТИСТИКА\n')
S = (a * b / 3600) ** 1 / 2
p = ti / 24  # статистическая вероятность воздействия

chel_f_per = [{fr: (magnetic_calc(y_chel, z_chel, fr), electric_calc(y_chel, z_chel, fr)) for fr in harm.keys()},
              (x_chel, y_chel, z_chel)]
no_ekran_per = full_field(chel_f_per)[2]
print('\nПеременное поле без экрана: %.4f' % no_ekran_per)

ekran_per = full_field(ekran(chel_f_per))[2]
print('Перменное поле с экраном %.4f' % ekran_per)
Dco = ekran_per * ti * S * p
Dpo = Dco / b
print('Удельная суточная доза поглощённой энергии: %.4f' % Dpo)

