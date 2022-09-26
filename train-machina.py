from math import pi, log, exp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri
from datetime import datetime
import matplotlib.colors as colors
plt.style.use('seaborn-white')


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


sum_harm_I = sum([v[0] for v in harm.values()])
sum_harm_U = sum([v[1] for v in harm.values()])

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


# ПОЛОЖЕНИЕ ШИН И ОБОРУДОВАНИЯ
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
x_vu2 = 6.700  # выпрямтельная установка дальняя
y_vu1 = 0.600 + 0.15 - 0.2
y_vu2 = y_vu1 + 1.4
l_vu = 1.120
h_vu = 0.4472
z_vu = 0.630  #TODO уточнить!
d_vu = 1.1
w_vu = 0.5  #TODO уточнить!


x_cp1 = x_vu1 + 1 - d_vu  # сглаживающий реактор
x_cp2 = x_vu2 - 1 + d_vu
y_cp = y_vu1 + 0.9
l_cp = 0.8
h_cp = 0.8
z_cp = 0.6
w_cp = -0.6  #TODO уточнить

x_td1_sr = 0.9  # тяговый двигатель
dy_td = 0.8
r_td = 0.604
l_td = 0.66
z_td = 1
kol_par = 1.5

I_vu_cp = 3150
U_vu_cp = 1400

I_cp_td = 880
U_cp_td = 950

I_vu = 1850
U_vu = 1500
n_vu = 7

I_cp = 3150
U_cp = 1400
n_cp = 1

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

koef_ekr_h_setka, koef_ekr_e_setka = 0, 0
for fr in harm.keys():
    lam = 300000000 / fr  # длина волны
    Zh = Z0 * 2 * pi * re_v / lam
    ekr_h = 0.012 * (d_v * Zh / ro) ** 0.5 * (lam / re_v) ** (1 / 3) * exp(pi * ds / (s_ - ds))
    koef_ekr_h_setka += harm[fr][0] / ekr_h

    delta = 0.016 / (fr ** 0.5)
    ekr_e = 60 * pi * 1 * delta / (ro * s_ * 2.83 * (ds ** 0.5)) * exp(ds / delta)
    koef_ekr_e_setka += harm[fr][1] / ekr_e

koef_ekr_h_splosh_v = 1 / (1 + (0.66 * metal_mu * d_v / re_v))
koef_ekr_h_splosh_z = 1 / (1 + (0.66 * metal_mu * d_z / re_z))
koef_ekr_e_splosh = 1 / ke_metal

# ШИНЫ И ОБОРУДОВАНИЕ

# ПЕРЕМЕННОЕ


# ПОСТОЯННОЕ

# todo итого: нужно ли ЗДЕСЬ напряжение и ток или в другом месте?
# формат: [(координаты начала), (смещение конца относительно начала), (ток, напряжение)]

# TODO уточнить высоту ВУ-СР шины

# todo для проверки
sh_test = [[(1, 1.4, z_vu),  (5, 0,0)]]

# ВУ-СР
sh_vu_cp = [
            [(x_vu1, y_vu1 + 0.2, z_vu), (1, 0, 0)],
            [(x_vu1, y_vu2 + 0.2, z_vu), (1, 0, 0)],
            [(x_vu1 + 1, y_vu1 + 0.2, z_vu), (0, 1.4, 0)],
            [(x_vu1+1, y_vu1 + 0.83, z_vu), (-0.6, 0, 0)],

            [(x_vu2, y_vu1 + 0.2, z_vu), (-1, 0, 0)],
            [(x_vu2, y_vu2 + 0.2, z_vu), (-1, 0, 0)],
            [(x_vu2 - 1, y_vu1 + 0.2, z_vu), (0, 1.4, 0)],
            [(x_vu2-1, y_vu1 + 0.83, z_vu), (0.6, 0, 0)]]

# СР-ТД
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

vu = [[(x_vu1, x_vu1 - l_vu, y_vu1, y_vu1 + h_vu, z_vu, z_vu+w_vu)],
      [(x_vu1, x_vu1 - l_vu, y_vu2, y_vu2 + h_vu, z_vu, z_vu+w_vu)],
      [(x_vu2, x_vu2 + l_vu, y_vu1, y_vu1 + h_vu, z_vu, z_vu+w_vu)],
      [(x_vu2, x_vu2 + l_vu, y_vu2, y_vu2 + h_vu, z_vu, z_vu+w_vu)]]

cp = [[(x_cp1, x_cp1 - l_cp, y_cp - 0.5 * h_cp, y_cp + 0.5 * h_cp, z_cp, z_cp+w_cp), (3150, 1400, 1)],
      [(x_cp2, x_cp2 + l_cp, y_cp - 0.5 * h_cp, y_cp + 0.5 * h_cp, z_cp, z_cp+w_cp), (3150, 1400, 1)]]


# TODO сколько их?
# TODO как рисуем?
# TODO как считаем?
# ted = [[x_td1_sr - l_td * 0.5, x_td1_sr + l_td * 0.5,
#         0.5 * (width + h_td), 0.5 * (width - h_td), 1 - z_td * 0.5, 1 + z_td * 0.5],
#        [x_td1_sr + kol_par - l_td * 0.5, x_td1_sr + kol_par + l_td * 0.5,
#         0.5 * (width + h_td), 0.5 * (width - h_td), 1 - z_td * 0.5, 1 + z_td * 0.5],
#        [x_td2_sr - kol_par - l_td * 0.5, x_td2_sr - kol_par + l_td * 0.5,
#         0.5 * (width + h_td), 0.5 * (width - h_td), 1 - z_td * 0.5, 1 + z_td * 0.5],
#        [x_td2_sr - l_td * 0.5, x_td2_sr + l_td * 0.5,
#         0.5 * (width + h_td), 0.5 * (width - h_td), 1 - z_td * 0.5, 1 + z_td * 0.5]
#        ]



def radius(st, ed):
    return ((st[0] - ed[0]) ** 2 + (st[1] - ed[1]) ** 2 + (st[2] - ed[2]) ** 2) ** 0.5

minus = []  # TODO формируем точки кузова


def shina(shinas, v1arr, v2arr, v3, I, U, type_='FRONT', ver_='PER'):
    dc = 10

    sh_p = []
    for sh in shinas:
        if sh[1][0]:
            arr = np.linspace(sh[0][0], sh[0][0]+sh[1][0], dc)
            sh_p.extend([(x, sh[0][1], sh[0][2]) for x in arr])
            print(sh_p)
        elif sh[1][1]:
            arr = np.linspace(sh[0][1], sh[0][1]+sh[1][1], dc)
            sh_p.extend([(sh[0][0], y, sh[0][2]) for y in arr])
        elif sh[1][2]:
            arr = np.linspace(sh[0][2], sh[0][2]+sh[1][2], dc)
            sh_p.extend([(sh[0][0], sh[0][1], z) for z in arr])

    sh_points = [(length+pp[0], -0.5*width+pp[1], floor+pp[2]) for pp in sh_p]
    print(sh_points)

    def in_point(x_, y_, z_):
        r = 0
        for point in sh_points:
            r += 1 / radius((x_, y_, z_), point)

        if ver_ == 'PER':
            return {f: [I * harm[f][0] * r / (2 * pi * len(sh_points)), U * harm[f][0] * r / len(sh_points)]
                    for f in harm.keys()}
        else:
            return [I * r / (2 * pi * len(sh_points)), U * r / len(sh_points)]

    if type_ == 'FRONT':
        res = [in_point(v3, y, z) for z in v2arr for y in v1arr]
    else:
        res = [in_point(x, y, v3) for y in v2arr for x in v1arr]

    return res

# имеется ТЭД, СУ, ВУ, ГК, и проч.
# посчитать электричество а потом туда мгнетизм

# в одном пуле - несколько

def oborud(element, v1arr, v2arr, v3, I, U, n, type_='FRONT', ver_='PER'):
    ds = 8

    if ob == 'com':
        nodes_x = []
    else:  # если это ТЭД
        # todo ещё ТЭДы
        nodes_x = [x_td1_sr + 0.5 * r_td * np.cos(ap) for ap in np.linspace(0, 2 * pi, ds)]
        nodes_z = [z_td + 0.5 * r_td * np.sin(ap) for ap in np.linspace(0, 2 * pi, ds)]
        nodes_y = [td - td_p for td in [dy_td, -dy_td] for td_p in np.linspace(-0.5 * l_td, 0.5 * l_td, 4)]
        points = [[x_, y_, z_] for z_ in nodes_z for y_ in nodes_y for x_ in nodes_x]

    # разбиваем кабину на узлы
    x_cab = np.linspace(length, all_length, 40)
    y_cab = np.linspace(-0.5 * width, 0.5 * width, 40)
    minus = [[x_, y_] for y_ in y_cab for x_ in x_cab]

    # if type_ == 'FRONT':
    #     x_cab = np.linspace(length, all_length, 40)
    #     y_cab = np.linspace(-0.5 * width, 0.5 * width, 40)
    #     minus = [[x_, y_] for y_ in y_cab for x_ in x_cab]
    # else:
    #     minus = [[x_, y_] for y_ in v2arr for x_ in v1arr]



    # TODO это постоянное а надо ещё и переменное
    def in_point(x_, y_, z_):
        H_ob, E_ob = 0, 0
        for p in points:
            r = ((p[0] - x_) ** 2 + (p[1] - y_) ** 2 + (p[2] - z_) ** 2) ** 0.5
            H_ob += I / (pi * l_ob) * atan(l_ob / 2 / r)
            E_ob += U / r / len(points)

        for m in minus:
            r_m = ((m[0] - x_) ** 2 + (m[1] - y_) ** 2 + (floor - z_) ** 2) ** 0.5
            if r_m != 0:
                E_ob += U_g / r_m / len(minus)
        return [[H_ob * n / len(points), E_ob], (x_, y_, z_)]

    if type_ == 'UP':
        return [in_point(x_, y_, z_graph) for y_ in y_arr for x_ in x_arr]
    else:
        return [in_point(x_chel, y_, z_) for z_ in y_arr for y_ in x_arr]





    # TODO чтобы ещё и ТД правильно считал
    obor_points = []  # TODO список точек xyz с указанием силы тока и напряжения

    def in_point(x_, y_, z_):
        if ver_ == 'PER':
            H_res = {f: x_*y_*z_*harm[f][0] for f in harm.keys()}
            # H_res = {f: 0 for f in harm.keys()}
            E_res = {f: x_*y_*z_*harm[f][1] for f in harm.keys()}
            # E_res = {f: 0 for f in harm.keys()}
            # for p in obor_points:
            #     r = radius([x_, y_, z_], p[:3])
            #     for f in harm.keys():
            #         H_res[f] += p[4] * harm[f][0] * ...  # TODO
            #         E_res[f] += p[5] * harm[f][1] / r /
            # for m in minus:
            #     r_m = radius([x_, y_, z_], m)
            #     for f in harm.keys():
            #         E_res += p[5] * harm[f][1]/ r_m /
        else:
            H_res, E_res = x_*y_*z_, x_*y_*z_
            # H_res, E_res = 0, 0
            # for p in obor_points:
            #     r = radius([x_, y_, z_], p[:3])
            #     H_res += p[4] * ...  # TODO
            #     E_res += p[5] / r
            # for m in minus:
            #     r_m = radius([x_, y_, z_], m)
            #     E_res += p[5] / r_m

    if type_ == 'FRONT':
        res = [in_point(v3, y, z) for z in v2arr for y in v1arr]
    else:
        res = [in_point(x, y, v3) for y in v2arr for x in v1arr]

    return res


def field_sum(arg):
    def summ(f, i):
        sum_h, sum_e = 0, 0
        for el in arg:
            sum_h += el[i][f][0]
            sum_e += el[i][f][1]
        return [sum_h, sum_e]

    return [{frq: summ(frq, i) for frq in harm.keys()} for i in range(0, len(arg[0]))]


def full_energy(en):
    sum_h, sum_e = 0, 0
    for en in en.values():
        sum_h += en[0]
        sum_e += en[1]
    return [sum_h, sum_e]


def do_draw(h_lines, v_lines, c, type_):
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


def lines_oborud(oborud_, color, type_='FRONT'):
    h_lines = []
    v_lines = []
    if type_ == 'FRONT':
        for ob in oborud_:
            h_lines.append([ob[0][4], ob[0][2], ob[0][3]])
            h_lines.append([ob[0][5], ob[0][2], ob[0][3]])
            v_lines.append([ob[0][2], ob[0][4], ob[0][5]])
            v_lines.append([ob[0][3], ob[0][4], ob[0][5]])
    else:
        for ob in oborud_:
            h_lines.append([ob[0][2], ob[0][0], ob[0][1]])
            h_lines.append([ob[0][3], ob[0][0], ob[0][1]])
            v_lines.append([ob[0][0], ob[0][2], ob[0][3]])
            v_lines.append([ob[0][1], ob[0][2], ob[0][3]])

    do_draw(h_lines, v_lines, color, type_)


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




# TODO достаём внешнее поле из первого модуля


def make_triang(x_arr, y_arr):
    nodes_x = [x_ for _ in y_arr for x_ in x_arr]
    nodes_y = [y_ for y_ in y_arr for _ in x_arr]
    elements = [[i + j * dis, i + 1 + j * dis, (j + 1) * dis + i + 1] for j in range(0, dis - 1) for i in
                range(0, dis - 1)]
    elements.extend(
        [[i + j * dis, (j + 1) * dis + i, (j + 1) * dis + i + 1] for j in range(0, dis - 1) for i in range(0, dis - 1)])
    return tri.Triangulation(nodes_x, nodes_y, elements)


def triang_draw(triangulation, scalar_, name_, x_lb='Ось x, метры', y_lb='Ось y, метры'):
    plt.axis('equal')
    plt.tricontourf(triangulation, scalar_, cmap='YlOrRd')
    plt.colorbar()
    tcf = plt.tricontour(triangulation, scalar_, alpha=0.75, colors='black', linestyles='dotted', levels=5)
    plt.clabel(tcf, fontsize=10)

    plt.xlabel(x_lb)
    plt.ylabel(y_lb)

    plt.title(name_)


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
        # TODO шины переменные только
        # lines_shina(sh_vu_cp, 'turquoise', type_='UP')
        # lines_shina(sh_cp_td, 'c', type_='UP')
        # lines_oborud(vu, 'darkblue', type_='UP')
        # lines_oborud(cp, 'magenta', type_='UP')

    # field = shina(sh_vu_cp, x_ln, y_ln, z_graph, I_vu_cp, U_vu_cp, type_='UP')
    # # TODO ещё одну
    #
    # summar = [full_energy(el) for el in field]
    # magnetic = [el[0] for el in summar]
    # electric = [el[1] for el in summar]
    # energy = [el[0]*el[1] for el in summar]

    plt.figure(1)
    plt.subplot(2, 1, 1)
    figure_draw(magnetic, 'Переменный магнетизм')
    plt.subplot(2, 1, 2)
    figure_draw(electric, 'Переменный электричество')

    # mng = plot.get_current_fig_manager()
    # mng.window.state('zoomed')
    name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_переменный.png"
    plt.savefig(name)

    # plt.figure(2)
    #    TODO переменный гаромоники ток
    # plt.figure(3)
    #    TODO переменный гаромоники электричество
    # plt.figure(4)
    #    TODO переменный гаромоники общий
    # plt.figure(5)
    #    TODO общая энергия без экрана, лист, сетка


def visual_up_post():
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
        lines_shina(sh_vu_cp, 'turquoise', type_='UP')
        lines_shina(sh_cp_td, 'c', type_='UP')
        lines_oborud(vu, 'darkblue', type_='UP')
        lines_oborud(cp, 'magenta', type_='UP')

    # field = shina(sh_test, x_ln, y_ln, z_graph, I_vu_cp, U_vu_cp, type_='UP', ver_='POST')

    # vu_cp = shina(sh_vu_cp, x_ln, y_ln, z_graph, I_vu_cp, U_vu_cp, type_='UP', ver_='POST')
    # cp_td = shina(sh_cp_td, x_ln, y_ln, z_graph, I_cp_td, U_cp_td, type_='UP', ver_='POST')

    vu_f = oborud(vu, x_ln,y_ln, z_graph, I_vu, U_vu, n_vu, type_='UP', ver_='POST')

    # field = field_sum([cp_td, vu_cp])


    summar = field_sum([])
    magnetic = [el[0] for el in summar]
    electric = [el[1] for el in summar]
    energy = [el[0]*el[1] for el in summar]


    # sh_f_post = shina(sh_post, x_arr, y_arr, z, type_='UP', ver_='POST')
    # ob_f_post = oborud(ob_post, x_ln, y_ln, z, type_='UP', ver_='POST')
    # magnetic = [sh_f_post[i][0]+ob_f_post[i][0] for i in range(0, len(x_arr)*len(y_arr)]
    # electric = [sh_f_post[i][1]+ob_f_post[i][1] for i in range(0, len(x_arr)*len(y_arr)]
    # post_all = [magnetic[i]*electric[i] for i in len(magnetic)]


    plt.figure(1)
    plt.subplot(2, 1, 1)
    figure_draw(magnetic, 'Магнетизм')
    plt.subplot(2, 1, 2)
    figure_draw(electric, 'Электричество')
    plt.suptitle('Постоянный, вид сверху')

    # mng = plot.get_current_fig_manager()
    # mng.window.state('zoomed')
    name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_постоянный.png"
    # plt.savefig(name)
    plt.show()

def visual_front():
    #  вид спереди 3-8:
    #    энергия постоянный
    #    энергия переменный
    #    энергия переменный гармоники
    print('График строится..................')

    Ymax = -0.5 * width
    Ymin = -Ymax
    Zmax = 0.1
    Zmin = height+floor

    y_ln = np.linspace(Ymin, Ymax, dis, endpoint=True)
    z_ln = np.linspace(Zmin, Zmax, dis, endpoint=True)

    tr = make_triang(y_ln, z_ln)

    i = 0  # потом другое число
    p_n = 2

    for no in SZ.keys():
        # TODO пепеменный
        G = [z * y * SZ[no] for z in z_ln for y in y_ln]
        # TODO пстоянный
        H = [z * y * log(SZ[no]) for z in z_ln for y in y_ln]
        # TODO это тоже пепемемный
        kab = [{fr: z * y * SZ[no] * harm[fr][0] for fr in harm.keys()} for z in z_ln for y in y_ln]

        plt.figure(i + p_n)
        i += 1
        name = f'Энергия. Вид спереди. Срез {SZ[no]} метров.'
        plt.subplot(1, 2, 1)
        triang_draw(tr, G, 'Переменный', y_lb='Ось z, метры')

        plt.subplot(1, 2, 2)
        triang_draw(tr, H, 'Постоянный', y_lb='Ось z, метры')
        lines_shina(sh_vu_cp, 'turquoise')
        lines_shina(sh_cp_td, 'blue')
        lines_oborud(vu, 'c')
        lines_oborud(cp, 'magenta')

        plt.suptitle(name)

        # mng = plot.get_current_fig_manager()
        # mng.window.state('zoomed')

        name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{no}_м.png"
        plt.savefig(name)

        plt.figure(i + p_n)
        i += 1
        name = 'Гармоники вид спереди'
        j = 0
        # chel_harm_e = []
        # TODO нужна ли какая-то гистограмма? Если нужна - то на какую точку?
        for fr in harm.keys():
            j += 1
            plt.subplot(3, 3, j)
            data = [dt[fr] for dt in kab]
            triang_draw(tr, data, '', y_lb=str(fr))
        # plt.subplot(3, 3, 9)
        # plt.bar(range(0, len(harm.keys())), chel_harm_e)
        plt.suptitle(name)

        name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_гарм_{no}_м.png"
        plt.savefig(name)


## РАСЧЁТ СТАТИСТИКИ ##

S = (a * b / 3600) ** 1 / 2
p = ti / 24  # статистическая вероятность воздействия


## ПОСТРОЕНИЕ ГРАФИКА ##

# формат
# номер среза: расстояние в метрах от стенки, разделяющей кабину и машинное отделение
# SZ = {3: 0.9,
#       4: 1.8,
#       5: 2.7,
#       6: 3.6,
#       7: 4.5,
#       8: 5.6}

SZ = {4: 1.8}

z_graph = z_vu

# TODO одновременно на одном СВЕРХУ и на одном СПЕРЕДИ смотрим:
#  1. шина
#  2. оборудование
#  после того как сработало на постоянном, проверяем переменные и гармоники

# visual_up_per()
visual_up_post()
# visual_front()

# Уже сделано:
# - рыбы графиков вид сверху
# - координаты постоянных шин и оборудования
# -- КРОМЕ ТЭД
# - рисование шин и оборудования с координат
# - расчёт постоянных шин
#  TODO проверить посчёт шин



# TODO переменные шины и оборудование

# TODO 2. набираем нужное количество графиков
# TODO 3. рыба на вывод поля: все массивы как надо, но значения - рыбо
# TODO 4. по очереди добавляем формулы и проверяем
# TODO      4.1 постоянное магнитное шины
# TODO      2. постоянное магнитное оборудование
# TODO      3. переменное магнитное всё
# TODO      4. постоянное электричесткое
# TODO      5. постоянное магнитное
# TODO 5. экран и статистика

plt.show()
