from math import log, exp, pi, atan
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from datetime import datetime
import matplotlib.colors as colors
plt.style.use('seaborn-white')


# РЕЖИМ РАБОТЫ СЕТИ

I = 300  # cуммарная сила тока, А
U = 30000  # cуммарное напряжение, В

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

# ДАННЫЕ О КОНТАКТНОЙ СЕТИ

xp = 0.760  # m - половина расстояния между рельсами
xp_kp = 0  # m - расстояние от центра между рельсами до КП (если левее центра - поставить минус)
xp_nt = 0  # m - расстояние от центра между рельсами до НТ (если левее центра - поставить минус)
xp_up = 3.7  # m - расстояние от центра между рельсами до УП
d_kp = 12.81 / 1000  # mm
d_nt = 12.5 / 1000  # mm
d_up = 17.5 / 1000  # mm
h_kp = 6.0  # КП
h_nt = 7.8  # НТ
h_up = 8.0  # УП

# ДАННЫЕ О ЛОКОМОТИВЕ

length = 1.3  # длина кабины
all_length = 15.2  # длина всего локомотива
width = 2.8  # ширина кабины
height = 2.6  # высота кабины
floor = 2  # расстояние от земли до дна кабины
chel = floor + 0.7  # где находится человек

metal_mu = 1000  # относительная магнитная проницаемость стали
glass_mu = 0.99  # относительная магнитная проницаемость стекла
metal_t = 0.0025  # толщина стали
glass_t = 0.015  # толщина стекла
metal_sigma = 10 ** 7  # удельная проводимость стали
glass_sigma = 10 ** -12  # удельная проводимость стекла


v_kab = length * width * height
metal_r = (v_kab * 3 / 4 / pi) ** 1 / 3
glass_r = (2.86 * 3 / 4 / pi) ** 1 / 3
kh_glass = {frq: 10 * log(1 + (glass_sigma * 2 * pi * frq * glass_mu * glass_r * glass_t / 2) ** 2, 10)
                 for frq in harm.keys()}
kh_metal = {frq: 10 * log(1 + (metal_sigma * 2 * pi * frq * metal_mu * metal_r * metal_t / 2) ** 2, 10)
                 for frq in harm.keys()}
ke_glass = 20 * log(60 * pi * glass_t * glass_sigma, 10)
ke_metal = 20 * log(60 * pi * metal_t * metal_sigma, 10)

kh_post = 1 / (1 + (0.66 * metal_mu * metal_t / metal_r))
ke_post =ke_metal

# ОБОРУДОВАНИЕ

x_td1_sr = 0.9  # тяговый двигатель
dy_td = 0.8
r_td = 0.604
l_td = 0.66
z_td = 1


def mix(h_x, h_zz):
    return (h_x ** 2 + h_zz ** 2) ** 0.5


def magnetic_calc(x_m, z_m, f_m):

    I_h = I * harm.get(f_m)[0]

    Ikp = 0.41 * I_h
    Int = 0.20 * I_h
    Iup = 0.39 * I_h

    x = x_m - xp_kp
    h1xkp = Ikp / (4 * pi) * (
                -z_m / ((x + xp) ** 2 + z_m**2) + (z_m - h_kp)/(x ** 2 + (h_kp - z_m)**2))
    h1zkp = Ikp / (4 * pi) * (x + xp) * (
                1 / ((x + xp) ** 2 + z_m ** 2) - 1/(x ** 2 + (h_kp - z_m) ** 2))
    h1kp = mix(h1xkp, h1zkp)
    x = x_m - 2*xp - xp_kp
    h2xkp = Ikp / (4 * pi) * (
                -z_m / ((x + xp) ** 2 + z_m ** 2) + (z_m - h_kp) / ((x + 2*xp) ** 2 + (h_kp - z_m) ** 2))
    h2zkp = Ikp / (4 * pi) * (x + 2 * xp) * (
                1 / ((x + xp) ** 2 + z_m ** 2) - 1 / ((x + 2*xp) ** 2 + (h_kp - z_m) ** 2))
    h2kp = mix(h2xkp, h2zkp)
    hkp = h1kp + h2kp

    x = x_m - xp_nt
    h1xnt = Ikp / (4 * pi) * (
            -z_m / ((x + xp) ** 2 + z_m ** 2) + (z_m - h_nt) / (x ** 2 + (h_nt - z_m) ** 2))
    h1znt = Ikp / (4 * pi) * (x + xp) * (
            1 / ((x + xp) ** 2 + z_m ** 2) - 1 / (x ** 2 + (h_nt - z_m) ** 2))
    h1nt = mix(h1xnt, h1znt)
    x = x_m - 2 * xp - xp_nt
    h2xnt = Ikp / (4 * pi) * (
            -z_m / ((x + xp) ** 2 + z_m ** 2) + (z_m - h_nt) / ((x + 2 * xp) ** 2 + (h_nt - z_m) ** 2))
    h2znt = Ikp / (4 * pi) * (x + 2 * xp) * (
            1 / ((x + xp) ** 2 + z_m ** 2) - 1 / ((x + 2 * xp) ** 2 + (h_nt - z_m) ** 2))
    h2nt = mix(h2xnt, h2znt)
    hnt = h1nt + h2nt

    x = x_m - xp_up
    x2 = -xp + xp_up
    h1xup = Iup / (4 * pi) * (
            -z_m / ((x2 + 2 * xp + x) ** 2 + z_m ** 2) + (z_m - h_up) / (x ** 2 + (h_up - z_m) ** 2))
    h1zup = Int / (4 * pi) * (x2 + 2 * xp + x) * (
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

    return hkp + hnt + hup


def electric_calc(x_e, z_e, f_e):

    U_h = U * harm.get(f_e)[1]

    e_res = U_h * (
            log(1 + 4 * h_nt * z_e / ((x_e - xp_nt) ** 2 + (h_nt - z_e) ** 2)) / (2 * z_e * log(2 * h_nt / d_nt)) +
            log(1 + 4 * h_kp * z_e / ((x_e - xp_kp) ** 2 + (h_kp - z_e) ** 2)) / (2 * z_e * log(2 * h_kp / d_kp)) +
            log(1 + 4 * h_up * z_e / ((x_e - xp_up) ** 2 + (h_up - z_e) ** 2)) / (2 * z_e * log(2 * h_up / d_up))
    )
    return e_res


def energy_pass(x_e, z_e):
    res_energy = {freq: [magnetic_calc(x_e, z_e, freq), electric_calc(x_e, z_e, freq)] for freq in harm.keys()}
    return res_energy


def full_energy(res_en):
    sum_h, sum_e = 0, 0
    for en in res_en.values():
        sum_h += en[0]
        sum_e += en[1]
    return [sum_e, sum_h]


def full_pass(ext_en):
    sum_h, sum_e = 0, 0
    for fr in ext_en.keys():
        sum_h += ext_en[fr][0] / kh_metal[fr]
        sum_e += ext_en[fr][1] / ke_metal
    return [sum_e, sum_h]


def visual_up():
    print('График строится..................')

    Xmin = -0.5
    Xmax = length + 0.5
    Ymax = -1 * 0.5 * width * 1.3
    Ymin = xp_up * 1.15

    x = np.linspace(Xmin, Xmax, dis, endpoint=True)
    y = np.linspace(Ymin, Ymax, dis, endpoint=True)

    every_f = [[energy_pass(y_, z) for _ in x] for y_ in y]

    all_field = [[full_energy(x_el) for x_el in y_list] for y_list in every_f]

    magnetic = [[x_el[0] for x_el in y_list] for y_list in all_field]
    electric = [[x_el[1] for x_el in y_list] for y_list in all_field]
    summar = [[x_el[0]*x_el[1] for x_el in y_list] for y_list in all_field]

    def do_graph(content, name_, x_lb='Ось x, метры', y_lb='Ось y, метры'):
        ct = plt.contour(x, y, content, alpha=0.75, colors='black', linestyles='dotted', levels=5)
        plt.clabel(ct, fontsize=10)
        plt.imshow(content, extent=[Xmin, Xmax, Ymax, Ymin], cmap='YlOrRd', alpha=0.95)
        plt.colorbar()

        for delta_y in [xp_kp, xp_up, xp_nt]:
            plt.hlines(delta_y, Xmin, Xmax, color='white', linewidth=2)
        plt.text(6, xp_kp+0.05, 'КП', color='white')
        plt.text(6.5, xp_up+0.05, 'УП', color='white')
        plt.text(5.5, xp_nt-0.3, 'НТ', color='white')

        plt.hlines(0.5 * width, 0, length, colors='red', linestyles='--')
        plt.hlines(-0.5 * width, 0, length, colors='red', linestyles='--')
        plt.vlines(0, -0.5 * width, 0.5 * width, colors='red', linestyles='--')
        plt.vlines(length, -0.5 * width, 0.5 * width, colors='red', linestyles='--')

        plt.xlabel(y_lb)
        plt.ylabel(x_lb)

        plt.title(name_)

        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')

        name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{name_}_U_{U}_В_I_{I}_В.png"
        plt.savefig(name)

    plt.figure(1)
    do_graph(magnetic, 'Магнитное от КС', x_lb='Ось x, метры', y_lb='Ось y, метры')
    plt.figure(2)
    do_graph(electric, 'Электрическое от КС', x_lb='Ось x, метры', y_lb='Ось y, метры')
    plt.figure(3)
    do_graph(summar, 'Контактная сеть общая энергия', x_lb='Ось x, метры', y_lb='Ось y, метры')

    print('График построен.')

    return every_f


def visual_front():
    print('График строится..................')

    Xmin = -1 * max(xp, width) * 1.15
    Xmax = xp_up * 1.2
    Zmax = 0.1
    Zmin = max(h_kp, h_nt, h_up) * 1.1

    x = np.linspace(Xmin, Xmax, dis, endpoint=True)
    z = np.linspace(Zmin, Zmax, dis, endpoint=True)

    every_f = [[energy_pass(x_, z_) for x_ in x] for z_ in z]
    all_field = [[full_energy(x_el) for x_el in y_list] for y_list in every_f]
    summar = [[x_el[0] * x_el[1] for x_el in y_list] for y_list in all_field]

    plt.figure(4)
    b = 10 ** (len(str(round(np.amin(summar)))) - 1)  # для правильного отображения линий
    ct = plt.contour(x, z, summar, alpha=0.75, colors='black', linestyles='dotted',
                     levels=[b, 2*b, 5*b, 7*b, 10*b, 20*b, 50*b, 100*b])
    plt.clabel(ct, fontsize=10)
    plt.imshow(summar, extent=[Xmin, Xmax, Zmax, Zmin], cmap='YlOrRd', alpha=0.95, norm=colors.LogNorm())
    plt.colorbar()

    plt.text(xp_kp, h_kp, 'КП', color='white',  fontsize=14)
    plt.text(xp_up, h_up, 'УП', color='white', fontsize=14)
    plt.text(xp_nt, h_nt, 'НТ', color='white', fontsize=14)

    plt.hlines(height + floor, -0.5 * width, 0.5 * width, colors='red', linestyles='--')
    plt.hlines(floor, -0.5 * width, 0.5 * width, colors='red', linestyles='--')
    plt.hlines(1, -0.5 * width, 0.5 * width, colors='red', linestyles='--')
    plt.vlines(-0.5 * width, 1, height+floor, colors='red', linestyles='--')
    plt.vlines(0.5 * width, 1, height+floor, colors='red', linestyles='--')

    plt.xlabel('Ось x, метры')
    plt.ylabel('Ось z, метры')

    plt.title('Вид сбоку')

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_вид сбоку_U_{U}_В_I_{I}_В.png"
    plt.savefig(name)

    print('График построен.')

def ted_field_calc(x_arr, y_arr, I_g, U_g, n):
    ds = 8

    # разбиваем ТЭД на узлы
    nodes_x = [x_td1_sr + 0.5*r_td * np.cos(ap) for ap in np.linspace(0, 2*pi, ds)]
    nodes_z = [z_td + 0.5*r_td * np.sin(ap) for ap in np.linspace(0, 2*pi, ds)]
    nodes_y = [td-td_p for td in [dy_td, -dy_td] for td_p in np.linspace(-0.5*l_td, 0.5*l_td, 4)]

    points = [[x_, y_, z_] for z_ in nodes_z for x_ in nodes_x for y_ in nodes_y]

    # разбиваем кабину на узлы
    nd_x = np.linspace(0, length, ds)
    nd_y = np.linspace(-width/2, width/2, ds)
    minus = [[x_, y_] for y_ in y_arr for x_ in x_arr]

    dz2 = (floor - z) ** 2

    def in_point(x_, y_):
        H_ob, E_ob = 0, 0
        for p in points:
            r = ((p[0]-x_)**2 + (p[1]-y_)**2 + (p[2]-z)**2) ** 0.5
            H_ob += I_g / (pi * l_td) * atan(l_td / 2 / r)
            E_ob += U_g / r / len(points)

        for m in minus:
            r_m = ((m[0] - x_) ** 2 + (m[1] - y_) ** 2 + dz2) ** 0.5
            if r_m != 0:
                E_ob += U_g / r_m / len(minus)
        return [H_ob * n / len(points), E_ob]

    return [in_point(x_, y_) for y_ in y_arr for x_ in x_arr]


def visual_up_locomotive(ext_f):
    print('График строится..................')

    Xmin = 0
    Xmax = length
    Ymax = -0.5 * width
    Ymin = -1 * Ymax

    dis = 100
    x_ln = np.linspace(Xmin, Xmax, dis, endpoint=True)
    y_ln = np.linspace(Ymin, Ymax, dis, endpoint=True)
    chel_x = np.where(x_ln == max([x_ for x_ in x_ln if x_ <= x_chel]))[0][0]
    chel_y = np.where(y_ln == max([y_ for y_ in y_ln if y_ <= y_chel]))[0][0]

    def graph_do(znach, name_, x_lb='', y_lb=''):
        ct = plt.contour(x_ln, y_ln, znach, alpha=0.75, colors='black', linestyles='dotted', levels=5)
        plt.clabel(ct, fontsize=10)
        plt.imshow(znach, extent=[Xmin, Xmax, Ymax, Ymin], cmap='YlOrRd', alpha=0.95, norm=colors.LogNorm())
        plt.colorbar()

        plt.xlabel(y_lb)
        plt.ylabel(x_lb)

        plt.title(name_)

        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')

        name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{name_}_вид_сверху_U_{U}_В_I_{I}_В.png"
        plt.savefig(name)

    ekran = [[full_pass(x_el) for x_el in y_list] for y_list in ext_f]

    magnetic = [[x_el[0] for x_el in y_list] for y_list in ekran]
    electric = [[x_el[1] for x_el in y_list] for y_list in ekran]
    summar = [[x_el[0]*x_el[1] for x_el in y_list] for y_list in ekran]
    chel_per = summar[chel_y][chel_x]

    plt.figure(5)
    graph_do(magnetic, 'Кабина магнитное переменное', x_lb='Ось x, метры', y_lb='Ось y, метры')
    plt.figure(6)
    i = 0
    chel_h_harm = []
    for fr in harm.keys():
        harm_h = [[x_el[fr][0] / kh_metal[fr] for x_el in y_list] for y_list in ext_f]
        chel_h_harm.append(harm_h[chel_y][chel_x])
        i += 1
        plt.subplot(3, 3, i)
        ct = plt.contour(x_ln, y_ln, harm_h, alpha=0.75, colors='black', linestyles='dotted', levels=3)
        plt.clabel(ct, fontsize=8)
        plt.imshow(harm_h, extent=[Xmin, Xmax, Ymax, Ymin],  cmap='YlOrRd', alpha=0.95)
        plt.title(str(fr))
    plt.subplot(3, 3, 9)
    print("Гармоники магнитного поля:\n", chel_h_harm)
    plt.bar(range(0, 8), chel_h_harm)

    plt.figure(7)
    graph_do(electric, 'Кабина электрическое переменное', x_lb='Ось x, метры', y_lb='Ось y, метры')
    plt.figure(8)
    i = 0
    chel_e_harm = []
    for fr in harm.keys():
        harm_e = [[x_el[fr][1] / ke_metal for x_el in y_list] for y_list in ext_f]
        chel_e_harm.append(harm_e[chel_y][chel_x])
        i += 1
        plt.subplot(3, 3, i)
        ct = plt.contour(x_ln, y_ln, harm_e, alpha=0.75, colors='black', linestyles='dotted', levels=3)
        plt.clabel(ct, fontsize=8)
        plt.imshow(harm_e, extent=[Xmin, Xmax, Ymax, Ymin], cmap='YlOrRd', alpha=0.95)
        plt.title(str(fr))
    plt.subplot(3, 3, 9)
    print("Гармоники электрического поля:\n", chel_h_harm)
    plt.bar(range(0, 8), chel_e_harm)

    plt.figure(9)
    graph_do(summar, 'Кабина общее переменное', x_lb='', y_lb='')

    def triang_do(scalar_, name_, x_lb='Ось x, метры', y_lb='Ось y, метры'):
        plt.axis('equal')
        plt.tricontourf(triangulation, scalar_, cmap='YlOrRd')
        plt.colorbar()
        tcf = plt.tricontour(triangulation, scalar_, alpha=0.75, colors='black', linestyles='dotted', levels=5)
        plt.clabel(tcf, fontsize=10)

        plt.hlines(dy_td - 0.5 * r_td, x_td1_sr - l_td * 0.5, x_td1_sr + l_td * 0.5, colors='blue', linestyles='--')
        plt.hlines(dy_td + 0.5 * r_td, x_td1_sr - l_td * 0.5, x_td1_sr + l_td * 0.5, colors='blue', linestyles='--')
        plt.vlines(x_td1_sr - l_td * 0.5, dy_td - 0.5 * r_td, dy_td + 0.5 * r_td, colors='blue', linestyles='--')
        plt.vlines(x_td1_sr + l_td * 0.5, dy_td - 0.5 * r_td, dy_td + 0.5 * r_td, colors='blue', linestyles='--')
        plt.hlines(-dy_td - 0.5 * r_td, x_td1_sr - l_td * 0.5, x_td1_sr + l_td * 0.5, colors='blue',
                   linestyles='--')
        plt.hlines(-dy_td + 0.5 * r_td, x_td1_sr - l_td * 0.5, x_td1_sr + l_td * 0.5, colors='blue',
                   linestyles='--')
        plt.vlines(x_td1_sr - l_td * 0.5, -dy_td - 0.5 * r_td, -dy_td + 0.5 * r_td, colors='blue', linestyles='--')
        plt.vlines(x_td1_sr + l_td * 0.5, -dy_td - 0.5 * r_td, -dy_td + 0.5 * r_td, colors='blue', linestyles='--')

        plt.xlabel(y_lb)
        plt.ylabel(x_lb)

        plt.title(name_)

        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')

        name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{name_}_U_{U}_В_I_{I}_В.png"
        plt.savefig(name)

    print('Расчёт поля от тяговых двигателей')
    dis = 40
    x_ln = np.linspace(Xmin, Xmax, dis, endpoint=True)
    y_ln = np.linspace(Ymin, Ymax, dis, endpoint=True)
    chel_x = np.where(x_ln == max([x_ for x_ in x_ln if x_ <= x_chel]))[0][0]
    chel_y = np.where(y_ln == max([y_ for y_ in y_ln if y_ <= y_chel]))[0][0]
    ted_field = ted_field_calc(x_ln, y_ln, 880, 1950, 5)

    magnetic = [el[0]/kh_post for el in ted_field]
    electric = [el[1]/ke_post for el in ted_field]
    summar = [el[0]/kh_post * el[1]/ke_post for el in ted_field]
    chel_post = summar[(dis-1)*chel_y + chel_x]

    nodes_x = [x_ for _ in y_ln for x_ in x_ln]
    nodes_y = [y_ for y_ in y_ln for _ in x_ln]
    elements = [[i + j * dis, i + 1 + j * dis, (j + 1) * dis + i + 1] for j in range(0, dis - 1) for i in
                range(0, dis - 1)]
    elements.extend(
        [[i + j * dis, (j + 1) * dis + i, (j + 1) * dis + i + 1] for j in range(0, dis - 1) for i in range(0, dis - 1)])
    triangulation = tri.Triangulation(nodes_x, nodes_y, elements)

    plt.figure(10)
    triang_do(magnetic, 'Кабина магнитное постоянное', x_lb='Ось x, метры', y_lb='Ось y, метры')
    plt.figure(11)
    triang_do(electric, 'Кабина электрическое постоянное', x_lb='Ось x, метры', y_lb='Ось y, метры')
    plt.figure(12)
    triang_do(summar, 'Кабина общее постоянное', x_lb='Ось x, метры', y_lb='Ось y, метры')

    print('График построен.')
    return (chel_per, chel_post)


## РАСЧЁТ СТАТИСТИКИ ##

S = (a * b / 3600) ** 1 / 2
p = ti / 24  # статистическая вероятность воздействия

z = 2
print('\nПараметры сети')
print(f'Высота КП: {h_kp} м')
print(f'Высота НЧ: {h_nt} м')
print(f'Высота УП: {h_up} м')
print(f'Напряжение: {U} Вольт')
print(f'Суммарный ток: {I} Ампер')
print(f'Высота среза: {z} метров')

## ПОСТРОЕНИЕ ГРАФИКА ##

print('\nБез электровоза')
cont_field = visual_up()

print('\nВид спереди')
visual_front()

print('\nПоле в кабин')
doza = visual_up_locomotive(cont_field)

print('\nВысота среза: %.2f' % chel)

chel_f = full_energy(energy_pass(x_chel, floor+0.7))
f_c = chel_f[0]*chel_f[1]
print('\nПеременное поле без экрана: %.4f' % f_c)

print('\nПерменное поле с экраном %.4f' % doza[0])
Dco = doza[0] * ti * S * p
Dpo = Dco / b
print('Удельная суточная доза поглощённой энергии: %.4f' % Dpo)

print('\nПостоянное поле с экраном %.4f' % doza[1])
Dco = doza[1] * ti * S * p
Dpo = Dco / b
print('Удельная суточная доза поглощённой энергии: %.4f' % Dpo)

plt.show()
