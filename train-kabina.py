from math import log, exp, pi, atan
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from datetime import datetime
import matplotlib.colors as colors
plt.style.use('seaborn-white')
cmap = 'YlOrRd'


# РЕЖИМ РАБОТЫ СЕТИ

I = 300  # cуммарная сила тока, А
U = 30000  # cуммарное напряжение, В

I_ted = 880  # сила тока в ТЭД, А
U_ted = 1950  # напряжение в ТЭД, В

# СТАТИСТИЧЕСКИЕ ДАННЫЕ
x_chel = 0.9  # положение человека по оси х
y_chel = 1  # положение человека по оси y
a = 1.75  # высота человека метры
b = 80  # масса человека килограммы
ti = 1  # длительность пребывания работника на рабочем месте, часов
z_graph = 2  # высота среза

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
bor = [0.27, 0.6, 1.2]  # узлы окна
z_chel = floor + 0.7  # где находится человек по оси z

metal_mu = 1000  # относительная магнитная проницаемость стали
glass_mu = 0.99  # относительная магнитная проницаемость стекла
metal_t = 0.0025  # толщина стали
glass_t = 0.015  # толщина стекла
metal_sigma = 10 ** 7  # удельная проводимость стали
glass_sigma = 10 ** -12  # удельная проводимость стекла


v_kab = length * width * height
metal_r = (v_kab * 3 / 4 / pi) ** 1 / 3
glass_r = ((2.86+0.8) * 3 / 4 / pi) ** 1 / 3

kh_glass = {frq: 10 * log(1 + (glass_sigma * 2 * pi * frq * glass_mu * glass_r * glass_t / 2) ** 2, 10)
                 for frq in harm.keys()}
ke_glass = 20 * log(60 * pi * glass_t * glass_sigma, 10)
# если числа очень маленькие, Питон считает их равными 0 и вызывает ошибку.
# поэтому, если эффект экранирования крайне низкий - считаем, что экрана вообще нет
for f in kh_glass.keys():
    if kh_glass[f] <= 0:
        kh_glass[f] = 1
if ke_glass <= 0:
    ke_glass = 1


kh_metal = {frq: 10 * log(1 + (metal_sigma * 2 * pi * frq * metal_mu * metal_r * metal_t / 2) ** 2, 10)
                 for frq in harm.keys()}
ke_metal = 20 * log(60 * pi * metal_t * metal_sigma, 10)


kh_post = 1 / (1 + (0.66 * metal_mu * metal_t / metal_r))
ke_post = ke_metal

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


def energy_pass(x_e, y_e, z_e):
    res_energy = {freq: [magnetic_calc(y_e, z_e, freq), electric_calc(y_e, z_e, freq)] for freq in harm.keys()}
    return [res_energy, (x_e, y_e, z_e)]


def full_energy(res_en):
    sum_h, sum_e = 0, 0
    for en in res_en.values():
        sum_h += en[0]
        sum_e += en[1]
    return [sum_h, sum_e]


def ekran(ext_en):
    k_h = {fr: 1 for fr in harm.keys()}
    k_e = 1
    if (ext_en[1][2] >= floor) and (ext_en[1][2] < floor+height):
        if abs(ext_en[1][1]) <= 0.5*width:
            if (ext_en[1][0] > bor[1]) and (ext_en[1][0] < bor[1]) and (abs(ext_en[1][1]) <= bor[2]):
                k_h = kh_glass
                k_e = ke_glass
            else:
                k_h = kh_metal
                k_e = ke_metal
    return [{fr: [ext_en[0][fr][0] / k_h[fr], ext_en[0][fr][1] / k_e] for fr in harm.keys()}, ext_en[1]]


def ekran_post(ext_en):
    k_h, k_e = 1, 1
    if (ext_en[1][2] > floor) and (ext_en[1][2] < floor+height):
        if abs(ext_en[1][1]) <= 0.5*width:
            k_h = kh_post
            k_e = ke_post
    return [[ext_en[0][0] / k_h, ext_en[0][1] / k_e], ext_en[1]]


def full_pass(f_el):
    sum_h, sum_e = 0, 0
    for fr in f_el[0].keys():
        sum_h += f_el[0][fr][0]
        sum_e += f_el[0][fr][1]
    return [sum_h, sum_e]


def visual_up():
    print('График строится..................')

    Xmin = -0.5
    Xmax = length + 0.5
    Ymax = -1 * 0.5 * width * 1.3
    Ymin = xp_up * 1.15

    x = np.linspace(Xmin, Xmax, dis, endpoint=True)
    y = np.linspace(Ymin, Ymax, dis, endpoint=True)

    every_f = [[energy_pass(x_, y_, z_graph) for x_ in x] for y_ in y]

    summar = [[full_energy(x_el[0]) for x_el in y_list] for y_list in every_f]

    magnetic = [[x_el[0] for x_el in y_list] for y_list in summar]
    electric = [[x_el[1] for x_el in y_list] for y_list in summar]
    eneegy = [[x_el[0]*x_el[1] for x_el in y_list] for y_list in summar]

    def do_graph(content, name_, x_lb='Ось x, метры', y_lb='Ось y, метры'):
        ct = plt.contour(x, y, content, alpha=0.75, colors='black', linestyles='dotted', levels=5)
        plt.clabel(ct, fontsize=10)
        plt.imshow(content, extent=[Xmin, Xmax, Ymax, Ymin], cmap='YlOrRd', alpha=0.95)

        for delta_y in [xp_kp, xp_up, xp_nt]:
            plt.hlines(delta_y, Xmin, Xmax, color='black', linewidth=2)
        plt.text(0.1, xp_kp+0.05, 'КП', color='white')
        plt.text(0.1, xp_up+0.05, 'УП', color='black')
        plt.text(1, xp_nt-0.3, 'НТ', color='white')

        plt.hlines(0.5 * width, 0, length, colors='red', linestyles='--')
        plt.hlines(-0.5 * width, 0, length, colors='red', linestyles='--')
        plt.vlines(0, -0.5 * width, 0.5 * width, colors='red', linestyles='--')
        plt.vlines(length, -0.5 * width, 0.5 * width, colors='red', linestyles='--')
        plt.xlabel(x_lb)
        plt.ylabel(y_lb)

        plt.title(name_)

    plt.figure(1)
    name = 'Контактная сеть вид сверху'
    plt.subplot(1, 3, 1)
    do_graph(magnetic, 'Магнитное', x_lb='Ось x, метры', y_lb='Ось y, метры')
    plt.subplot(1, 3, 2)
    do_graph(electric, 'Электрическое', x_lb='Ось x, метры', y_lb='Ось y, метры')
    plt.subplot(1, 3, 3)
    do_graph(eneegy, 'Энергия', x_lb='Ось x, метры', y_lb='Ось y, метры')

    plt.suptitle(name)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    plt.savefig(f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{name}.png")

    print('График построен.')

    return every_f


def fr_kab_lines():
    plt.hlines(height + floor, -0.5 * width, 0.5 * width, colors='red', linestyles='--')
    plt.hlines(floor, -0.5 * width, 0.5 * width, colors='red', linestyles='--')
    plt.hlines(1, -0.5 * width, 0.5 * width, colors='red', linestyles='--')
    plt.vlines(-0.5 * width, 1, height+floor, colors='red', linestyles='--')
    plt.vlines(0.5 * width, 1, height+floor, colors='red', linestyles='--')


def visual_front():
    print('График строится..................')

    Ymin = -1 * max(xp, width) * 1.15
    Ymax = xp_up * 1.2
    Zmax = 0.1
    Zmin = max(h_kp, h_nt, h_up) * 1.1

    y = np.linspace(Ymin, Ymax, dis, endpoint=True)
    z = np.linspace(Zmin, Zmax, dis, endpoint=True)

    every_f = [[energy_pass(x_chel, y_, z_) for y_ in y] for z_ in z]
    all_field = [[full_energy(x_el[0]) for x_el in y_list] for y_list in every_f]
    summar = [[x_el[0] * x_el[1] for x_el in y_list] for y_list in all_field]

    plt.figure(2)
    b = 10 ** (len(str(round(np.amin(summar)))) - 1)  # для правильного отображения линий
    ct = plt.contour(y, z, summar, alpha=0.75, colors='black', linestyles='dotted',
                     levels=[b, 2*b, 5*b, 7*b, 10*b, 20*b, 50*b, 100*b])
    plt.clabel(ct, fontsize=10)
    plt.imshow(summar, extent=[Ymin, Ymax, Zmax, Zmin], cmap=cmap, alpha=0.95, norm=colors.LogNorm())
    plt.colorbar()

    plt.text(xp_kp, h_kp, 'КП', color='white',  fontsize=14)
    plt.text(xp_up, h_up, 'УП', color='white', fontsize=14)
    plt.text(xp_nt, h_nt, 'НТ', color='white', fontsize=14)

    fr_kab_lines()

    plt.xlabel('Ось y, метры')
    plt.ylabel('Ось z, метры')

    plt.title('Вид сбоку')

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_вид сбоку.png"
    plt.savefig(name)

    print('График построен.')
    return every_f


def ted_field_calc(x_arr, y_arr, I_g, U_g, n, type_='UP'):
    ds = 8

    # разбиваем ТЭД на узлы
    nodes_x = [x_td1_sr + 0.5*r_td * np.cos(ap) for ap in np.linspace(0, 2*pi, ds)]
    nodes_z = [z_td + 0.5*r_td * np.sin(ap) for ap in np.linspace(0, 2*pi, ds)]
    nodes_y = [td-td_p for td in [dy_td, -dy_td] for td_p in np.linspace(-0.5*l_td, 0.5*l_td, 4)]

    points = [[x_, y_, z_] for z_ in nodes_z for y_ in nodes_y for x_ in nodes_x]

    # разбиваем кабину на узлы
    if type_ == 'UP':
        minus = [[x_, y_] for y_ in y_arr for x_ in x_arr]
    else:
        x_cab = np.linspace(0, length, 40)
        y_cab = np.linspace(-0.5*width, 0.5*width, 40)
        minus = [[x_, y_] for y_ in y_cab for x_ in x_cab]

    def in_point(x_, y_, z_):
        H_ob, E_ob = 0, 0
        for p in points:
            r = ((p[0]-x_)**2 + (p[1]-y_)**2 + (p[2]-z_)**2) ** 0.5
            H_ob += I_g / (pi * l_td) * atan(l_td / 2 / r)
            E_ob += U_g / r / len(points)

        for m in minus:
            r_m = ((m[0] - x_) ** 2 + (m[1] - y_) ** 2 + (floor - z_) ** 2) ** 0.5
            if r_m != 0:
                E_ob += U_g / r_m / len(minus)
        return [[H_ob * n / len(points), E_ob], (x_, y_, z_)]

    if type_ == 'UP':
        return [in_point(x_, y_, z_graph) for y_ in y_arr for x_ in x_arr]
    else:
        return [in_point(x_chel, y_, z_) for z_ in y_arr for y_ in x_arr]


def kab_lines_up():
    d = 0.12
    plt.hlines(y_chel-d, x_chel-d, x_chel+d, colors='white', linestyles='--')
    plt.hlines(y_chel+d, x_chel-d, x_chel+d, colors='white', linestyles='--')
    plt.hlines(-y_chel-d, x_chel-d, x_chel+d, colors='white', linestyles='--')
    plt.hlines(-y_chel+d, x_chel-d, x_chel+d, colors='white', linestyles='--')
    plt.vlines(x_chel-d, y_chel-d, y_chel+d, colors='white', linestyles='--')
    plt.vlines(x_chel+d, y_chel-d, y_chel+d, colors='white', linestyles='--')
    plt.vlines(x_chel-d, -y_chel-d, -y_chel+d, colors='white', linestyles='--')
    plt.vlines(x_chel+d, -y_chel-d, -y_chel+d, colors='white', linestyles='--')

    plt.hlines(y_chel-d, x_chel+d+0.05, x_chel+d+0.10, colors='white', linestyles='--')
    plt.hlines(y_chel+d, x_chel+d+0.05, x_chel+d+0.10, colors='white', linestyles='--')
    plt.hlines(-y_chel-d, x_chel+d+0.05, x_chel+d+0.10, colors='white', linestyles='--')
    plt.hlines(-y_chel+d, x_chel+d+0.05, x_chel+d+0.10, colors='white', linestyles='--')
    plt.vlines(x_chel+d+0.05, y_chel-d, y_chel+d, colors='white', linestyles='--')
    plt.vlines(x_chel+d+0.10, y_chel-d, y_chel+d, colors='white', linestyles='--')
    plt.vlines(x_chel+d+0.05, -y_chel-d, -y_chel+d, colors='white', linestyles='--')
    plt.vlines(x_chel+d+0.10, -y_chel-d, -y_chel+d, colors='white', linestyles='--')

    plt.vlines(bor[0], bor[2], -bor[2], colors='white', linestyles='--')
    plt.vlines(bor[1], bor[2], -bor[2], colors='white', linestyles='--')
    plt.hlines(bor[2], bor[0], bor[1], colors='white', linestyles='--')
    plt.hlines(-bor[2], bor[0], bor[1], colors='white', linestyles='--')

    plt.plot(np.array([0.01, bor[0]]), np.array([0, bor[2]]), c='white', linestyle='--')
    plt.plot(np.array([0.01, bor[0]]), np.array([0, -bor[2]]), c='white', linestyle='--')

    plt.hlines(0.5*width-0.01, 0, length, colors='white', linestyles='--')
    plt.hlines(-0.5*width+0.01, 0, length, colors='white', linestyles='--')
    plt.vlines(0.01, 0.5*width, -0.5*width, colors='white', linestyles='--')
    plt.vlines(length-0.01, 0.5*width, -0.5*width, colors='white', linestyles='--')


def kab_lines_front():
    d = 0.13
    plt.hlines(z_chel, y_chel-d, y_chel+d, colors='white', linestyles='--')
    plt.hlines(z_chel, -y_chel-d, -y_chel+d, colors='white', linestyles='--')
    plt.hlines(z_chel-0.05, y_chel-d, y_chel+d, colors='white', linestyles='--')
    plt.hlines(z_chel-0.05, -y_chel-d, -y_chel+d, colors='white', linestyles='--')

    plt.vlines(y_chel-d, z_chel, z_chel-0.05, colors='white', linestyles='--')
    plt.vlines(y_chel+d, z_chel, z_chel-0.05, colors='white', linestyles='--')
    plt.vlines(-y_chel-d, z_chel, z_chel-0.05, colors='white', linestyles='--')
    plt.vlines(-y_chel+d, z_chel, z_chel-0.05, colors='white', linestyles='--')

    d = 0.12
    plt.hlines(z_chel+0.05, y_chel-d, y_chel+d, colors='white', linestyles='--')
    plt.hlines(z_chel+0.05+2*d, y_chel-d, y_chel+d, colors='white', linestyles='--')
    plt.hlines(z_chel+0.05, -y_chel-d, -y_chel+d, colors='white', linestyles='--')
    plt.hlines(z_chel+0.05+2*d, -y_chel-d, -y_chel+d, colors='white', linestyles='--')

    plt.vlines(y_chel-d, z_chel+0.05, z_chel+0.05+2*d, colors='white', linestyles='--')
    plt.vlines(y_chel+d, z_chel+0.05, z_chel+0.05+2*d, colors='white', linestyles='--')
    plt.vlines(-y_chel-d, z_chel+0.05, z_chel+0.05+2*d, colors='white', linestyles='--')
    plt.vlines(-y_chel+d, z_chel+0.05, z_chel+0.05+2*d, colors='white', linestyles='--')

    plt.hlines(floor+0.01, 0.5*width, -0.5*width, colors='white', linestyles='--')
    plt.hlines(height+floor-0.01, 0.5*width, -0.5*width, colors='white', linestyles='--')
    plt.vlines(0.5*width-0.01, height, height+floor, colors='white', linestyles='--')
    plt.vlines(-0.5*width+0.01, height, height+floor, colors='white', linestyles='--')


def ted_lines():
    pass
    # x_td, y_td = r_td, l_td
    # plt.hlines(dy_td - 0.5 * y_td, x_td1_sr - x_td * 0.5, x_td1_sr + x_td * 0.5, colors='blue', linestyles='--')
    # plt.hlines(dy_td + 0.5 * y_td, x_td1_sr - x_td * 0.5, x_td1_sr + x_td * 0.5, colors='blue', linestyles='--')
    # plt.vlines(x_td1_sr - x_td * 0.5, dy_td - 0.5 * y_td, dy_td + 0.5 * y_td, colors='blue', linestyles='--')
    # plt.vlines(x_td1_sr + x_td * 0.5, dy_td - 0.5 * y_td, dy_td + 0.5 * y_td, colors='blue', linestyles='--')
    # plt.hlines(-dy_td - 0.5 * y_td, x_td1_sr - x_td * 0.5, x_td1_sr + x_td * 0.5, colors='blue',
    #            linestyles='--')
    # plt.hlines(-dy_td + 0.5 * y_td, x_td1_sr - x_td * 0.5, x_td1_sr + x_td * 0.5, colors='blue',
    #            linestyles='--')
    # plt.vlines(x_td1_sr - x_td * 0.5, -dy_td - 0.5 * y_td, -dy_td + 0.5 * y_td, colors='blue', linestyles='--')
    # plt.vlines(x_td1_sr + x_td * 0.5, -dy_td - 0.5 * y_td, -dy_td + 0.5 * y_td, colors='blue', linestyles='--')


def ted_lines_front():
    plt.hlines(z_td + 0.5*r_td, dy_td - 0.5*l_td, dy_td + 0.5*l_td, colors='blue', linestyles='--')
    plt.hlines(z_td - 0.5*r_td, dy_td - 0.5*l_td, dy_td + 0.5*l_td, colors='blue', linestyles='--')
    plt.hlines(z_td + 0.5*r_td, -dy_td - 0.5*l_td, -dy_td + 0.5*l_td, colors='blue', linestyles='--')
    plt.hlines(z_td - 0.5*r_td, -dy_td - 0.5*l_td, -dy_td + 0.5*l_td, colors='blue', linestyles='--')

    plt.vlines(dy_td - 0.5*l_td, z_td - 0.5*r_td, z_td + 0.5*r_td, colors='blue', linestyles='--')
    plt.vlines(dy_td + 0.5*l_td, z_td - 0.5*r_td, z_td + 0.5*r_td, colors='blue', linestyles='--')
    plt.vlines(-dy_td - 0.5*l_td, z_td - 0.5*r_td, z_td + 0.5*r_td, colors='blue', linestyles='--')
    plt.vlines(-dy_td + 0.5*l_td, z_td - 0.5*r_td, z_td + 0.5*r_td, colors='blue', linestyles='--')


def triang_do(triangulation, scalar_, name_, x_lb='Ось x, метры', y_lb='Ось y, метры', lev=5):
    plt.axis('equal')
    plt.tricontourf(triangulation, scalar_, cmap=cmap)
    tcf = plt.tricontour(triangulation, scalar_, alpha=0.75, colors='black', linestyles='dotted', levels=lev)
    plt.clabel(tcf, fontsize=10)
    plt.colorbar()

    plt.xlabel(x_lb)
    plt.ylabel(y_lb)

    plt.title(name_)


def visual_up_locomotive(ext_f):
    print('График строится..................')

    Xmin = 0
    Xmax = length
    Ymax = -0.5 * width
    Ymin = -Ymax

    inside = [[full_pass(ekran(x_el)) for x_el in y_list if (x_el[1][0] >= Xmin) and (x_el[1][0] <= Xmax)]
              for y_list in ext_f if abs(y_list[0][1][1]) <= 0.5 * width]

    x_ln = np.linspace(Xmin, Xmax, len(inside[0]), endpoint=True)
    y_ln = np.linspace(Ymin, Ymax, len(inside), endpoint=True)

    def graph_do(znach, name_, x_lb='', y_lb=''):
        ct = plt.contour(x_ln, y_ln, znach, alpha=0.95, colors='white', linestyles='dotted', levels=5)
        plt.clabel(ct, fontsize=10)
        plt.imshow(znach, extent=[Xmin, Xmax, Ymax, Ymin], cmap='YlOrRd', alpha=0.95, norm=colors.LogNorm())
        plt.colorbar()

        plt.xlabel(x_lb)
        plt.ylabel(y_lb)

        plt.title(name_)

    magnetic = [[x_el[0] for x_el in y_list] for y_list in inside]
    electric = [[x_el[1] for x_el in y_list] for y_list in inside]
    summar = [[x_el[0]*x_el[1] for x_el in y_list] for y_list in inside]


    plt.figure(3)
    name = 'Вид сверху кабина переменное экран'
    plt.subplot(1, 3, 1)
    kab_lines_up()
    graph_do(magnetic, 'Магнитное', x_lb='Ось x, метры', y_lb='Ось y, метры')
    plt.subplot(1, 3, 2)
    kab_lines_up()
    graph_do(electric, 'Электрическое', x_lb='Ось x, метры')
    plt.subplot(1, 3, 3)
    kab_lines_up()
    graph_do(summar, 'Общее', x_lb='Ось x, метры',)

    plt.suptitle(name)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.savefig(f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{name}.png")


def visual_up_post():
    print('Расчёт поля от тяговых двигателей....')
    Xmin = 0
    Xmax = length
    Ymax = -0.5 * width
    Ymin = -Ymax

    dis = 40
    x_ln = np.linspace(Xmin, Xmax, dis, endpoint=True)
    y_ln = np.linspace(Ymin, Ymax, dis, endpoint=True)

    nodes_x = [x_ for _ in y_ln for x_ in x_ln]
    nodes_y = [y_ for y_ in y_ln for _ in x_ln]
    elements = [[i + j * dis, i + 1 + j * dis, (j + 1) * dis + i + 1] for j in range(0, dis - 1) for i in
                range(0, dis - 1)]
    elements.extend(
        [[i + j * dis, (j + 1) * dis + i, (j + 1) * dis + i + 1] for j in range(0, dis - 1) for i in range(0, dis - 1)])
    tr = tri.Triangulation(nodes_x, nodes_y, elements)

    ted_field = ted_field_calc(x_ln, y_ln, I_ted, U_ted, 5)

    magnetic = [el[0][0]/kh_post for el in ted_field]
    electric = [el[0][1]/ke_post for el in ted_field]
    energy = [el[0][0]/kh_post * el[0][1]/ke_post for el in ted_field]

    plt.figure(4)
    name = 'Вид сверху кабина постоянное'
    plt.subplot(1, 3, 1)
    triang_do(tr, magnetic, 'Магнитное', x_lb='Ось x, метры', y_lb='Ось y, метры')
    ted_lines()
    kab_lines_up()
    plt.subplot(1, 3, 2)
    triang_do(tr, electric, 'Электрическое', x_lb='Ось x, метры')
    ted_lines()
    kab_lines_up()
    plt.subplot(1, 3, 3)
    triang_do(tr, energy, 'Общее', x_lb='Ось x, метры')
    ted_lines()
    kab_lines_up()

    plt.suptitle(name)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.savefig(f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{name}.png")

    print('График построен.')


def visual_front_locomotive(ext_f):
    Ymin, Ymax = -0.6*width, 0.6*width
    Zmin, Zmax = floor+height+1, 0.1

    ekran_ = [[ekran(y_el) for y_el in z_list if abs(y_el[1][1]) <= Ymax] for z_list in ext_f
               if z_list[0][1][2] < Zmin]
    summar = [[full_pass(x_el) for x_el in y_list] for y_list in ekran_]
    energy = [[x_el[0]*x_el[1] for x_el in y_list] for y_list in summar]


    plt.figure(5)
    name = 'Вид cпереди с экраном. Энергия.'
    plt.imshow(energy, extent=[Ymin, Ymax, Zmax, Zmin], cmap=cmap, alpha=0.95, norm=colors.LogNorm())
    plt.colorbar()
    fr_kab_lines()
    plt.title(name)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.savefig(f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{name}.png")

    Ymin, Ymax = -0.5*width, 0.5*width
    Zmin, Zmax = floor+height, floor
    kanina = [[y_el for y_el in z_list if abs(y_el[1][1]) <= Ymax] for z_list in ekran_
                if (z_list[0][1][2] > Zmax) and (z_list[0][1][2] < Zmin)]
    y_ln = np.linspace(Ymin, Ymax, len(kanina[0]), endpoint=True)
    z_ln = np.linspace(Zmin, Zmax, len(kanina), endpoint=True)
    chel_y = np.where(y_ln == max([y_ for y_ in y_ln if y_ <= y_chel]))[0][0]
    chel_z = np.where(z_ln == max([z_ for z_ in z_ln if z_ <= z_chel]))[0][0]

    def graph_do(znach, name_, x_lb='', y_lb=''):
        ct = plt.contour(y_ln, z_ln, znach, alpha=0.95, colors='black', linestyles='dotted', levels=5)
        plt.clabel(ct, fontsize=10)
        plt.imshow(znach, extent=[Ymin, Ymax, Zmax, Zmin],  cmap=cmap,  alpha=0.95, norm=colors.LogNorm())
        plt.colorbar()

        plt.xlabel(x_lb)
        plt.ylabel(y_lb)
        plt.title(name_)

    summar = [[full_pass(x_el) for x_el in y_list] for y_list in kanina]
    magnetic = [[x_el[0] for x_el in y_list] for y_list in summar]
    electric = [[x_el[1] for x_el in y_list] for y_list in summar]
    energy = [[x_el[0]*x_el[1] for x_el in y_list] for y_list in summar]

    plt.figure(6)
    name = 'Cпереди кабина переменное с экраном'
    plt.subplot(1, 3, 1)
    graph_do(magnetic, 'Магнитное', x_lb='Ось x, метры', y_lb='Ось y, метры')
    kab_lines_front()
    plt.subplot(1, 3, 2)
    graph_do(electric, 'Электрическое', x_lb='Ось x, метры')
    kab_lines_front()
    plt.subplot(1, 3, 3)
    graph_do(energy, 'Общее', x_lb='Ось x, метры',)
    kab_lines_front()
    plt.suptitle(name)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.savefig(f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{name}.png")

    plt.figure(7)
    name = 'Гармоники магнитное вид спереди'
    i = 0
    chel_harm_h = []
    for fr in harm.keys():
        i += 1
        plt.subplot(3, 3, i)
        data = [[el[0][fr][0] for el in lst]for lst in kanina]
        chel_harm_h.append(data[chel_z][chel_y])
        graph_do(data, str(fr))
    plt.subplot(3, 3, 9)
    plt.bar(range(0, len(harm.keys())), chel_harm_h)
    plt.suptitle(name)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.savefig(f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{name}.png")

    plt.figure(8)
    name = 'Гармоники электрическое вид спереди'
    i = 0
    chel_harm_e = []
    for fr in harm.keys():
        i += 1
        plt.subplot(3, 3, i)
        data = [[el[0][fr][1] for el in lst]for lst in kanina]
        chel_harm_e.append(data[chel_z][chel_y])
        graph_do(data, str(fr))
    plt.subplot(3, 3, 9)
    plt.bar(range(0, len(harm.keys())), chel_harm_e)
    plt.suptitle(name)

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.savefig(f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{name}.png")

    print('Гармоники магнитного поля для человека', chel_harm_h,
          'Гармоники электрического поля для человека', chel_harm_e,
          sep='\n')



def visual_front_post():

    print('Расчёт поля от тяговых двигателей')
    dis_y, dis_z = 60, 40
    y_ln = np.linspace(-0.6*width, 0.6*width, dis_y, endpoint=True)
    z_ln = np.linspace(floor+height+1, 0.1, dis_z, endpoint=True)

    nodes_y = [y_ for _ in z_ln for y_ in y_ln]
    nodes_z = [z_ for z_ in z_ln for _ in y_ln]
    elements = [[i + j * dis_y, i + 1 + j * dis_y, (j + 1) * dis_y + i + 1]
                for j in range(0, dis_z - 1) for i in range(0, dis_y - 1)]
    elements.extend(
        [[i + j * dis_y, (j + 1) * dis_y + i, (j + 1) * dis_y + i + 1]
         for j in range(0, dis_z - 1) for i in range(0, dis_y - 1)])
    tr = tri.Triangulation(nodes_y, nodes_z, elements)

    ted_field = ted_field_calc(y_ln, z_ln, I_ted, U_ted, 5, type_='FRONT')

    plt.figure(9)
    name = 'Вид спереди постоянное'
    all_f = [el[0][0] * el[0][1] for el in ted_field]
    plt.subplot(1, 2, 1)
    triang_do(tr, all_f, 'Магнитное', x_lb='Ось y, метры', y_lb='Ось z, метры')
    fr_kab_lines()
    plt.title('Без экрана')
    front_ekran = [ekran_post(el) for el in ted_field]

    plt.subplot(1, 2, 2)

    summar = [el[0][0] * el[0][1] for el in front_ekran]
    triang_do(tr, summar, 'Магнитное', x_lb='Ось y, метры', y_lb='Ось z, метры')
    fr_kab_lines()
    plt.title('С экраном')

    plt.suptitle(name)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.savefig(f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{name}.png")

    def graph_do(znach, name_, x_lb='', y_lb='', lev=5):
        ct = plt.contour(y_ln, z_ln, znach, alpha=0.95, colors='black', linestyles='dotted', levels=lev)
        plt.clabel(ct, fontsize=10)
        plt.imshow(znach, extent=[Ymin, Ymax, Zmax, Zmin],  cmap=cmap,  alpha=0.95, norm=colors.LogNorm())
        plt.colorbar()

        plt.xlabel(x_lb)
        plt.ylabel(y_lb)
        plt.title(name_)

    Ymin, Ymax = -0.5*width, 0.5*width
    Zmax, Zmin = floor, floor+height
    z_points = [el[1][2] for el in ted_field if (el[1][2] > Zmax) and (el[1][2] < Zmin)]
    z_kab = list(sorted(set(z_points), reverse=True))
    y_points = [el[1][1] for el in ted_field if abs(el[1][1]) < Ymax]
    y_kab = list(sorted(set(y_points)))

    kabina = [[el for el in ted_field if (el[1][2] == z_) and (abs(el[1][1]) < Ymax)] for z_ in z_kab]

    magnetic = [[el[0][0] for el in z_list] for z_list in kabina]
    electric = [[el[0][1] for el in z_list] for z_list in kabina]
    energy = [[el[0][0] * el[0][0] for el in z_list] for z_list in kabina]

    y_ln, z_ln = y_kab, z_kab

    plt.figure(10)
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

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.savefig(f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{name}.png")

    print('График построен.')


## РАСЧЁТ СТАТИСТИКИ ##

S = (a * b / 3600) ** 1 / 2
p = ti / 24  # статистическая вероятность воздействия


print('\nПараметры сети')
print(f'Высота КП: {h_kp} м')
print(f'Высота НЧ: {h_nt} м')
print(f'Высота УП: {h_up} м')
print(f'Напряжение: {U} Вольт')
print(f'Суммарный ток: {I} Ампер')
print(f'Напряжение ТЭД: {U_ted} Вольт')
print(f'Ток ТЭД: {I_ted} Ампер')
print(f'Высота среза: {z_graph} метров')


## ПОСТРОЕНИЕ ГРАФИКА ##

print('\nБез электровоза')
cont_f_up = visual_up()

print('\nВид спереди')
cont_f_front = visual_front()

print('\nПоле в кабине сверху')
visual_up_locomotive(cont_f_up)
visual_up_post()

print('\nПоле в кабине спереди')
visual_front_locomotive(cont_f_front)
visual_front_post()

chel_f_per = energy_pass(x_chel, y_chel, floor+0.7)
no_ekran_per = full_pass(chel_f_per)[0]*full_pass(chel_f_per)[1]
print('\nПеременное поле без экрана: %.4f' % no_ekran_per)


ekran_per_f = full_pass(ekran(chel_f_per))
ekran_per = ekran_per_f[0]*ekran_per_f[1]
print('\nПерменное поле с экраном %.4f' % ekran_per)
Dco = ekran_per * ti * S * p
Dpo = Dco / b
print('Удельная суточная доза поглощённой энергии: %.4f' % Dpo)

chel_f_post = ted_field_calc([y_chel], [z_chel], I_ted, U_ted, 5, type_='FRONT')[0][0]
ekran_post_ = chel_f_post[0] / kh_post * chel_f_post[1] / ke_post
print('\nПостоянное поле с экраном %.4f' % ekran_post_)
Dco = ekran_post_ * ti * S * p
Dpo = Dco / b
print('Удельная суточная доза поглощённой энергии: %.4f' % Dpo)

plt.show()
