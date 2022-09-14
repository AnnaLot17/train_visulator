from math import log, exp, pi, atan
import matplotlib.pyplot as plt
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
        250: [0.1469, 0.0115],
        350: [0.0612, 0.0050],
        450: [0.0429, 0.0040],
        550: [0.0282, 0.0036],
        650: [0.0196, 0.0032],
        750: [0.0147, 0.0022]}

sum_harm_I = sum([v[0] for v in harm.values()])
sum_harm_U = sum([v[1] for v in harm.values()])

# ДАННЫЕ О КОНТАКТНОЙ СЕТИ

xp_r = 0.760  # m - половина расстояния между рельсами
xp_kp = 0  # m - расстояние от центра между рельсами до КП (если левее центра - поставить минус)
xp_nt = 0  # m - расстояние от центра между рельсами до НТ (если левее центра - поставить минус)
xp_up = 0.6 + 3.1  # m - расстояние от центра между рельсами до УП
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


v_kab = all_length * width * height
metal_r = (v_kab * 3 / 4 / pi) ** 1 / 3
glass_r = (2.86 * 3 / 4 / pi) ** 1 / 3
kh_glass = {str(frq): 10 * log(1 + (glass_sigma * 2 * pi * frq * glass_mu * glass_r * glass_t / 2) ** 2, 10)
                 for frq in harm.keys()}
kh_metal = {str(frq): 10 * log(1 + (metal_sigma * 2 * pi * frq * metal_mu * metal_r * metal_t / 2) ** 2, 10)
                 for frq in harm.keys()}
ke_glass = 20 * log(60 * pi * glass_t * glass_sigma, 10)
ke_metal = 20 * log(60 * pi * metal_t * metal_sigma, 10)


# ПОЛОЖЕНИЕ ШИН И ОБОРУДОВАНИЯ
# x - от стенки кабины
# y - от нижнежнего края

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
z_vu = 0.630

d_vu = 1.1

x_cp1 = x_vu1 + 1 - d_vu  # сглаживающий реактор
x_cp2 = x_vu2 - 1 + d_vu
y_cp = y_vu1 + 0.9
l_cp = 0.8
h_cp = 0.8
z_cp = 0.6

# x_pr = 5.370  # переходной реактор

x_td1_sr = x_cp1 - l_cp - 2.5  # тяговый двигатель
x_td2_sr = x_cp2 + l_cp + 2.5
kol_par = 1.5
h_td = 0.6
l_td = 0.66
z_td = l_td

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


def mix(h_x, h_zz):
    return (h_x ** 2 + h_zz ** 2) ** 0.5


def magnetic_calc(x_m, z_m, f_m):

    I_h = I * harm.get(f_m)[0]

    Ikp = 0.41 * I_h
    Int = 0.20 * I_h
    Iup = 0.39 * I_h

    x_rigth = z_m / ((xp_r + x_m) ** 2 + z_m ** 2)
    x_left = z_m / ((-xp_r + x_m) ** 2 + z_m ** 2)
    z_rigth = (xp_r + x_m) / ((xp_r + x_m) ** 2 + z_m ** 2)
    z_left = (-xp_r + x_m) / ((-xp_r + x_m) ** 2 + z_m ** 2)

    h_x_kp = Ikp / (4 * pi) * (x_rigth + x_left +
                               (-h_kp + z_m) / ((xp_kp - x_m) ** 2 + (h_kp - z_m) ** 2))
    h_z_kp = Ikp / (4 * pi) * (z_rigth + z_left +
                               (xp_kp + x_m) / ((xp_kp - x_m) ** 2 + (h_kp - z_m) ** 2))

    h_x_nt = Int / (4 * pi) * (x_rigth + x_left +
                               (-h_nt + z_m) / ((xp_nt - x_m) ** 2 + (h_nt - z_m) ** 2))
    h_z_nt = Int / (4 * pi) * (z_rigth + z_left +
                               (xp_nt + x_m) / ((xp_nt - x_m) ** 2 + (h_nt - z_m) ** 2))

    h_x_up = Iup / (4 * pi) * (x_rigth + x_left +
                               (-h_up + z_m) / ((xp_up - x_m) ** 2 + (h_up - z_m) ** 2))
    h_z_up = Iup / (4 * pi) * (z_rigth + z_left +
                               (xp_up + x_m) / ((xp_up - x_m) ** 2 + (h_up - z_m) ** 2))

    return mix(h_x_kp, h_z_kp) + mix(h_x_up, h_z_up) + mix(h_x_nt, h_z_nt)


def electric_calc(x_e, z_e, f_e):

    U_h = U * harm.get(f_e)[1]

    e_res = U_h * (
            log(1 + 4 * h_nt * z_e / ((x_e - xp_nt) ** 2 + (h_nt - z_e) ** 2)) / (2 * z_e * log(2 * h_nt / d_nt)) +
            log(1 + 4 * h_kp * z_e / ((x_e - xp_kp) ** 2 + (h_kp - z_e) ** 2)) / (2 * z_e * log(2 * h_kp / d_kp)) +
            log(1 + 4 * h_up * z_e / ((x_e - xp_up) ** 2 + (h_up - z_e) ** 2)) / (2 * z_e * log(2 * h_up / d_up))
    )
    return e_res


def energy_pass(x_e, z_e):
    res_energy = 0
    for freq in harm.keys():
        res_energy += magnetic_calc(x_e, z_e, freq) * electric_calc(x_e, z_e, freq)
    return res_energy


def visual_up(z=chel):
    print('График строится..................')

    Xmin = -0.5
    Xmax = all_length + 0.5
    Ymax = -1 * 0.5 * width * 1.3
    Ymin = xp_up * 1.15

    x = np.linspace(Xmin, Xmax, dis, endpoint=True)
    y = np.linspace(Ymin, Ymax, dis, endpoint=True)

    F = [[energy_pass(y_, z) for _ in x] for y_ in y]

    plt.figure(1)
    ct = plt.contour(x, y, F, alpha=0.75, colors='black', linestyles='dotted', levels=5)
    plt.clabel(ct, fontsize=10)
    plt.imshow(F, extent=[Xmin, Xmax, Ymax, Ymin], cmap='YlOrRd', alpha=0.95)
    plt.colorbar()

    for delta_y in [xp_kp, xp_up, xp_nt]:
        plt.hlines(delta_y, Xmin, Xmax, color='white', linewidth=2)
    plt.text(6, xp_kp+0.05, 'КП', color='white')
    plt.text(6.5, xp_up+0.05, 'УП', color='white')
    plt.text(5.5, xp_nt-0.3, 'НТ', color='white')

    plt.hlines(0.5 * width, 0, all_length, colors='red', linestyles='--')
    plt.hlines(-0.5 * width, 0, all_length, colors='red', linestyles='--')
    plt.vlines(0, -0.5 * width, 0.5 * width, colors='red', linestyles='--')
    plt.vlines(length, -0.5 * width, 0.5 * width, colors='red', linestyles='--')
    plt.vlines(all_length, -0.5 * width, 0.5 * width, colors='red', linestyles='--')

    plt.xlabel('Ось y, метры')
    plt.ylabel('Ось x, метры')

    plt.title('Вид сверху без электровоза')

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_без_локомотива_U_{U}_В_I_{I}_В.png"
    plt.savefig(name)

    print('График построен.')


def visual_front():
    print('График строится..................')

    Xmin = -1 * max(xp_r, width) * 1.15
    Xmax = xp_up * 1.2
    Zmax = 0.1
    Zmin = max(h_kp, h_nt, h_up) * 1.1

    x = np.linspace(Xmin, Xmax, dis, endpoint=True)
    z = np.linspace(Zmin, Zmax, dis, endpoint=True)

    F = [[energy_pass(x_, z_) for x_ in x] for z_ in z]

    plt.figure(2)
    b = 10 ** (len(str(round(np.amin(F)))) - 1)
    ct = plt.contour(x, z, F, alpha=0.75, colors='black', linestyles='dotted',
                     levels=[b, 2*b, 5*b, 7*b, 10*b, 20*b, 50*b, 100*b])
    plt.clabel(ct, fontsize=10)

    plt.imshow(F, extent=[Xmin, Xmax, Zmax, Zmin], cmap='YlOrRd', alpha=0.95, norm=colors.LogNorm())
    plt.colorbar()

    plt.text(xp_kp, h_kp, 'КП', color='white',  fontsize=14)
    plt.text(xp_up, h_up, 'УП', color='white', fontsize=14)
    plt.text(xp_nt, h_nt, 'НТ', color='white', fontsize=14)

    def lines_():
        plt.hlines(height + floor, -0.5 * width, 0.5 * width, colors='red', linestyles='--')
        plt.hlines(floor, -0.5 * width, 0.5 * width, colors='red', linestyles='--')
        plt.hlines(1, -0.5 * width, 0.5 * width, colors='red', linestyles='--')
        plt.vlines(-0.5 * width, 1, height+floor, colors='red', linestyles='--')
        plt.vlines(0.5 * width, 1, height+floor, colors='red', linestyles='--')

    lines_()
    plt.xlabel('Ось x, метры')
    plt.ylabel('Ось z, метры')

    plt.title('Вид сбоку')

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_вид сбоку_U_{U}_В_I_{I}_В.png"
    plt.savefig(name)

    plt.figure(3)
    x_st = np.where(x == max([x_ for x_ in x if x_ <= -0.5 * width * 1.15]))[0][0]
    x_ed = np.where(x == min([x_ for x_ in x if x_ >= 0.5 * width * 1.15]))[0][0]
    z_st = np.where(z == min([z_ for z_ in z if z_ >= (height + floor) * 1.15]))[0][0]
    z_ed = np.where(z == max([z_ for z_ in z if z_ <= 0.2]))[0][0]

    G = np.array(F)[z_st:z_ed, x_st:x_ed]

    ct = plt.contour(x[x_st:x_ed], z[z_st:z_ed], G, alpha=0.75, colors='black', linestyles='dotted',
                     levels=[b, 2*b, 5*b, 7*b, 10*b, 20*b, 50*b, 100*b])
    plt.clabel(ct, fontsize=10)
    plt.imshow(G, extent=[-0.5 * width * 1.15, 0.5 * width * 1.15, 0.2, (height + floor) * 1.15],
               cmap='YlOrRd', alpha=0.95, norm=colors.LogNorm())
    plt.colorbar()
    lines_()
    plt.xlabel('Ось x, метры')
    plt.ylabel('Ось z, метры')

    plt.title('Вид сбоку увеличенный')

    # mng = plt.get_current_fig_manager()
    # mng.window.state('zoomed')

    name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_вид сбоку_увелич_U_{U}_В_I_{I}_В.png"
    plt.savefig(name)

    print('График построен.')


def locomotive_energy_pass(x_e, y_e, z_e):
    h_ext, e_ext = 0, 0

    for fr in harm.keys():
        h_ext += magnetic_calc(x_e, z_e, fr) / kh_metal[str(fr)]
        e_ext += electric_calc(x_e, z_e, fr) / ke_metal

    return [h_ext, e_ext]


def shina(x_p, y_p, shinas, I_s, U_s):
    dc = 7
    points = []
    for sh in shinas:
        x_arr = np.linspace(sh[0], sh[0]+sh[1], dc, endpoint=False)
        y_arr = np.linspace(sh[2], sh[2]+sh[3], dc, endpoint=False)
        z_arr = np.linspace(sh[4], sh[4]+sh[5], dc, endpoint=False)
        points.extend([[length+x_arr[i], y_arr[i]-width/2, floor+z_arr[i]] for i in range(0, len(x_arr))])

    def in_point(x_i, y_i):
        r = 0
        for p_ in points:
            r += 1 / ((x_i - p_[0]) ** 2 + (y_i - p_[1]) ** 2 + (chel - p_[2]) ** 2) ** 0.5
        return [I_s * sum_harm_I * r / (2 * pi * len(points)), U_s * sum_harm_U * r / len(points)]

    return [[in_point(x_, y_) for x_ in x_p] for y_ in y_p]


def zarayd(inf, type_='M'):
    if type_ == 'M':
        dis1, dis2 = 10, 7
    else:
        dis1, dis2 = 5, 3
    x_d = np.linspace(inf[0], inf[1], dis1, endpoint=True)
    y_d = np.linspace(inf[2], inf[3], dis2, endpoint=False)
    z_d = np.linspace(inf[4], inf[5], dis2, endpoint=False)
    zarayd_ = [[x_, y_d[0], z_] for x_ in x_d for z_ in z_d]
    zarayd_.extend([[x_, y_d[-1], z_] for x_ in x_d for z_ in z_d])
    zarayd_.extend([[x_d[0], y_, z_] for y_ in y_d[1:-1] for z_ in z_d])
    zarayd_.extend([[x_d[-1], y_, z_] for y_ in y_d[1:-1] for z_ in z_d])
    return zarayd_

minus = zarayd([length, all_length, width/2, -width/2, floor, height])


def oborud(x_arr, y_arr, element, I_g, U_g):
    dc = 5
    points = []
    plus = []
    for el in element:
        x_p = np.linspace(el[0], el[1], dc, endpoint=True)
        y_p = np.linspace(el[2], el[3], dc, endpoint=True)
        z_p = np.linspace(el[4], el[5], dc, endpoint=True)
        points.extend([[length + x_, y_ - width/2, floor+z_] for x_ in x_p for y_ in y_p for z_ in z_p])
        plus.extend(zarayd(el, type_='P'))

    def in_point(x_, y_):
        H_ob, E_ob = 0, 0
        for p_ in points:
            r = ((p_[0]-x_)**2 + (p_[1]-y_)**2 + (p_[2]-chel)**2) ** 0.5
            H_ob += I_g / (pi * l_gk) * atan(l_gk / (2 * r))


            # TODO очень долго считает
            for mn in minus_p:
                r_m = ((mn[0] - x_) ** 2 + (mn[1] - y_) ** 2 + (mn[2] - chel) ** 2) ** 0.5
                for pl in plus_p:
                    r_p = ((pl[0] - x_) ** 2 + (pl[1] - y_) ** 2 + (pl[2] - chel) ** 2) ** 0.5
                    E_ob += U_g * (r_p + r_m) / r_p / r_m

            # TODO не так. Учитываем направление!
            for mn in minus:
                r_m = ((mn[0] - x_) ** 2 + (mn[1] - y_) ** 2 + (mn[2] - chel) ** 2) ** 0.5
                E_ob += U_g / r_m

            for pl in plus:
                r_p = ((pl[0] - x_) ** 2 + (pl[1] - y_) ** 2 + (pl[2] - chel) ** 2) ** 0.5
                E_ob += U_g / r_p

        return [H_ob * sum_harm_I / len(points), E_ob * sum_harm_U / len(plus) / len(points)]

    return [[in_point(x_, y_) for x_ in x_arr] for y_ in y_arr]


def up_lines():
    li = '--'

    def do_draw(h_lines, v_lines, c):
        for h in h_lines:
            plt.hlines(h[2] - width / 2, length + h[0], length + h[1], colors=c, linestyles=li)
        for w in v_lines:
            plt.vlines(length + w[0], w[1] - width / 2, w[2] - width / 2, colors=c, linestyles=li)

    h_lines_ = [[x_tt, x_tt+l_tt, y_tt],
               [x_tt, x_tt+l_tt, y_tt+h_tt],
               [x_gk, x_gk+l_gk, y_gk],
               [x_gk, x_gk+l_gk, y_gk+h_gk]]
    v_lines_ = [[x_tt-d_tt, y_tt+0.2, y_tt+h_tt-0.2],
               [x_tt+l_tt+d_tt, y_tt+0.2, y_tt+h_tt-0.2],
               [x_gk, y_gk, y_gk+h_gk],
               [x_gk+l_gk, y_gk, y_gk+h_gk]]
    do_draw(h_lines_, v_lines_, 'white')

    h_lines_ = [[x_gk, x_gk-0.8, y_gk+h_gk + 0.5],
                [x_gk - 0.8, x_gk - 0.8-0.15, y_gk + h_gk + 0.5 - 0.3],
                [x_gk-0.8, x_gk-0.8-0.15, y_gk+h_gk + 0.5 - 0.3 - 1.4],

                [x_gk+l_gk, x_gk+0.8+l_gk, y_gk+h_gk + 0.5],
                [x_gk+0.8+l_gk, x_gk+0.8+0.15+l_gk, y_gk + h_gk + 0.5 - 0.3],
                [x_gk+0.8+l_gk, x_gk+0.8+0.15+l_gk, y_gk+h_gk + 0.5 - 0.3 - 1.4]
                ]
    v_lines_ = [[x_gk, y_gk+h_gk, y_gk+h_gk+0.5],
                [x_gk-0.8, y_gk+h_gk + 0.5, y_gk+h_gk + 0.5-0.8],
                [x_gk-0.8, y_gk+h_gk + 0.5 - 0.8, y_gk+h_gk + 0.5 - 0.8-0.9],

                [x_gk+l_gk, y_gk+h_gk, y_gk+h_gk+0.5],
                [x_gk+0.8+l_gk, y_gk+h_gk + 0.5, y_gk+h_gk + 0.5-0.8],
                [x_gk+0.8+l_gk, y_gk+h_gk + 0.5 - 0.8, y_gk+h_gk + 0.5 - 0.8-0.9]
                ]
    do_draw(h_lines_, v_lines_, 'yellow')

    h_lines_ = [[x_vu1, x_vu1-l_vu, y_vu1],
                [x_vu1, x_vu1-l_vu, y_vu1+h_vu],
                [x_vu1, x_vu1 - l_vu, y_vu2],
                [x_vu1, x_vu1 - l_vu, y_vu2 + h_vu],

                [x_vu2, x_vu2 + l_vu, y_vu1],
                [x_vu2, x_vu2 + l_vu, y_vu1 + h_vu],
                [x_vu2, x_vu2 + l_vu, y_vu2],
                [x_vu2, x_vu2 + l_vu, y_vu2 + h_vu],

                [x_cp1, x_cp1-l_cp, y_cp-h_cp/2],
                [x_cp1, x_cp1-l_cp, y_cp+h_cp/2],
                [x_cp2, x_cp2+l_cp, y_cp-h_cp/2],
                [x_cp2, x_cp2+l_cp, y_cp+h_cp/2]]

    v_lines_ = [[x_vu1, y_vu1, y_vu1+h_vu],
                [x_vu1-l_vu, y_vu1, y_vu1+h_vu],
                [x_vu1, y_vu2, y_vu2 + h_vu],
                [x_vu1 - l_vu, y_vu2, y_vu2 + h_vu],

                [x_vu2, y_vu1, y_vu1 + h_vu],
                [x_vu2 + l_vu, y_vu1, y_vu1 + h_vu],
                [x_vu2, y_vu2, y_vu2 + h_vu],
                [x_vu2 + l_vu, y_vu2, y_vu2 + h_vu],

                [x_cp1, y_cp-h_cp/2, y_cp+h_cp/2],
                [x_cp1-l_cp, y_cp-h_cp/2, y_cp+h_cp/2],
                [x_cp2, y_cp-h_cp/2, y_cp+h_cp/2],
                [x_cp2+l_cp, y_cp-h_cp/2, y_cp+h_cp/2]]

    do_draw(h_lines_, v_lines_, 'blue')

    h_lines_ = [[x_vu2, x_vu2-1, y_vu1+0.2],
                [x_vu2, x_vu2-1, y_vu2+0.2],
                [x_vu1, x_vu1+1, y_vu1+0.2],
                [x_vu1, x_vu1+1, y_vu2+0.2],
                [x_vu2-1, x_vu2-1+d_vu, y_vu1+0.83+0.2],
                [x_vu1+1, x_vu1+1-d_vu, y_vu1+0.83+0.2],
                [x_cp2 + l_cp, x_cp2 + l_cp + 2.5, y_cp],
                [x_cp1 - l_cp, x_cp1 - l_cp - 2.5, y_cp]]
    v_lines_ = [[x_vu1+1, y_vu1+0.2, y_vu2+0.2],
                [x_vu2-1, y_vu1+0.2, y_vu2+0.2]]

    do_draw(h_lines_, v_lines_, 'turquoise')

    h_lines_ = [[x_td1_sr-l_td*0.5, x_td1_sr + l_td*0.5, 0.5*(width-h_td)],
                [x_td1_sr-l_td*0.5, x_td1_sr+l_td*0.5, 0.5*(width+h_td)],
                [x_td1_sr+kol_par-l_td*0.5, x_td1_sr+kol_par+l_td*0.5, 0.5*(width-h_td)],
                [x_td1_sr+kol_par-l_td*0.5, x_td1_sr+kol_par+l_td*0.5, 0.5*(width+h_td)],

                [x_td2_sr - kol_par - l_td * 0.5, x_td2_sr - kol_par + l_td * 0.5, 0.5 * (width - h_td)],
                [x_td2_sr - kol_par - l_td * 0.5, x_td2_sr - kol_par + l_td * 0.5, 0.5 * (width + h_td)],
                [x_td2_sr - l_td * 0.5, x_td2_sr + l_td * 0.5, 0.5 * (width - h_td)],
                [x_td2_sr - l_td * 0.5, x_td2_sr + l_td * 0.5, 0.5 * (width + h_td)]
                ]

    v_lines_ = [[x_td1_sr-l_td*0.5, 0.5*(width-h_td), 0.5*(width+h_td)],
                [x_td1_sr+l_td*0.5, 0.5*(width-h_td), 0.5*(width+h_td)],
                [x_td1_sr + kol_par - l_td * 0.5, 0.5 * (width - h_td), 0.5 * (width + h_td)],
                [x_td1_sr + kol_par + l_td * 0.5, 0.5 * (width - h_td), 0.5 * (width + h_td)],

                [x_td2_sr - kol_par - l_td * 0.5, 0.5 * (width - h_td), 0.5 * (width + h_td)],
                [x_td2_sr - kol_par + l_td * 0.5, 0.5 * (width - h_td), 0.5 * (width + h_td)],
                [x_td2_sr - l_td * 0.5, 0.5 * (width - h_td), 0.5 * (width + h_td)],
                [x_td2_sr + l_td * 0.5, 0.5 * (width - h_td), 0.5 * (width + h_td)]
                ]

    do_draw(h_lines_, v_lines_, 'red')


def visual_up_locomotive(z_vl=chel):
    print('График строится..................')

    Xmin = 0
    Xmax = all_length
    Ymax = -0.5 * width
    Ymin = -1 * Ymax

    x_ln = np.linspace(Xmin, Xmax, dis, endpoint=True)
    y_ln = np.linspace(Ymin, Ymax, dis, endpoint=True)

    print('.....расчёт внешнего поля')
    F = [[locomotive_energy_pass(y_, x_, z_vl) for x_ in x_ln] for y_ in y_ln]

    print('.....расчёт поля от шин')

    mednaya_truba = [[x_mt, 0, y_mt, 0, height, 0]]
    med_trub = shina(x_ln, y_ln, mednaya_truba, 550, 27000)

    shinas_tt_gk = [[x_tt, (l_tt/6)*i, y_tt, 0, 0, -1.5] for i in range(0, 7)]
    sh_tt_gk = shina(x_ln, y_ln, shinas_tt_gk, 1750, 1218)

    shinas_gk_vu = [[x_gk, 0, y_gk+h_gk, 0.5, 0.65, 0],
              [x_gk, 0, y_gk+h_gk + 0.5, 0, 0.65, 0.8],
              [x_gk, -0.8, y_gk+h_gk + 0.5, 0, 0.65+0.8, 0],
              [x_gk-0.8, 0, y_gk+h_gk + 0.5, -0.8, 0.65 + 0.8, 0],
              [x_gk-0.8, 0, y_gk+h_gk + 0.5 - 0.8, 0, 0.65 + 0.8, -1.6],
              [x_gk - 0.8, -0.15, y_gk + h_gk + 0.5 - 0.3, 0, 0.65 + 0.8 - 1.1, 0],
              [x_gk-0.8, 0, y_gk+h_gk + 0.5 - 0.3, -0.9, 0.65 + 0.8 - 1.6, 0],
              [x_gk-0.8, -0.15, y_gk+h_gk + 0.5 - 0.3 - 1.4, 0, 0.65 + 0.8 - 1.6, 0],

                    [x_gk + l_gk, 0, y_gk + h_gk, 0.5, 0.65, 0],
                    [x_gk + l_gk, 0, y_gk + h_gk + 0.5, 0, 0.65, 0.8],
                    [x_gk + l_gk, -0.8, y_gk + h_gk + 0.5, 0, 0.65 + 0.8, 0],
                    [x_gk + l_gk + 0.8, 0, y_gk + h_gk + 0.5, -0.8, 0.65 + 0.8, 0],
                    [x_gk + l_gk + 0.8, 0, y_gk + h_gk + 0.5 - 0.8, 0, 0.65 + 0.8, -1.6],
                    [x_gk + l_gk + 0.8, + 0.15, y_gk + h_gk + 0.5 - 0.3, 0, 0.65 + 0.8 - 1.1, 0],
                    [x_gk + l_gk + 0.8, 0, y_gk + h_gk + 0.5 - 0.3, -0.9, 0.65 + 0.8 - 1.6, 0],
                    [x_gk + l_gk + 0.8, +0.15, y_gk + h_gk + 0.5 - 0.3 - 1.4, 0, 0.65 + 0.8 - 1.6, 0]
                    ]

    sh_gk_vu = shina(x_ln, y_ln, shinas_gk_vu, 3150, 1400)

    print('.....расчёт поля от оборудования')

    tt = [x_tt, x_tt+l_tt, y_tt, y_tt + w_tt, 0.5, 0.5-h_tt]
    tt_field = oborud(x_ln, y_ln, [tt], 750, 1218)

    gk = [x_gk, x_gk+l_gk, y_gk, y_gk+w_gk, z_gk, z_gk+h_gk]
    gk_field = oborud(x_ln, y_ln, [gk], 1300, 3000)

    all_field = np.array(sh_tt_gk) + np.array(sh_gk_vu) + np.array(med_trub) + \
                + np.array(F) +\
                np.array(tt_field) + np.array(gk_field)

    print('.....расчёт экрана')

    res_h_field = [[d[0] for d in el] for el in all_field]
    res_e_field = [[d[1] for d in el] for el in all_field]

    kuzov_ind = np.where(x_ln == max([x_ for x_ in x_ln if x_ <= length]))[0][0]
    kamera_ind = np.where(x_ln == max([x_ for x_ in x_ln if x_ <= length + 0.6]))[0][0]

    h_ekr_sp_set = np.diag([koef_ekr_h_splosh_z * koef_ekr_h_setka if i <= kuzov_ind else
                            koef_ekr_h_setka if i <= kamera_ind else 1
                            for i in range(0, dis)])
    h_ekr_sp_sp = np.diag([koef_ekr_h_splosh_z * koef_ekr_h_splosh_v if i <= kuzov_ind else
                           koef_ekr_h_splosh_v if i <= kamera_ind else 1
                           for i in range(0, dis)])
    e_ekr_sp_set = np.diag([koef_ekr_e_splosh * koef_ekr_e_setka if i <= kuzov_ind else
                            koef_ekr_e_setka if i <= kamera_ind else 1
                            for i in range(0, dis)])
    e_ekr_sp_sp = np.diag([koef_ekr_e_splosh * koef_ekr_e_splosh if i <= kuzov_ind
                           else koef_ekr_e_splosh if i <= kamera_ind else 1
                           for i in range(0, dis)])

    res_h_sp_st = np.dot(res_h_field, h_ekr_sp_set)
    res_e_sp_st = np.dot(res_e_field, e_ekr_sp_set)

    res_h_sp_sp = np.dot(res_h_field, h_ekr_sp_sp)
    res_e_sp_sp = np.dot(res_e_field, e_ekr_sp_sp)

    res_sp_st = np.multiply(res_h_sp_st, res_e_sp_st)
    res_sp_sp = np.multiply(res_h_sp_sp, res_e_sp_sp)

    chel_x = np.where(x_ln == max([x_ for x_ in x_ln if x_ <= x_chel]))[0][0]
    chel_y = np.where(y_ln == max([y_ for y_ in y_ln if y_ <= y_chel]))[0][0]
    en_chel_sp_st = res_sp_st[chel_y, chel_x]
    en_chel_sp_sp = res_sp_sp[chel_y, chel_x]

    def graph_do(znach, graph_name):
        ct = plt.contour(x_ln, y_ln, znach, alpha=0.75, colors='black', linestyles='dotted', levels=5)
        plt.clabel(ct, fontsize=10)
        plt.imshow(znach, extent=[Xmin, Xmax, Ymax, Ymin], cmap='YlOrRd', alpha=0.95, norm=colors.LogNorm())
        plt.colorbar()

        if graph_name != 'контактная сеть':
            up_lines()

        plt.xlabel('Ось y, метры')
        plt.ylabel('Ось x, метры')

        plt.title(f"Электровоз {graph_name} вид сверху")

        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')

        name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_электровоз_{graph_name}_переменное_вид_сверху_U_{U}_В_I_{I}_В.png"
        plt.savefig(name)

    plt.figure(4)
    res_cs = [[d[0] * d[1] for d in el] for el in F]
    graph_do(res_cs, 'контактная сеть')
    plt.figure(5)
    res_all = [[d[0] * d[1] for d in el] for el in all_field]
    graph_do(res_all, 'без экранов')
    plt.figure(6)
    graph_do(res_sp_st, 'экран сетка')
    plt.figure(7)
    graph_do(res_sp_sp, 'сплошной экран')

    print('График построен.')
    return en_chel_sp_st, en_chel_sp_sp


def shina_post(x_p, y_p, z_, shinas, I_s, U_s):
    dc = 7
    points = []
    for sh in shinas:
        x_arr = np.linspace(sh[0], sh[0]+sh[1], dc, endpoint=True)
        y_arr = np.linspace(sh[2], sh[2]+sh[3], dc, endpoint=True)
        z_arr = np.linspace(sh[4], sh[4]+sh[5], dc, endpoint=True)
        points.extend([[length+x_arr[i], y_arr[i]-width/2, floor+z_arr[i]] for i in range(0, len(x_arr))])

    def in_point(x_i, y_i):
        r = 0
        for p_ in points:
            r += 1 / ((x_i - p_[0]) ** 2 + (y_i - p_[1]) ** 2 + (z_ - p_[2]) ** 2) ** 0.5
        return [I_s * r / (2 * pi * len(points)), U_s * r / len(points)]

    return [[in_point(x_, y_) for x_ in x_p] for y_ in y_p]


def oborud_post(x_arr, y_arr, element, I_g, U_g, n):
    z_po = chel
    dc = 4
    points, plus = [], []
    for el in element:
        x_p = np.linspace(el[0], el[1], dc, endpoint=True)
        y_p = np.linspace(el[2], el[3], dc, endpoint=True)
        z_p = np.linspace(el[4], el[5], dc, endpoint=True)
        points.extend([[length + x_, y_ - width/2, floor+z_] for x_ in x_p for y_ in y_p for z_ in z_p])
        plus = zarayd(el, type_='P')

    def in_point(x_, y_):
        H_ob, E_ob = 0, 0
        for p in points:
            r = ((p[0]-x_)**2 + (p[1]-y_)**2 + (p[2]-z_po)**2) ** 0.5
            H_ob += I_g / (pi * l_gk) * atan(l_gk / 2 / r)

            # TODO и тут поправит
            for pl in plus:
                r_p = ((pl[0] - x_) ** 2 + (pl[1] - y_) ** 2 + (pl[2] - chel) ** 2) ** 0.5
                E_ob += U_g / r_p

        return [H_ob * n / len(points), E_ob / len(plus) / len(points)]

    return [[in_point(x_, y_) for x_ in x_arr] for y_ in y_arr]


def visual_up_post(z_p=chel):
    print('График строится..................')

    Xmin = 0
    Xmax = all_length
    Ymax = -0.5 * width
    Ymin = -1 * Ymax

    x_ln = np.linspace(Xmin, Xmax, dis, endpoint=True)
    y_ln = np.linspace(Ymin, Ymax, dis, endpoint=True)

    print('.....расчёт поля от шин')

    shinas_vu_sr = [[x_vu2, -1, y_vu1+0.2, 0, 0, 0],
                    [x_vu2, -1, y_vu2+0.2, 0, 0, 0],
                    [x_vu1, +1, y_vu1+0.2, 0, 0, 0],
                    [x_vu1, +1, y_vu2+0.2, 0, 0, 0],
                    [x_vu2-1, 0.6, y_vu1+0.83+0.2, 0, 0, 0],
                    [x_vu1+1, -0.6, y_vu1+0.83+0.2, 0, 0, 0],
                    [x_vu1 + 1, 0, y_vu1 + 0.2, y_vu2 + 1.4, 0, 0],
                    [x_vu2 - 1, 0, y_vu1 + 0.2, y_vu2 + 1.4, 0, 0]]

    shinas_sr_ted = [[x_cp2+l_cp, 0, y_cp, 0, 0, 1.900],
                     [x_cp2+l_cp, 0, y_cp, 0, 1.900, -0.800],
                     [x_cp2+l_cp, 2.500, y_cp, 0, 1.100, 0],
                     [x_cp2+l_cp+2.500, 0, y_cp, 0, 1.100, -1.700-0.500],

                     [x_cp1-l_cp, 0, y_cp, 0, 0, 1.900],
                     [x_cp1-l_cp, 0, y_cp, 0, 1.900, -0.800],
                     [x_cp1-l_cp, -2.500, y_cp, 0, 1.100, 0],
                     [x_cp1-l_cp - 2.500, 0, y_cp, 0, 1.100, -1.700 - 0.500],]

    sh_vu_sr = shina_post(x_ln, y_ln, z_p, shinas_vu_sr, 3150, 1400)
    sh_sr_ted = shina_post(x_ln, y_ln, z_p, shinas_sr_ted, 880, 950)

    print('.....расчёт поля от оборудования')

    vu = [[x_vu1, x_vu1 - l_vu, y_vu1, y_vu1 + h_vu, 0.6, z_vu],
          [x_vu1, x_vu1 - l_vu, y_vu2, y_vu2 + h_vu, 0.6, z_vu],
          [x_vu2, x_vu2 + l_vu, y_vu1, y_vu1 + h_vu, 0.6, z_vu],
          [x_vu2, x_vu2 + l_vu, y_vu2, y_vu2 + h_vu, 0.6, z_vu]]

    sr = [[x_cp1, x_cp1-l_cp, y_cp-0.5*h_cp, y_cp+0.5*h_cp, 0.4, z_cp],
          [x_cp2, x_cp2+l_cp, y_cp-0.5*h_cp, y_cp+0.5*h_cp, 0.4, z_cp]]

    ted = [[x_td1_sr - l_td * 0.5, x_td1_sr + l_td * 0.5,
            0.5 * (width + h_td), 0.5 * (width - h_td), 1-z_td*0.5, 1+z_td*0.5],
           [x_td1_sr + kol_par - l_td * 0.5, x_td1_sr + kol_par + l_td * 0.5,
            0.5 * (width + h_td), 0.5 * (width - h_td), 1 - z_td * 0.5, 1 + z_td * 0.5],
           [x_td2_sr - kol_par - l_td * 0.5, x_td2_sr - kol_par + l_td * 0.5,
            0.5 * (width + h_td), 0.5 * (width - h_td), 1 - z_td * 0.5, 1 + z_td * 0.5],
           [x_td2_sr - l_td * 0.5, x_td2_sr + l_td * 0.5,
            0.5 * (width + h_td), 0.5 * (width - h_td), 1 - z_td * 0.5, 1 + z_td * 0.5]
           ]

    vu_field = oborud_post(x_ln, y_ln, vu, 1850, 1500, 7)
    sr_field = oborud_post(x_ln, y_ln, sr, 3150, 1400, 1)
    ted_field = oborud_post(x_ln, y_ln, ted, 880, 1950, 5)

    all_field = np.array(sh_vu_sr) + np.array(sh_sr_ted)+\
                np.array(vu_field) + np.array(sr_field) +\
                np.array(ted_field)

    print('.....расчёт экрана')

    k_post_ekr_e_setka = 1 / (55.45 + 20*log(ds**2*metal_sigma/s_, 10))
    k_post_ekr_h_setka = 1 / exp(pi*d_v / s_)

    res_h_field = [[d[0] for d in el] for el in all_field]
    res_e_field = [[d[1] for d in el] for el in all_field]

    kuzov_ind = np.where(x_ln == max([x_ for x_ in x_ln if x_ <= length]))[0][0]
    kamera_ind = np.where(x_ln == max([x_ for x_ in x_ln if x_ <= length + 0.6]))[0][0]

    h_ekr_sp_set = np.diag([koef_ekr_h_splosh_z * k_post_ekr_h_setka if i <= kuzov_ind else
                            k_post_ekr_h_setka if i <= kamera_ind else 1
                            for i in range(0, dis)])
    h_ekr_sp_sp = np.diag([koef_ekr_h_splosh_z * koef_ekr_h_splosh_v if i <= kuzov_ind else
                           koef_ekr_h_splosh_v if i <= kamera_ind else 1
                           for i in range(0, dis)])
    e_ekr_sp_set = np.diag([koef_ekr_e_splosh * k_post_ekr_e_setka if i <= kuzov_ind else
                            k_post_ekr_e_setka if i <= kamera_ind else 1
                            for i in range(0, dis)])
    e_ekr_sp_sp = np.diag([koef_ekr_e_splosh * koef_ekr_e_splosh if i <= kuzov_ind
                           else koef_ekr_e_splosh if i <= kamera_ind else 1
                           for i in range(0, dis)])

    res_h_sp_st = np.dot(res_h_field, h_ekr_sp_set)
    res_e_sp_st = np.dot(res_e_field, e_ekr_sp_set)

    res_h_sp_sp = np.dot(res_h_field, h_ekr_sp_sp)
    res_e_sp_sp = np.dot(res_e_field, e_ekr_sp_sp)

    res_sp_st = np.multiply(res_h_sp_st, res_e_sp_st)
    res_sp_sp = np.multiply(res_h_sp_sp, res_e_sp_sp)

    chel_x = np.where(x_ln == max([x_ for x_ in x_ln if x_ <= x_chel]))[0][0]
    chel_y = np.where(y_ln == max([y_ for y_ in y_ln if y_ <= y_chel]))[0][0]
    en_chel_sp_st = res_sp_st[chel_y, chel_x]
    en_chel_sp_sp = res_sp_sp[chel_y, chel_x]

    def graph_do(znach, graph_name):
        ct = plt.contour(x_ln, y_ln, znach, alpha=0.75, colors='black', linestyles='dotted', levels=5)
        plt.clabel(ct, fontsize=10)
        plt.imshow(znach, extent=[Xmin, Xmax, Ymax, Ymin], cmap='YlOrRd', alpha=0.95, norm=colors.LogNorm())
        plt.colorbar()

        up_lines()

        plt.xlabel('Ось y, метры')
        plt.ylabel('Ось x, метры')

        plt.title(f"Электровоз {graph_name} постоянное вид сверху")

        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')

        name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_электровоз_{graph_name}_постоянное_вид_сверху_U_{U}_В_I_{I}_В.png"
        plt.savefig(name)

    plt.figure(8)
    res_all = [[d[0] * d[1] for d in el] for el in all_field]
    graph_do(res_all, 'без экранов')
    plt.figure(9)
    graph_do(res_sp_st, 'экран сетка')
    plt.figure(10)
    graph_do(res_sp_sp, 'сплошной экран')

    print('График построен.')
    return en_chel_sp_st, en_chel_sp_sp


## РАСЧЁТ СТАТИСТИКИ ##

S = (a * b / 3600) ** 1 / 2
p = ti / 24  # статистическая вероятность воздействия

print('\nПараметры сети')
print(f'Высота КП: {h_kp} м')
print(f'Высота НЧ: {h_nt} м')
print(f'Высота УП: {h_up} м')
print(f'Напряжение: {U} Вольт')
print(f'Суммарный ток: {I} Ампер')

## ПОСТРОЕНИЕ ГРАФИКА ##

print('\nБез электровоза')
visual_up()
print('\nБез электровоза, вид спереди')
visual_front()
print('\nЭлектровоз переменное поле')
e1, e2 = visual_up_locomotive()
print('\nЭлектровоз постоянное поле')
ep1, ep2 = visual_up_post()


print('\nВысота среза: %.2f' % chel)
field = 0
for f in harm.keys():
    field += electric_calc(x_chel, chel, f) * magnetic_calc(x_chel, chel, f)
print('\nПоле без экрана: %.4f' % field)
e = e1+ep1
print('\nПоле c экраном - сетка: %.4f' % e)
Dco = e * ti * S * p
Dpo = Dco / b
print('Удельная суточная доза поглощённой энергии: %.4f' % Dpo)
e = e2+ep2
print('\nПоле c экраном - сплошной: %.4f' % e)
Dco = e * ti * S * p
Dpo = Dco / b
print('Удельная суточная доза поглощённой энергии: %.4f' % Dpo)

plt.show()
