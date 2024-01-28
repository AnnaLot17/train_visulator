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

# ДАННЫЕ О ЛОКОМОТИВЕ

length = 1.3  # длина кабины
all_length = 15.2  # длина всего локомотива
width = 2.8  # ширина кабины
height = 2.6  # высота кабины
# min_x, max_x, min_y, max_y, min_z, max_z
bor = [0.2, 0.6, -1.2, 1.2, floor+1.5, floor+2.2]  # узлы окна
# min_x, max_x, min_z, max_z
sbor = [0.3, 1, floor+1.5, floor+2.2]  # узлы для бокового окна

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

min_up = Point(-0.5*width, sbor[3]).distance(Point(xp_up, h_up))
max_up = Point(-0.5*width, sbor[2]).distance(Point(xp_up, h_up))

# ЭКРАН
# стекло - высчитываем d для подсчёта энергии преломлённой волны
e1 = 1
e2 = 4
mu1 = 1
mu2 = 0.99

n1 = (e1*mu1) ** 0.5
n2 = (e2*mu2) ** 0.5
d_glass = 4*n1*n2/((n1-n2) ** 2)


# РАСЧЁТЫ

# по теореме Пифагора высчитывает значение вектора магнитного поля из составляющих х и y
def mix(h_x, h_zz):
    return (h_x ** 2 + h_zz ** 2) ** 0.5

# магнитное поле гармоники f для заданной координаты x и z
def magnetic_calc(x_m, z_m, f_m):
    return [1,1,1] # todo
    # общая сила тока гармоники
    I_h = I * harm.get(f_m)[0]

    # сила тока по проводам
    Ikp = 0.41 * I_h
    Int = 0.20 * I_h
    Iup = 0.39 * I_h

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

    # результат выполнения этой функции - значения магнитных полей КП, НТ, УП для выбранной гармоники
    return [hkp, hnt, hup]

# расчёт электрического поля для гармоники f в точке x, z
def electric_calc(x_e, z_e, f_e):
    return [1, 1, 1]  # todo
    # напряжение гармоники
    U_h = U * harm.get(f_e)[1]

    # электрическое поле от каждого провода
    ekp = U_h * log(1 + 4 * h_nt * z_e / ((x_e - xp_nt) ** 2 + (h_nt - z_e) ** 2)) / (2 * z_e * log(2 * h_nt / d_nt))
    ent = U_h * log(1 + 4 * h_kp * z_e / ((x_e - xp_kp) ** 2 + (h_kp - z_e) ** 2)) / (2 * z_e * log(2 * h_kp / d_kp))
    eup = U_h * log(1 + 4 * h_up * z_e / ((x_e - xp_up) ** 2 + (h_up - z_e) ** 2)) / (2 * z_e * log(2 * h_up / d_up))

    # результат - список значений полей от каждого провода
    return [ekp, ent, eup]

# суммироввание всех полей для каждой точки:
def full_field(res_en):
    sum_h, sum_e, sum_g = 0, 0, 0
    for en in res_en[0].values():
        sum_h += sum(en[0])  # магнитная составляющая
        sum_e += sum(en[1])  # электрическая составляющая
        # для расчёта энергии, перемножаем значение магнитного и электрического поля, затем складываем полученные значения
        sum_g += en[0][0] * en[1][0] + en[0][1] * en[1][1] + en[0][2] * en[1][2]
    # возвращаем значения магнитной, электрической и энергетической составляющей
    return [sum_h, sum_e, sum_g]


#  расчёт экрана переменного поля
def ekran(en):

    x, y, z = en[1]  # координаты точки

    # расстояние от текущей точки до КТ и НТ - для расчёта лобовых окон
    kppth = LineString([(x, y, z), (x, xp_kp, h_kp)])
    ntpth = LineString([(x, y, z), (x, xp_nt, h_nt)])

    # проверяем, попадает ли лобовое окно
    kp_pass = kppth.intersects(frontWindleft) or kppth.intersects(frontWindright)
    nt_pass = ntpth.intersects(frontWindleft) or ntpth.intersects(frontWindright)

    # для каждого провода проверяем, попадает ли текущая точка в тень от бокового окна или нет
    kp_dist = Point(y, z).distance(Point(xp_kp, h_kp))  # направление от точки до провода
    # пересекает ли это направление окно
    kp_pass |= (kp_dist >= min_kp) and (kp_dist <= max_kp) and (x >= sbor[0]) and (x <= sbor[1])

    nt_dist = Point(y, z).distance(Point(xp_nt, h_nt))
    nt_pass |= (nt_dist >= min_nt) and (nt_dist <= max_nt) and (x >= sbor[0]) and (x <= sbor[1])

    up_dist = Point(y, z).distance(Point(xp_up, h_up))
    up_pass = (up_dist >= min_up) and (up_dist <= max_up) and (x >= sbor[0]) and (x <= sbor[1])

    # для каждой точки внутри кабины проверяем, проходит ли для неё какое-либо поле через стекло
    # если да - применяем коэффициент преломлённой волны

    # сталь: электрическое поле полностью отражается, магнитное полностью затухает
    # стекло: и электрическое, и магнитное домножаются на d_glass по формуле:
    # Эпрел = Эпад*d = (ExH)*d = E*d x H*d
    # TODO
    '''
    if (abs(y) <= 0.5*width) and (z >= floor) and (z <= floor+height):
        # внутри кабины
        if kp_pass:
            # поле КП через стекло
        if nt_pass:
            # поле НТ через стекло
        if up_pass:
            # поле УП через стекло
        if not (kp_pass or nt_pass or up_pass):
            # вообще всё по нулям




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
    '''
    return en


# ГРАФИКА И ВЫВОД

# сохранение файла с картинкой
def show(name):
    mng = plt.get_current_fig_manager()  # захват изображения 
    # mng.window.state('zoomed')  # вывод изображения на весь экран если граф.оболочка это поддерживает
    # todo plt.savefig(f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{name}.png")
    # сохранение картинки в файл дата_время_название.png в папку со скриптом

# рисование линий кабины вид спереди
def fr_kab_lines():
    ln_ = '--'

    cl_ = 'royalblue'
    plt.hlines(bor[4], bor[2], bor[3], colors=cl_, linestyles=ln_)
    plt.hlines(bor[5], bor[2], bor[3], colors=cl_, linestyles=ln_)
    plt.vlines(bor[2], bor[4], bor[5], colors=cl_, linestyles=ln_)
    plt.vlines(bor[3], bor[4], bor[5], colors=cl_, linestyles=ln_)
    plt.vlines(0, bor[4], bor[5], colors=cl_, linestyles=ln_)

    cl_ = 'red'
    plt.hlines(bor[4] - .3, -0.5 * width, 0.5 * width, colors=cl_, linestyles='solid', lw=3)
    plt.hlines(floor + .3, -0.5 * width, 0.5 * width, colors=cl_, linestyles='solid', lw=3)
    plt.hlines(1.4, -0.5 * width, 0.5 * width, colors=cl_, linestyles='solid', lw=4)
    plt.scatter(0, 2.7, s=200, marker='*', color=cl_)

    cl_ = 'forestgreen'
    plt.hlines(height + floor, -0.5 * width, 0.5 * width, colors=cl_, linestyles=ln_)
    plt.hlines(floor + .1, -0.5 * width, 0.5 * width, colors=cl_, linestyles=ln_)
    plt.hlines(1, -0.5 * width, 0.5 * width, colors=cl_, linestyles=ln_)
    plt.vlines(-0.5 * width, 1, height + floor, colors=cl_, linestyles=ln_)
    plt.vlines(0.5 * width, 1, height + floor, colors=cl_, linestyles=ln_)

    bj1 = ptch.Arc((0, floor + height), width, 1, theta1=0, theta2=180, color=cl_, linestyle=ln_)
    bj2 = ptch.Circle((0, floor + height + 0.3), 0.2, color=cl_, linestyle=ln_, fill=None)

    for bj in [bj1, bj2]:
        plt.gca().add_artist(bj)

# рисование линий кабины вид спереди - для варианта с экраном (более схематично + стулья)
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


# рисование линий кабины вид сверху
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


# построение вида сверху без электровоза
def visual_up():
    print('График строится..................')

    # границы графика
    Xmin = -0.5
    Xmax = length + 0.5
    Ymax = 1 * 0.5 * width * 1.3
    Ymin = xp_up * 1.15

    # разбиение по точкам
    x = np.linspace(Xmin, Xmax, dis)
    y = np.linspace(Ymin, Ymax, dis)

    # расчёт значений полей для каждой точки графика
    every_f = [[[{fr: [magnetic_calc(y_, z_graph, fr), electric_calc(y_, z_graph, fr)] for fr in harm.keys()},
                 (x_, y_, z_graph)] for x_ in x] for y_ in y]

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
        for delta_y in [xp_kp, xp_up, xp_nt]:
            plt.hlines(delta_y, Xmin, Xmax, color='black', linewidth=2)
        plt.text(0.1, xp_kp+0.05, 'КП', color='white')
        plt.text(0.1, xp_up+0.05, 'УП', color='black')
        plt.text(1, xp_nt-0.3, 'НТ', color='white')

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
    Ymax = 1 * max(xp, width) * 1.15
    Ymin = xp_up * 1.2
    Zmax = 0.1
    Zmin = max(h_kp, h_nt, h_up) * 1.1

    # разбиение на точки
    y = np.linspace(Ymin, Ymax, dis)
    z = np.linspace(Zmin, Zmax, dis)

    # расчёт значений полей для каждой точки графика
    every_f = [[({fr: (magnetic_calc(y_, z_, fr), electric_calc(y_, z_, fr)) for fr in harm.keys()},
                 (x_chel, y_, z_)) for y_ in y] for z_ in z]

    # считаем итоговое значение для каждой точки
    all_field = [[full_field(x_el) for x_el in y_list] for y_list in every_f]
    summar = [[x_el[2] for x_el in y_list] for y_list in all_field]

    # создаём новое окно
    global gph_num
    gph_num += 1
    plt.figure(gph_num)
    # задаём уровни
    b = 10 ** (len(str(round(np.amin(summar)))) - 1)  # для правильного отображения линий
    # создаём объект точек графика
    ct = plt.contour(y, z, summar, alpha=0.75, colors='black', linestyles='dotted',
                     levels=[b, 2*b, 5*b, 7*b, 10*b, 20*b, 50*b, 100*b, 200*b, 500*b, 700*b])
    # создаём линии уровней из объекта точек     
    plt.clabel(ct, fontsize=10)
    # отрисовка    
    plt.imshow(summar, extent=[Ymin, Ymax, Zmax, Zmin], cmap=cmap, alpha=0.95, norm=colors.LogNorm())
    # раскраска   
    plt.colorbar()

    # названия проводов
    plt.text(xp_kp, h_kp, 'КП', color='white',  fontsize=14)
    plt.text(xp_up, h_up, 'УП', color='white', fontsize=14)
    plt.text(xp_nt, h_nt, 'НТ', color='white', fontsize=14)

    # очертания кабины
    fr_kab_lines()

    # название осей 
    plt.xlabel('Ось y, метры')
    plt.ylabel('Ось z, метры')

    plt.title('Контактная сеть вид спереди (без электровоза)') # подпись названия

    show('вид сбоку') # вывести и сохранить

    print('График построен.')
    return every_f   # возвращаем поле для перерасчёта с локомотивом

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

    # разбиение на точки
    x_ln = np.linspace(Xmin, Xmax, len(inside[0]))
    y_ln = np.linspace(Ymin, Ymax, len(inside))

    # формируем массивы значений на магнитную, электрическую составляющую и энергию
    magnetic = [[x_el[0] for x_el in y_list] for y_list in inside]
    electric = [[x_el[1] for x_el in y_list] for y_list in inside]
    energy = [[x_el[2] for x_el in y_list] for y_list in inside]

    # общая функция отрисовки графика
    def graph_do(znach, name_, x_lb='', y_lb=''):
        # # создаём объект точек графика
        # ct = plt.contour(x_ln, y_ln, znach, alpha=0.95, colors='black', linestyles='dotted', levels=5)
        # # создаём линии уровней из объекта точек         
        # plt.clabel(ct, fontsize=10)
        # #TODO линии уровней убрали чтобы не загораживали график
        # отрисовка
        plt.imshow(znach, extent=[Xmin, Xmax, Ymax, Ymin], cmap='YlOrRd', alpha=0.95, norm=colors.LogNorm())
        # раскраска          
        plt.colorbar()

        # название осей 
        plt.xlabel(x_lb)
        plt.ylabel(y_lb)

        plt.title(name_)  # подпись названия

    # отрисовка по очереди магнитного, электрического и энергии 
    global gph_num
    gph_num += 1
    plt.figure(gph_num)
    name = 'Кабина вид сверху (c экраном)'
    #TODO только энергия
    plt.title(name)
    plt.subplot(1, 3, 1)
    kab_lines_up()
    graph_do(magnetic, 'Магнитное', x_lb='Ось x, метры', y_lb='Ось y, метры')
    plt.subplot(1, 3, 2)
    kab_lines_up()
    graph_do(electric, 'Электрическое', x_lb='Ось x, метры')
    plt.subplot(1, 3, 3)
    kab_lines_up()
    graph_do(energy, 'Энергия', x_lb='Ось x, метры',)
    show(name)


def visual_front_locomotive(ext_f):
    print('График строится..................')

    # границы графика
    Ymin, Ymax = -0.6*width, 0.6*width
    Zmin, Zmax = floor+height+1, 0.1

    # применяем экран
    ekran_ = [[ekran(y_el) for y_el in z_list if abs(y_el[1][1]) <= Ymax] for z_list in ext_f
              if z_list[0][1][2] < Zmin]

    # перевод значений посчитанных для каждой гармоники каждого провода в одно значение
    summar = [[full_field(x_el) for x_el in y_list] for y_list in ekran_]
    # составление графика на магнитную, электрическую составляющую и энергию
    magnetic = [[x_el[0] for x_el in y_list] for y_list in summar]
    electric = [[x_el[1] for x_el in y_list] for y_list in summar]
    energy = [[x_el[2] for x_el in y_list] for y_list in summar]

    # разбиение по точкам
    y_ln = np.linspace(Ymin, Ymax, len(ekran_[0]))
    z_ln = np.linspace(Zmin, Zmax, len(ekran_))
    # находим координаты человека в массиве точек
    chel_y = np.where(y_ln == max([y_ for y_ in y_ln if y_ <= y_chel]))[0][0]
    chel_z = np.where(z_ln == max([z_ for z_ in z_ln if z_ <= z_chel]))[0][0]

    # общая функция отрисовки графика
    def graph_do(znach, name_, x_lb='', y_lb=''):
        # # создаём объект точек графика
        # ct = plt.contour(y_ln, z_ln, znach, alpha=0.95, colors='black', linestyles='dotted', levels=5)
        # # создаём линии уровней из объекта точек           
        # plt.clabel(ct, fontsize=10)
        # отрисовка        
        plt.imshow(znach, extent=[Ymin, Ymax, Zmax, Zmin],  cmap=cmap,  alpha=0.95, norm=colors.LogNorm())
        # раскраска    
        plt.colorbar()

        # очертания кабины
        fr_kab_lines()
        # название осей 
        plt.xlabel(x_lb)
        plt.ylabel(y_lb)

        plt.title(name_) # подпись названия

    # отрисовка по очереди магнитного, электрического и энергии
    #TODO только энергия
    global gph_num
    gph_num += 1
    plt.figure(gph_num)
    name = 'Кабина вид спереди (c экраном)'
    plt.title(name)
    plt.subplot(1, 3, 1)
    graph_do(magnetic, 'Магнитное', x_lb='Ось x, метры', y_lb='Ось y, метры')
    kab_lines_front()
    plt.subplot(1, 3, 2)
    graph_do(electric, 'Электрическое', x_lb='Ось x, метры')
    kab_lines_front()
    plt.subplot(1, 3, 3)
    graph_do(energy, 'Энергия', x_lb='Ось x, метры', )
    kab_lines_front()
    plt.suptitle(name)
    show(name)

    # отрисовка гармоник на одном графике
    #TODO только энергия
    gph_num += 1
    plt.figure(gph_num)
    name = 'Гармоники магнитного поля вид спереди'
    plt.title(name)
    i = 0
    chel_harm_h = []
    # для каждой гармоники формируем массив точек на отрисовку + считаем воздействие в положении человека
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

    # то же самое для электрического поля
    gph_num += 1
    plt.figure(gph_num)
    name = 'Гармоники электрического поля вид спереди'
    plt.title(name)
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
#cont_f_up = visual_up()

# todo

print('\nВид спереди')
cont_f_front = visual_front()
plt.show()
exit()

print('\nКабина электровоза:')
print('\nВид сверху')
visual_up_locomotive(cont_f_up)

print('\nВид спереди')
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
print('\nПерменное поле с экраном %.4f' % ekran_per)
Dco = ekran_per * ti * S * p
Dpo = Dco / b
print('Удельная суточная доза поглощённой энергии: %.4f' % Dpo)

plt.show()
