import random as r
import numpy as np
import math
from functools import reduce
from itertools import compress
from scipy.stats import f, t


x1min, x2min, x3min = -15, 30, 30
x1max, x2max, x3max = 30, 80, 35

x_min = (x1min + x2min + x3min) / 3   #mean of x1min, x2min, x3min
x_max = (x1max + x2max + x3max) / 3


m = 3
N = 8
x0 = [1 for i in range(N)]

#                     x1  x2   x3  x12 x13 x23 x123
norm_x_table = [[-1, -1, -1, +1, +1, +1, -1],
                      [-1, +1, +1, -1, -1, +1, -1],
                      [+1, -1, +1, -1, +1, -1, -1],
                      [+1, +1, -1, +1, -1, -1, -1],

                      [-1, -1, +1, +1, -1, -1, +1],
                      [-1, +1, -1, -1, +1, -1, +1],
                      [+1, -1, -1, -1, -1, +1, +1],
                      [+1, +1, +1, +1, +1, +1, +1]]

#               1     2      3       12            13           23             123
x_table =  [[x1min, x2min, x3min, x1min*x2min, x1min*x3min, x2min*x3min, x1min*x2min*x3min],
            [x1min, x2max, x3max, x1min*x2max, x1min*x3max, x2max*x3max, x1min*x2max*x3max],
            [x1max, x2min, x3max, x1max*x2min, x1max*x3max, x2min*x3max, x1max*x2min*x3max],
            [x1max, x2max, x3min, x1max*x2max, x1max*x3min, x2max*x3min, x1max*x2max*x3min],

            [x1min, x2min, x3max, x1min*x2min, x1min*x3max, x2min*x3max, x1min*x2min*x3max],
            [x1min, x2max, x3min, x1min*x2max, x1min*x3min, x2max*x3min, x1min*x2max*x3min],
            [x1max, x2min, x3min, x1max*x2min, x1max*x3min, x2min*x3min, x1max*x2min*x3min],
            [x1max, x2max, x3max, x1max*x2max, x1max*x3max, x2max*x3max, x1max*x2max*x3max]]

y_min = round(200 + x_min)
y_max = round(200 + x_max)

y_arr = [[r.randint(y_min, y_max) for j in range(m)] for i in range(N)]  # i rows and j columns
print(y_arr)

#arrays with x1(i), x2(i),x3(i)
x1i = np.array([x_table[i][0] for i in range(8)])
x2i = np.array([x_table[i][1] for i in range(8)])
x3i = np.array([x_table[i][2] for i in range(8)])
yi = np.array([np.average(i) for i in y_arr]) #average for each i row in y_arr


def m_ij(*arrays):
    return np.average(reduce(lambda el_1, el_2: el_1+el_2, arrays)) #reduce: sums all el in given arrays


coefs = [[N,m_ij(x1i),    m_ij(x2i),    m_ij(x3i),    m_ij(x1i*x2i),    m_ij(x1i*x3i),    m_ij(x2i*x3i),    m_ij(x1i*x2i*x3i)],
          [m_ij(x1i), m_ij(x1i**2), m_ij(x1i*x2i), m_ij(x1i*x3i), m_ij(x1i**2*x2i), m_ij(x1i**2*x3i), m_ij(x1i*x2i*x3i), m_ij(x1i**2*x2i*x3i)],
          [m_ij(x2i), m_ij(x1i*x2i), m_ij(x2i**2), m_ij(x2i*x3i), m_ij(x1i*x2i**2), m_ij(x1i*x2i*x3i), m_ij(x2i**2*x3i), m_ij(x1i*x2i**2*x3i)],
          [m_ij(x3i), m_ij(x1i*x3i), m_ij(x2i*x3i), m_ij(x3i**2), m_ij(x1i*x2i*x3i), m_ij(x1i*x3i**2), m_ij(x2i*x3i**2), m_ij(x1i*x2i*x3i**2)],

          [m_ij(x1i*x2i), m_ij(x1i**2*x2i), m_ij(x1i*x2i**2), m_ij(x1i*x2i*x3i), m_ij(x1i**2*x2i**2), m_ij(x1i**2*x2i*x3i), m_ij(x1i*x2i**2*x3i), m_ij(x1i**2*x2i**2*x3i)],
          [m_ij(x1i*x3i), m_ij(x1i**2*x3i), m_ij(x1i*x2i*x3i), m_ij(x1i*x3i**2), m_ij(x1i**2*x2i*x3i), m_ij(x1i**2*x3i**2), m_ij(x1i*x2i*x3i**2), m_ij(x1i**2*x2i*x3i**2)],
          [m_ij(x2i*x3i), m_ij(x1i*x2i*x3i), m_ij(x2i**2*x3i), m_ij(x2i*x3i**2), m_ij(x1i*x2i**2*x3i), m_ij(x1i*x2i*x3i**2), m_ij(x2i**2*x3i**2), m_ij(x1i*x2i**2*x3i**2)],

          [m_ij(x1i*x2i*x3i), m_ij(x1i**2*x2i*x3i), m_ij(x1i*x2i**2*x3i), m_ij(x1i*x2i*x3i**2), m_ij(x1i**2*x2i**2*x3i), m_ij(x1i**2*x2i*x3i**2), m_ij(x1i*x2i**2*x3i**2), m_ij(x1i**2*x2i**2*x3i**2)]]

free_vals = [m_ij(yi), m_ij(yi*x1i), m_ij(yi*x2i), m_ij(yi*x3i), m_ij(yi*x1i*x2i), m_ij(yi*x1i*x3i), m_ij(yi*x2i*x3i), m_ij(yi*x1i*x2i*x3i)]

#solution of system of equations(array)
b_i = np.linalg.solve(coefs, free_vals)

#just arrays from table
nat_x1 = np.array([norm_x_table[i][0] for i in range(8)])
nat_x2 = np.array([norm_x_table[i][1] for i in range(8)])
nat_x3 = np.array([norm_x_table[i][2] for i in range(8)])

norm_b_i = [m_ij(yi*1), m_ij(yi*nat_x1), m_ij(yi*nat_x2), m_ij(yi*nat_x3),
           m_ij(yi*nat_x1*nat_x2), m_ij(yi*nat_x1*nat_x3), m_ij(yi*nat_x2*nat_x3),
           m_ij(yi*nat_x1*nat_x2*nat_x3)]

#main functions

def get_cochran(f1, f2, q):
    partResult1 = q / f2
    params = [partResult1, f1, (f2 - 1) * f1]
    fisher = f.isf(*params)
    result = fisher/(fisher + (f2 - 1))
    return result.__round__(3)


def theor_y(x_table, b_coef, importance):
    x_table = [list(compress(row, importance)) for row in x_table] #update: if importance 0 - get rid of x(ij)
    b_coef = list(compress(b_coef, importance)) #update: if importance 0 - get rid of b
    y_vals = np.array([sum(map(lambda x, b: x*b, row, b_coef)) for row in x_table])
    return y_vals

def student_criteria(m, N, y_table, norm_x_table):
    print("\nЗа критерієм Стьюдента: m = {}, N = {} ".format(m, N))
    avg_variation = np.average(list(map(np.var, y_table)))   #var = mean(abs(y - y.mean())**2) in numpy
    y_avrg = np.array(list(map(np.average, y_table)))
    variation_beta_s = avg_variation/N/m
    deviation_beta_s = math.sqrt(variation_beta_s)
    x_i = np.array([[el[i] for el in norm_x_table] for i in range(len(norm_x_table))])
    coef_beta_s = np.array([round(np.average(y_avrg*x_i[i]), 3) for i in range(len(x_i))])
    print("Оцінки коефіцієнтів β(s): " + ", ".join(list(map(str, coef_beta_s))))
    t_i = np.array([abs(coef_beta_s[i])/deviation_beta_s for i in range(len(coef_beta_s))])
    print("Коефіцієнти t:         " + ", ".join(list(map(lambda i: "{:.2f}".format(i), t_i))))
    f3 = (m-1)*N
    q = 0.05
    t = get_student(f3, q)
    importance = [True if el > t else False for el in list(t_i)]
    print("f3 = {}; q = {}; tтабл = {}".format(f3, q, t))
    beta_i = ["β0", "β1", "β2", "β3", "β12", "β13", "β23", "β123"]
    updated_importance = [" - значимий" if i else " - незначимий" for i in importance]
    to_print = map(lambda x: x[0] + " " + x[1], zip(beta_i, updated_importance))
    x_i_names = list(compress(["", " x1", " x2", " x3", " x12", " x13", " x23", " x123"], importance)) #if importance 0 - get rid of it
    betas_to_print = list(compress(coef_beta_s, importance))
    print(*to_print, sep="; ")
    equation = " ".join(["".join(i) for i in zip(list(map(lambda x: "{:+.2f}".format(x), betas_to_print)), x_i_names)])
    print("Рівняння регресії без незначимих членів: y = " + equation)
    return importance

def get_student(f3, q):
    return (abs(t.ppf(q/2, f3))).__round__(3)

def get_fisher(f3, f4, q):
    return (abs(f.isf(q, f4, f3))).__round__(3)

def cochran_criteria(m, N, y_table):
    print("За критерієм Кохрена: m = {}, N = {} ".format(m, N))
    y_variations = [np.var(i) for i in y_table]
    max_y_variation = max(y_variations)
    gp = max_y_variation/sum(y_variations)
    f1 = m - 1
    f2 = N
    p = 0.95
    q = 1-p
    gt = get_cochran(f1, f2, q)
    print("Gp = {}; Gt = {}; f1 = {}; f2 = {}; q = {:.2f}".format(gp, gt, f1, f2, q))
    if gp < gt:
        print("Gp < Gt => дисперсії рівномірні")
        return True
    else:
        print("Gp > Gt => дисперсії нерівномірні")
        return False


check = cochran_criteria(m, 4, y_arr)
while not check:
    m += 1
    y_table = [[r.randint(y_min, y_max) for column in range(m)] for row in range(N)]
print("Матриця планування:")
labels = "   x1    x2    x3    x12    x13    x23   x123     y1    y2    y3"
rows_table = [list(x_table[i]) + list(y_arr[i]) for i in range(N)]
print(labels)
print("\n".join([" ".join(map(lambda j: "{:<6}".format(j), rows_table[i])) for i in range(len(rows_table))]),"\n")

norm_x_table_with_x0 = [[+1]+row for row in norm_x_table]
importance = student_criteria(m, N, y_arr, norm_x_table_with_x0) #shows should each b(ij)*x(i) be in our main equation

def fisher_criteria(m, N, d, nat_x_table, y_table, b_coefficients, importance):
    print("\nЗа критерієм Фішера: m = {}, N = {} ".format(m, N))
    f3 = (m - 1) * N
    f4 = N - d
    q = 0.05
    theoret_y = theor_y(nat_x_table, b_coefficients, importance)
    theor_values_to_print = list(zip(map(lambda x: "x1 = {0[0]}, x2 = {0[1]}, x3 = {0[2]}".format(x),nat_x_table),theoret_y))
    print("Теоретичні y:")
    print("\n".join(["{val[0]}: y = {val[1]}".format(val = el) for el in theor_values_to_print]))
    y_averages = np.array(list(map(np.average, y_table)))
    s_ad = m/(N-d)*(sum((theoret_y-y_averages)**2))
    y_variations = np.array(list(map(np.var, y_table)))
    s_v = np.average(y_variations)
    f_p = round(float(s_ad/s_v), 3)
    f_t = get_fisher(f3, f4, q)
    print("Fp = {}, Ft = {}".format(f_p, f_t))
    print("Fp < Ft --> модель адекватна" if f_p < f_t else "Fp > Ft --> неадекватна")
    return True if f_p < f_t else False

fisher_criteria(m, N, 1, x_table, y_arr, b_i, importance)










