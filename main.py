import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math

# Параметры системы
m = 0.12  # масса груза в кг
k = 1.1844  # жесткость пружины в Н/м (используем точное значение)
c = 1.78e-2  # коэффициент силы трения в кг/с

# Уравнение движения: m*a + c*v + k*x = 0
def spring_system(t, y):
    x, v = y
    dxdt = v
    dvdt = -(k/m)*x - (c/m)*v
    return [dxdt, dvdt]

# Начальные условия
x0 = 0.1  # начальное положение (отклонение от равновесия, например, 10 см)
v0 = 0.0  # начальная скорость равна нулю (маятник отпущен)
y0 = [x0, v0]

# Интервал времени для решения
t_span = (0, 40)  # от 0 до 40 секунд
t_eval = np.linspace(0, 40, 1000)  # точки для вывода

# Решение уравнения с помощью solve_ivp
solution = solve_ivp(spring_system, t_span, y0, t_eval=t_eval, method='RK45')

# Построение графика положения маятника
plt.figure(figsize=(10, 6))
plt.plot(solution.t, solution.y[0], 'b-', linewidth=2)
plt.title('График положения маятника x(t)')
plt.xlabel('Время, с')
plt.ylabel('Положение x, м')
plt.grid(True)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)  # линия равновесия
plt.savefig('pendulum_position.png')
plt.show()

# Вывод параметров системы
omega = math.sqrt(k/m)
zeta = c/(2*math.sqrt(k*m))
print(f"Жесткость пружины k = {k:.4f} Н/м")
print(f"Частота колебаний ω = {omega:.4f} рад/с")
print(f"Коэффициент затухания ζ = {zeta:.4f}")