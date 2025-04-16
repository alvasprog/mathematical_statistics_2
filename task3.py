import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1. Генерация данных.
data1 = np.random.normal(loc=-2, scale=1, size=300)
data2 = np.random.normal(loc=2, scale=0.5, size=200)
all_data = np.concatenate((data1, data2))
np.savetxt('data.csv', all_data, delimiter=',')


# 2. Ядерная оценка плотности.
def gaussian_kernel(u):
    return np.exp(-u ** 2 / 2) / np.sqrt(2 * np.pi)


def kde(x, data, h_window=0.5):
    n = len(data)
    return np.sum([gaussian_kernel((x - xi) / h_window) for xi in data]) / (n * h_window)


# 3. Истинная плотность смеси распределений.
def true_pdf(x):
    return (0.6 * norm.pdf(x, loc=-2, scale=1) +
            0.4 * norm.pdf(x, loc=2, scale=0.5))


# 4. Построение графиков.
x_vals = np.linspace(-10, 10, 1000)
kde_vals = np.array([kde(x, all_data, h_window=0.5) for x in x_vals])
true_vals = true_pdf(x_vals)

plt.figure(figsize=(12, 6))

# Гистограмма и истинная плотность.
plt.subplot(1, 2, 1)
plt.hist(all_data, bins=50, density=True, alpha=0.5, label='Гистограмма.')
plt.plot(x_vals, true_vals, 'r-', lw=2, label='Истинная плотность.')
plt.title('Гистограмма и истинная плотность.')
plt.xlabel('x')
plt.ylabel('Плотность')
plt.legend()

# Сравнение KDE и истинной плотности.
plt.subplot(1, 2, 2)
plt.plot(x_vals, true_vals, 'r-', lw=2, label='Истинная плотность.')
plt.plot(x_vals, kde_vals, 'b--', lw=2, label='Ядерная оценка (h=0.5).')
plt.title('Сравнение KDE и истинной плотности.')
plt.xlabel('x')
plt.ylabel('Плотность')
plt.legend()

plt.tight_layout()
plt.show()

# 5. Эксперименты с разной шириной окна.
plt.figure(figsize=(12, 6))
for i, h in enumerate([0.05, 0.5, 1], 1):
    kde_vals = np.array([kde(x, all_data, h_window=h) for x in x_vals])
    plt.subplot(1, 3, i)
    plt.plot(x_vals, true_vals, 'r-', label='Истинная плотность.')
    plt.plot(x_vals, kde_vals, 'b--', label=f'KDE (h={h}).')
    plt.title(f'Ширина окна h={h}.')
    plt.legend()

plt.tight_layout()
plt.show()
