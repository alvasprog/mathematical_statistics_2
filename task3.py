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
plt.figure(figsize=(10, 6))
plt.hist(all_data, bins=30, density=True, alpha=0.5, label='Гистограмма.')
plt.plot(x_vals, true_pdf(x_vals), 'r-', lw=2, label='Истинная плотность.')
plt.title('Гистограмма и истинная плотность.')
plt.legend()
plt.savefig('histogram_true.png', dpi=300, bbox_inches='tight')
plt.close()

# Сравнение KDE и истинной плотности.
plt.figure(figsize=(10, 6))
plt.plot(x_vals, true_pdf(x_vals), 'r-', lw=2, label='Истинная плотность.')
plt.plot(x_vals, [kde(x, all_data, 0.5) for x in x_vals], 'b--', lw=2, label='KDE (h=0.5).')
plt.title('Сравнение KDE и истинной плотности.')
plt.legend()
plt.savefig('kde_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Эксперименты с разной шириной окна.
plt.figure(figsize=(10, 6))
plt.plot(x_vals, true_pdf(x_vals), 'k-', lw=2, label='Истинная плотность.')
for h, style in [(0.05, '--'), (0.5, '-'), (1, ':')]:
    plt.plot(x_vals, [kde(x, all_data, h) for x in x_vals], 
             label=f'KDE (h={h}.)', linestyle=style)
plt.title('Влияние ширины окна h на оценку плотности.')
plt.legend()
plt.savefig('bandwidth_experiment.png', dpi=300, bbox_inches='tight')
plt.close()
