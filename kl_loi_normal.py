import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def hist_par_colonnes(tab,xn=1,yn=2):
    for i in range(tab.shape[1]):
        d = np.abs(np.max(tab[:, i])-np.min(tab[:, i])).astype(int)
        fig, ax = plt.subplots()
        occurrences, bins, _ = ax.hist(tab[:, i], bins=2*d+1)
        ax.set_xlabel('Valeur')
        ax.set_ylabel('Nombre d\'occurrences')
        ax.set_title('Histogramme')
        ax.grid(axis='y')
        ax.yaxis.set_major_locator(plt.MultipleLocator(yn))
        ax.xaxis.set_major_locator(plt.MultipleLocator(xn))
        valeur_max = bins[np.argmax(occurrences)]
        print(f'La valeur correspondant à l\'occurrence maximale de la colonne {i + 1} est {valeur_max}')
        plt.show()

# #########################################################################################
def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def normal_approximation(data):
    mean, std = norm.fit(data)
    d = np.abs(np.max(data) - np.min(data)).astype(int)
    fig, ax = plt.subplots()
    x = np.linspace(data.min(), data.max(), len(data))
    ax.hist(data, density=True, bins=2 * d + 1)
    ax.hist(data, bins='auto', density=True, alpha=0.7)
    ax.plot(x, norm.pdf(x, mean, std), 'r-', lw=1, label='normal pdf')
    ax.axvline(x=round(mean), color='k', linestyle='--', label=f'mean ={mean:.2f} soit {round(mean):.0f}')
    # ax.yaxis.set_major_locator(plt.MultipleLocator(2))
    # ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.legend(loc='best')
    ax.grid(axis='y')
    plt.show()
    return mean, std


def normal_parameters(matrix):
    n, m = matrix.shape
    parameters = []
    for j in range(m):
        data = matrix[:,j]
        mean, std = normal_approximation(data)
        parameters.append((mean, std))
    return parameters

def kl_distrib(matrix):
    parameters = normal_parameters(matrix)
    for j in range(m):
        p = norm.pdf(matrix[:,j], *parameters[j])
        q = norm.pdf(matrix[:,j], matrix[:,j].mean(), matrix[:,j].std())
        kl = kl_divergence(p, q)
        print(f"Column {j}: Mean = {parameters[j][0]}, Std = {parameters[j][1]}, KL-Divergence = {kl}")

# #########################################################################################
# Example usage:
n, m = 30000, 5
def generate_matrix(n):
    matrix = np.zeros((n, m), dtype=int)
    for i in range(n):
        row = np.random.choice(49, size=m, replace=False) + 1
        matrix[i,:] = row
    return matrix

matrix = np.apply_along_axis(np.sort, axis=1, arr=generate_matrix(n))
# matrix = generate_matrix(n)
kl_distrib(matrix)

# hist_par_colonnes(matrix,2,1)