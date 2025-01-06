import matplotlib.pyplot as plt
import numpy as np


def gini(p: float) -> float:
    return 2 * p * (1 - p)


def entropy(p: float) -> float:
    return - p * np.log2(p) - (1 - p) * np.log2(1 - p)


def classification_error(p: float) -> float:
    return 1 - np.max([p, 1 - p])


def main():
    x = np.arange(0.0, 1.0, 0.01)
    entropy_values = [entropy(p) if p != 0.0 else None for p in x]
    scaled_entropy = [e * 0.5 if e else None for e in entropy_values]
    classification_errors = [classification_error(p) for p in x]
    gini_values = [gini(p) for p in x]
    plt.figure()
    ax = plt.subplot(111)
    for value, label \
            in zip((entropy_values, scaled_entropy, gini_values, classification_errors,),
                   ('Entropy', 'Scaled entropy', "Gini's measure", 'Classification error',)):
        ax.plot(x, value, label=label, lw=2)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5, fancybox=True, shadow=False)
    ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
    ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
    plt.ylim([0, 1.1])
    plt.xlabel('p')
    plt.ylabel('Impurity measure')
    plt.show()
