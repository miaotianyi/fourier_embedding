import math
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns   # for prettier plots

import torch
from layers.sinusoidal_encoding import SinusoidalEncoding


def plot_sinusoidal_encoding():
    n_samples = 100
    n_features = 128
    # embed = SinusoidalEncoding(n_features, period_range=(0.01, 3), progression="geometric")
    embed = SinusoidalEncoding(n_features, period_range=(1, 10000), progression="geometric")
    # x = np.linspace(-5, 5, n_samples)
    x = np.linspace(0, 1000, n_samples)
    x_tensor = torch.from_numpy(x)
    # print(x_tensor.shape)
    with torch.inference_mode():
        z = embed(x_tensor).numpy()

    # prepare plotting
    vlag = sns.color_palette("vlag", as_cmap=True)  # better cmap using seaborn
    # z.T [embedding_dim, in_dim] has x=-3 -> 3 from left to right, period min->max from up to down
    plt.matshow(z.T, cmap=vlag, interpolation="none", aspect="auto")
    plt.xlabel("input feature value")
    n_x_ticks = 10
    x_ticks = np.linspace(0, len(x) - 1, n_x_ticks)
    plt.xticks(x_ticks, [f"{x[int(i)]:.2f}" for i in x_ticks])

    plt.ylabel("linear periods")
    n_y_ticks_half = 6
    y_ticks = np.linspace(0, embed.embedding_dim // 2 - 1, n_y_ticks_half)
    y_labels = [f"{math.tau / embed.w[int(i)]:.2f}" for i in y_ticks]
    y_ticks = np.concatenate([y_ticks, y_ticks + embed.embedding_dim // 2])
    y_labels = np.concatenate([y_labels, y_labels])
    plt.yticks(y_ticks, y_labels)

    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    plot_sinusoidal_encoding()
