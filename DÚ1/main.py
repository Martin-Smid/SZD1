import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Gaus
def gaussian(x, a, b, c):
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2))
#ukradeno ze stránky
def find_bin_centers(edges):
    return (edges[1:] + edges[:-1]) / 2


def main():
    chosenBins = np.array([11, 12, 13, 14, 15, 16])
    numOfBins = chosenBins.size  # Number of raa bins to fit
    MCsize = 100000  # Number of MC simulations

    # Arrays to hold data
    ratios, gaus1, gaus2, centers, values, edges = [[] for _ in range(6)]

    # Generování Gaussian-distributed dat
    for i in chosenBins:
        # Create data for gaus1 and gaus2 from the given bin content
        gaus1_data = np.random.normal(hAA_content[i], hAA_errors[i], MCsize)
        gaus2_data = np.random.normal(hpp_content[i], hpp_errors[i], MCsize)

        gaus1.append(gaus1_data)
        gaus2.append(gaus2_data)

        # počírání podílu z gaussů
        ratio_data = gaus1_data / gaus2_data
        ratios.append(ratio_data)

        # Calculate histogram values and edges for plotting
        hist, bin_edges = np.histogram(ratio_data, bins=100, range=(0, 10), density=True)
        values.append(hist)
        edges.append(bin_edges)
        centers.append(find_bin_centers(bin_edges))

    # Set font size for the figures
    plt.rcParams['font.size'] = 14

    # Create figure and axes
    fig, ax = plt.subplots(numOfBins, 3,
                           figsize=[15, 4 * numOfBins])  # Prepare figure and axes (numOfBins rows, 3 columns)

    # ukládání odchylek
    sigma_left = []
    sigma_right = []

    # loop přes biny
    for i in range(numOfBins):
        for a, h in zip(ax[i], [gaus1[i], gaus2[i], ratios[i]]):
            ax[i][0].hist(gaus1[i], bins=50, density=True, alpha=0.6, label=f'bin {i}')
            ax[i][1].hist(gaus2[i], bins=50, density=True, alpha=0.6, label=f'bin {i}')
            ax[i][2].hist(ratios[i], bins=50, density=True, alpha=0.6, label=f'bin {i}')


        hist_values = values[i]
        hist_centers = centers[i]

        # fitnutí celého histogramz
        try:
            popt_total, _ = curve_fit(gaussian, hist_centers, hist_values, p0=[np.max(hist_values), hist_centers[np.argmax(hist_values)], 1], maxfev=10000)
            fit_total = gaussian(hist_centers, *popt_total)
        except RuntimeError:
            print(f"Warning: Total fit failed for bin {i}")
            fit_total = np.zeros_like(hist_centers)

        # fit z leva
        try:
            left_mask = hist_centers <= popt_total[1]
            popt_left, _ = curve_fit(gaussian, hist_centers[left_mask], hist_values[left_mask], p0=popt_total, maxfev=10000)
            fit_left = gaussian(hist_centers[left_mask], *popt_left)
            sigma_left.append(popt_left[2])
        except RuntimeError: #error handeling od chatgpt, protože to házelo stejný error, kvůli dělení nulou
            print(f"Warning: Left-side fit failed for bin {i}")
            fit_left = np.zeros_like(hist_centers[left_mask])
            sigma_left.append(0)

        # fit z prava
        try:
            right_mask = hist_centers >= popt_total[1]
            popt_right, _ = curve_fit(gaussian, hist_centers[right_mask], hist_values[right_mask], p0=popt_total, maxfev=10000)
            fit_right = gaussian(hist_centers[right_mask], *popt_right)
            sigma_right.append(popt_right[2])
        except RuntimeError:
            print(f"Warning: Right-side fit failed for bin {i}")
            fit_right = np.zeros_like(hist_centers[right_mask])
            sigma_right.append(0)

        # Plot the fits
        ax[i][2].plot(hist_centers, fit_total, label=f'Total $\sigma = {np.round(popt_total[2], 3)}$')
        ax[i][2].plot(hist_centers[left_mask], fit_left, label=f'Left $\sigma = {np.round(sigma_left[-1], 3)}$')
        ax[i][2].plot(hist_centers[right_mask], fit_right, label=f'Right $\sigma = {np.round(sigma_right[-1], 3)}$')

        ax[i][2].legend(loc='upper right')

    # Calculate Raa and its asymmetric errors
    Raa = np.array(hAA_content) / np.array(hpp_content)
    sigma_left = np.array(sigma_left)
    sigma_right = np.array(sigma_right)

    # Plot Raa with asymmetric error bars
    fig2, ax2 = plt.subplots(figsize=[8, 6])
    ax2.errorbar([xAxis1[i] * 0.5 + xAxis1[i + 1] * 0.5 for i in chosenBins],
                 [Raa[i] for i in chosenBins],
                 yerr=[sigma_left, sigma_right], fmt='o', capsize=4, color='black', label='$R_{AA}$ with asymmetric errors')

    ax2.set_xlabel('$p_T$')
    ax2.set_ylabel('$R_{AA}$')
    ax2.legend()
    fig2.savefig('bins_with_asymmetric_errors.png')

    plt.show()

# 1. Bin edges (xAxis1)
xAxis1 = np.array([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50, 60, 70, 80, 100])

# 2. Bin content for hpp
hpp_content = np.array(
    [1.118516, 0.1920894, 0.007120608, 0.001577695, 0.0004146155, 0.0001296542, 4.644327e-05, 2.057404e-05,
     9.958874e-06, 4.334099e-06, 1.442999e-06, 5.654266e-07, 2.407638e-07, 1.094223e-07, 3.354351e-08, 6.04691e-09,
     7.423793e-10, 3.350337e-11, 0, 0, 0, 0])

# 3. Bin errors for hpp
hpp_errors = np.array(
    [0.0006107765, 0.0003700558, 4.49924e-05, 1.821935e-05, 7.447584e-06, 2.939223e-06, 1.155389e-06, 5.875219e-07,
     1.855936e-07, 1.141886e-07, 3.0114e-08, 5.526975e-09, 2.701332e-09, 1.525211e-09, 6.343152e-10, 9.460454e-11,
     2.709424e-11, 4.271672e-13, 0, 0, 0, 0])

# 4. Bin content for hAA
hAA_content = np.array(
    [0, 0, 0, 0.0004135252, 0.0001128615, 5.329878e-05, 2.740152e-05, 1.417387e-05, 8.002939e-06, 3.765388e-06,
     1.418623e-06, 5.724047e-07, 2.704585e-07, 1.308473e-07, 3.482406e-08, 5.969517e-09, 8.159869e-10, 1.976192e-11, 0,
     0, 0, 0])

# 5. Bin errors for hAA
hAA_errors = np.array(
    [0, 0, 0, 1.06535e-06, 4.957864e-07, 3.151375e-07, 2.136749e-07, 1.451485e-07, 1.005136e-07, 5.988379e-08,
     3.659272e-08, 2.201622e-08, 1.464843e-08, 9.694462e-09, 3.96629e-09, 1.470905e-09, 4.142606e-10, 1.499208e-11, 0,
     0, 0, 0])

if __name__ == "__main__":
    main()
