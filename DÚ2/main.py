from bs4 import BeautifulSoup
import requests
import math
import numpy as np
import matplotlib.pyplot as plt

response = requests.get('https://kf.fjfi.cvut.cz/myska/SZD/szd1/radiator/radiator.html').content

soup = BeautifulSoup(response, 'html.parser')
#print(soup.find("article").text)

#rozměry mojí místnosti:
x_min = 0
x_max = 50
y_min = 1
y_max = 10

#můj zářič je v pozici:
y_0_real = 3
x_0_real = 17


#spočte_mi x_i z úhlu radiace
def find_x_i(y_0, x_0, x_min, x_max, y_min, y_max, alpha):
    print(y_0)
    x_i = y_0 * math.tan(alpha) + x_0
    if x_min <= x_i <= x_max:
        return x_i


#spočte hodnotu likelihoodu
def calculate_L(y_0, x_0,x_is):
    lnL = 0
    for x_i in x_is:
        lnL += 1 / (2 * math.pi) * y_0 / ((x_i - x_0)**2 + y_0**2)
    #L = math.exp(lnL)
    return lnL

#generuje alfy od 0 do 2 pi
alphas = np.linspace(0, 2 * np.pi, num=1000)
x_is = []
for alpha in alphas:
    x_i = find_x_i(y_0_real, x_0_real, x_min, x_max, y_min, y_max, alpha)
    if x_i is not None:
        x_is.append(x_i)

max_L = -float('inf')
best_x_0, best_y_0 = None, None



x_0s = np.linspace(x_min, x_max, num=100)
y_0s = np.linspace(y_min, y_max, num=100)
likelihood_grid = np.zeros((len(y_0s), len(x_0s)))

for i, y_0 in enumerate(y_0s):
    #print(y_0)
    #print(f"i = {i}")
    for j, x_0 in enumerate(x_0s):
        #print(x_0, f"j = {j}")
        L = calculate_L(y_0, x_0, x_is)
        likelihood_grid[i, j] = L

        if L > max_L:
            max_L = L
            best_x_0, best_y_0 = x_0, y_0


#print(likelihood_grid)
print(f"x_0 = {best_x_0}, y_0 = {best_y_0} with likelihood L = {max_L}")

plt.figure(figsize=(10, 6))
plt.imshow(
    likelihood_grid,
    extent=[x_min, x_max, y_min, y_max],
    origin='lower',
    aspect='auto',
    cmap='viridis'
)


plt.colorbar(label="Likelihood (L)")
plt.xlabel("x_0 (Position along x-axis)")
plt.ylabel("y_0 (Position along y-axis)")
plt.title("Heatmap of Likelihood Function")
plt.scatter([x_0_real], [y_0_real], color="red", label="True Position", marker='x')  # Mark true position
plt.legend()


plt.show()