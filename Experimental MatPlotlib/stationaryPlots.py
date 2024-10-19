import matplotlib.pyplot as plt
import numpy as np

# fig, ax = plt.subplots()
# ax.plot([1, 2, 3, 4], [1, 2, 3, 5])
# plt.show()

# b = np.matrix([[1, 2], [3, 4]])
# print(np.asarray(b))

#----#
# data = {'a': np.arange(50),
#         'c': np.random.randint(0, 50, 50),
#         'd': np.random.randn(50)}
# # print(data['a'])
# data['b'] = data['a'] + 10 * np.random.randn(50)
# data['d'] = np.abs(data['d']) * 100
# # print(data['d'])

# fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
# ax.scatter('a', 'b', c='c', s=50, data=data)
# ax.set_xlabel('entry a')
# ax.set_ylabel('entry b')

# plt.show()
# #---#

# x = np.linspace(0, 2, 100)

# b = {'x':np.linspace(0, 2, 100),
#      'x2':np.linspace(0, 2, 100)**4}

# fig, ax = plt.subplots(figsize=(5, 2.7), layout='tight')
# ax.plot(x, x, label='linear')
# ax.plot('x', 'x2', label='quadratic', data=b)
# ax.plot(x, x**3, label='cubic')
# ax.set_xlabel('x label')
# ax.set_ylabel('y label')
# ax.set_title('Simple Plot')
# ax.legend()

# plt.show()

#----#

# def my_plotter(ax, data1, data2, param_dict):
#     out = ax.scatter(data1, data2, **param_dict)
#     return out

# data1, data2, data3, data4 = np.random.randn(4, 100)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.7))
# my_plotter(ax1, data1, data2, {'facecolor':'C0', 'edgecolor':'k'})
# my_plotter(ax2, data3, data4, {'marker': 'o'})
# plt.show()

#---#

# mu, sigma = 115, 15
# x = mu + sigma * np.random.randn(10000)
# fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')

# n, bins, patches = ax.hist(x, 50, density=True, facecolor='C0', alpha=0.75)
# ax.set_xlabel('Length [cm]')
# ax.set_ylabel('Probability')
# ax.set_title('Aardvark lengths\n (not really)')
# ax.text(75, .025, r'$\mu=115,\ \sigma=15$')
# ax.axis([55, 175, 0, 0.03])
# ax.grid(True)

# plt.show()

#---#

# fig, ax = plt.subplots(figsize=(5, 2.7))
# t = np.arange(0.0, 5.0, 0.01)
# s = np.cos(2 * np.pi * t)
# line, = ax.plot(t, s, lw=2)

# ax.annotate('local max', xy=(1, 1), xytext=(4, 1.5), arrowprops=dict(facecolor='black', shrink=0.05))
# ax.set_ylim(-2, 2)
# plt.show()

# -- #

# fig, ax = plt.subplots(layout='constrained')
# categories = ['turnips', 'rutabag', 'cucumber', 'pumpkins']
# ax.bar(categories, np.array([1, 2, 3, 4]))

# plt.show()

#--#

from matplotlib.colors import LogNorm

data1, data2, data3, data4 = np.random.randn(4, 100)

X, Y = np.meshgrid(np.linspace(-3, 3, 128), np.linspace(-3, 3, 128))
Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)

fig, axs = plt.subplots(2, 2, layout='constrained')
pc = axs[0, 0].pcolormesh(X, Y, Z, vmin=-1, vmax=1, cmap='RdBu_r')
fig.colorbar(pc, ax=axs[0, 0])
axs[0, 0].set_title('pcolormesh()')

co = axs[0, 1].contourf(X, Y, Z, levels=np.linspace(-1.25, 1.25, 11))
fig.colorbar(co, ax=axs[0, 1])
axs[0, 1].set_title('contourf()')

pc = axs[1, 0].imshow(Z**2 * 100, cmap='plasma', norm=LogNorm(vmin=0.01, vmax=100))
fig.colorbar(pc, ax=axs[1, 0], extend='both')
axs[1, 0].set_title('imshow() with LogNorm()')

pc = axs[1, 1].scatter(data1, data2, c=data3, cmap='RdBu_r')
fig.colorbar(pc, ax=axs[1, 1], extend='both')
axs[1, 1].set_title('scatter()')

plt.show()


#-----