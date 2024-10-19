# import matplotlib.pyplot as plt
# import time

# fig, ax = plt.subplots()
# # plt.ion()  # Turn on interactive mode
# # plt.show()

# # Initial plot
# ln, = ax.plot(range(5))
# fig.canvas.draw()  # Redraw the figure to display the initial plot

# plt.pause(1)  # Wait for 10 seconds

# # Change color of the line
# ln.set_color('orange')
# fig.canvas.draw()  # Redraw the figure to update the color
# plt.pause(10)  # Keep the plot open for another 10 seconds


#----#

# import matplotlib.pyplot as plt
# import numpy as np
  
# plt.plot([1.4, 2.5])
# plt.title(" Sample interactive plot")
  
# axes = plt.gca()
# axes.plot([3.1, 2.2])
# plt.show()

#---#
import matplotlib.pyplot as plt
import numpy as np
plt.ion()  # Turn on interactive mode

x = np.linspace(0, 10, 100)
y = np.sin(x)

for i in range(10):
    plt.gcf().canvas.flush_events()
    plt.plot(x, y * (i + 1))
    plt.draw()
    plt.pause(0.5)  # Pause to update the plot

plt.show()  # Display the final plot

#---#
# import numpy as np
# import time
# from matplotlib import pyplot as plt

# def live_update_demo():

#     plt.subplot(2, 1, 1)
#     h1 = plt.imshow(np.random.randn(30, 30))
#     redraw_figure()
#     plt.subplot(2, 1, 2)
#     h2, = plt.plot(np.random.randn(50))
#     redraw_figure()

#     t_start = time.time()
#     for i in range(1000):
#         h1.set_data(np.random.randn(30, 30))
#         redraw_figure()
#         h2.set_ydata(np.random.randn(50))
#         redraw_figure()
#         # print 'Mean Frame Rate: %.3gFPS' % ((i+1) / (time.time() - t_start))

# def redraw_figure():
#     plt.draw()
#     plt.pause(0.00001)

# live_update_demo()
