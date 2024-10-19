import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define an Animation class
class SineWaveAnimation:
    def __init__(self, xlim=(0, 2 * np.pi), ylim=(-1, 1), interval=50):
        # Set up the figure and axis
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        
        # Initialize data for the plot
        self.xdata, self.ydata = [], []
        self.ln, = self.ax.plot([], [], 'r-', animated=True)  # Empty line
        
        # Create the animation object
        self.ani = FuncAnimation(self.fig, self.update, frames=np.linspace(0, 2 * np.pi, 100),
                                 init_func=self.init, interval=interval, blit=True)
    
    def init(self):
        """Initialize the background of the plot."""
        self.ln.set_data([], [])
        return self.ln,
    
    def update(self, frame):
        """Update the data for each frame."""
        self.xdata.append(frame)
        self.ydata.append(np.sin(frame))
        self.ln.set_data(self.xdata, self.ydata)
        return self.ln,
    
    def show(self):
        """Display the plot."""
        plt.show()

# Create an instance of the animation class and run it
if __name__ == "__main__":
    animation = SineWaveAnimation()
    animation.show()