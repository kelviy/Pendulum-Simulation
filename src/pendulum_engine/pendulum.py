import numpy as np

class Pendulum:
    # Constants
    g = 9.81        # Acceleration due to gravity (m/s^2)
    b = 0.5        # Damping coefficient
    M = 1           # Mass of Pendulum Ball

    def __init__(self, screen_width, screen_length, length = 1, theta = 0, omega = 0):
        self.L = length                         # Length of the pendulum (m)
        self.theta = theta                      # Initial angle (radians)
        self.omega = omega                      # Initial angular velocity (rad/s)

        self.pivot_x = screen_width/2           # Initial x position of the pivot
        self.pivot_y = screen_length/2 - 100    # Y position of the pivot (fixed)

    # Equations of motion for the pendulum - returns the angular acceleration
    def derivatives(self, x_acc):
        theta_acc = (3/2) * ((self.g * np.sin(-self.theta) - x_acc * np.cos(-self.theta))/self.L) - self.b * self.omega
        return theta_acc

    # Update Physics Parameters of Pendulum
    def update_physics(self, x_acc, dt):
        self.omega += self.derivatives(x_acc) * dt 
        self.theta += self.omega * dt

    # Returns the Pivot's Location
    def get_pivot_loc(self):
        return (self.pivot_x, self.pivot_y)
    
    # Returns the Pendulum Ball's Centre Location
    def get_pendulum_loc(self):
        pendulum_x = self.pivot_x + int(self.L * 100 * np.sin(self.theta))
        pendulum_y = self.pivot_y + int(self.L * 100 * np.cos(self.theta))
        return (pendulum_x, pendulum_y)

        