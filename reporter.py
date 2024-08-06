import matplotlib.pyplot as plt
import numpy as np

class Reporter:
    pivot_position = []
    pivot_velocity = []
    pivot_acceleration = []
    pendulum_angle = []
    pendulum_angular_velocity = []
    pendulum_angular_acceleration = []
    time = []
    score = []

    def __init__(self, dt, environment):
        self.dt = dt
        self.environment = environment

    def record(self):
        m_loc = self.environment.pendulum_obj.pivot_x
        m_velocity = self.environment.x_velocity
        m_acceleration = self.environment.x_acceleration
        
        # p_angle = np.arcsin(np.sin(self.environment.pendulum_obj.theta))
        p_angle = self.environment.pendulum_obj.theta % (np.pi * 2)

        if np.pi - 0.5 <= p_angle <= np.pi + 0.5:
            print("Score!")

        p_angular_velocity = self.environment.pendulum_obj.omega
        p_angular_acceleration = self.environment.pendulum_obj.derivatives(self.environment.x_acceleration)

        print("-------------------")
        print("Time:", self.environment.time)
        print("Pivot Location:", m_loc)
        print("Pivot Velocity:", m_velocity)
        print("Pivot Acceleration:", m_acceleration)
        print("Pendulum Angle:", p_angle)
        print("Pendulum Angular Velocity:", p_angular_velocity)
        print("Pendulum Angular Acceleration: ", p_angular_acceleration)

        self.pivot_position.append(m_loc)
        self.pivot_velocity.append(m_velocity)
        self.pivot_acceleration.append(m_acceleration)
        self.pendulum_angle.append(p_angle)
        self.pendulum_angular_velocity.append(p_angular_velocity)
        self.pendulum_angular_acceleration.append(p_angular_acceleration)
        self.time.append(self.environment.time)
        self.score.append(self.environment.score)

    
    def plot(self):
        fig, axs = plt.subplots(4, 2, layout="constrained")
        fig.suptitle('Pendulum Statistics')

        axs[0, 0].plot(self.time, self.pivot_position)
        axs[0, 0].set_title('Pivot Position')
        axs[1, 0].plot(self.time, self.pivot_velocity)
        axs[1, 0].set_title('Pivot Velocity')
        axs[2, 0].plot(self.time, self.pivot_acceleration)
        axs[2, 0].set_title('Pivot Acceleration')

        axs[3,0].plot(self.time,self.score)
        axs[3,0].set_title('Score vs Time')

        axs[0, 1].plot(self.time, self.pendulum_angle)
        axs[0, 1].set_title('Pendulum Angle')
        axs[1, 1].plot(self.time, self.pendulum_angular_velocity)
        axs[1, 1].set_title('Pendulum Velocity')
        axs[2, 1].plot(self.time, self.pendulum_angular_acceleration)
        axs[2, 1].set_title('Pendulum Acceleration')

        plt.savefig("game_statistics")
