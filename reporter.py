import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class Reporter:
    stats = dict(
        pivot_position=[],
        pivot_velocity=[],
        pivot_acceleration=[],
        pendulum_angle=[],
        pendulum_angular_velocity=[],
        pendulum_angular_acceleration=[],
        time=[],
        score=[]
    )
    descriptions = ["Pivot Location",
                    "Pivot Velocity",
                    "Pivot Acceleration",
                    "Pendulum Angle",
                    "Pendulum Angular Velocity",
                    "Pendulum Angular Acceleration",
                    "Time",
                    "Score"]

    def __init__(self, dt, environment):
        self.dt = dt
        self.environment = environment

    def record(self):
        temp_stats = self.environment.return_state()

        if np.pi - 0.5 <= temp_stats[3] % (np.pi * 2) <= np.pi + 0.5:
            print("Score!")

        print("-------------------")

        # printing
        for i, text in enumerate(self.descriptions):
            print(text, temp_stats[i], sep=":\t\t\t")

        # storing
        for i, key in enumerate(self.stats.keys()):
            self.stats[key].append(temp_stats[i])

    def plot(self):
        # fig, axs = plt.subplots(4, 2, layout="constrained")

        fig, axs = plt.subplot_mosaic([["Pivot Location", "Pivot Velocity", "Pivot Acceleration"],
                                       ["Pendulum Angle", "Pendulum Angular Velocity", "Pendulum Angular Acceleration"],
                                       ["Score", "Score", "Score"]],
                                      layout="constrained", figsize=(12.8, 8))

        fig.suptitle('Pendulum Statistics')

        temp_desc = self.descriptions[:]
        temp_desc.remove("Time")
        temp_key = list(self.stats.keys())
        temp_key.remove("time")

        descriptions_key = {"Pivot Location": "pivot_position",
                            "Pivot Velocity": "pivot_velocity",
                            "Pivot Acceleration": "pivot_acceleration",
                            "Pendulum Angle": "pendulum_angle",
                            "Pendulum Angular Velocity": "pendulum_angular_velocity",
                            "Pendulum Angular Acceleration": "pendulum_angular_acceleration",
                            "Time": "time",
                            "Score": "score"}

        for key, ax in axs.items():
            ax.plot(self.stats["time"], self.stats[descriptions_key[key]])
            ax.set_title('Time vs ' + key)

        # plt.show()
        save_path = Path() / "Output"
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / "game_statistics")
