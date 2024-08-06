import numpy as np
import pygame
import sys

# Constants
g = 9.81  # Acceleration due to gravity (m/s^2)
L = 1.0   # Length of the pendulum (m)
b = 0.5 # Damping coefficient
dt = 0.01  # Time step (s)
M = 1       # Mass of Pendulum Ball
acc_offset = 3

# Initial conditions of pendulum
theta = 0 # np.pi / 4  # Initial angle (radians)
omega = 0  # Initial angular velocity (rad/s)


# Pygame setup
pygame.init()
screen = pygame.display.set_mode((1280,720))
clock = pygame.time.Clock()

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (70,130,180)

# Initial conditions of pivot
pivot_x = screen.get_width()/2  # Initial x position of the pivot
pivot_y = screen.get_height()/2 - 100  # Y position of the pivot (fixed)

# player area
play_area = pygame.Surface((screen.get_width()-100, 100))
play_area_rect = play_area.get_rect(topleft = (50, screen.get_height()/2 + 100))
play_area.fill(BLACK)


# Equations of motion for the pendulum - returns the angular acceleration
def derivatives(theta, omega, x_acc):
    # Linear solution
    # sin_theta = np.sin(theta)
    # theta_acc = - (g / L) * sin_theta - b * omega
    theta_acc = (3/2) * ((g * np.sin(-theta) - x_acc * np.cos(-theta))/L) - b * omega
    print("x acceleration:", theta_acc)
    return theta_acc

# get starting mouse position
mouse_pos = pygame.mouse.get_pos()
x_velocity = 0
y_velocity = 0
x_velocity_current = 0

# Main loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    # position of mouse 
    mouse_pos_current = pygame.mouse.get_pos()
    collide = pygame.Rect.collidepoint(play_area_rect, mouse_pos_current)

    if collide:
        x_velocity_current = mouse_pos_current[0] - mouse_pos[0]   
        y_velocity_current = mouse_pos_current[1] - mouse_pos[1]
    else:
        x_velocity_current = 0
        
    # acceleration of cart
    acceleration = (x_velocity_current * 10 - x_velocity * 10) * acc_offset

    x_velocity = x_velocity_current

    print("x velocity:", x_velocity)
    print("y velocity:", y_velocity)
    mouse_pos = mouse_pos_current

    # player movement
    # keys = pygame.key.get_pressed()
    # if keys[pygame.K_a]:
    #     pivot_x -= 1
    # if keys[pygame.K_d]:
    #     pivot_x += 1
    pivot_x += x_velocity

    # update omega from acceleration
    # acceleration_angle = (3/2) * ((g * np.sin(np.pi - theta) - acceleration * np.cos(np.pi - theta))/L)
    # velocity_angle = acceleration_angle * dt

    # Update the physics of the pendulum
    theta_acc = derivatives(theta, omega, acceleration)
    omega += theta_acc * dt 
    theta += omega * dt
    print("Angular Speed:", omega)
    # Clear the screen
    screen.fill(WHITE)


    # Draw player area
    screen.blit(play_area, play_area_rect)

    # Draw the pivot
    pygame.draw.circle(screen, BLACK, (pivot_x, pivot_y), 5)

    # Draw the pendulum
    pendulum_x = pivot_x + int(L * 100 * np.sin(theta))
    pendulum_y = pivot_y + int(L * 100 * np.cos(theta))
    pygame.draw.line(screen, BLACK, (pivot_x, pivot_y), (pendulum_x, pendulum_y), 2)
    pygame.draw.circle(screen, BLUE, (pendulum_x, pendulum_y), 10)

    # Update the display
    pygame.display.flip()
    clock.tick(60)
