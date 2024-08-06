import numpy as np
import pygame
import sys
import pendulum
import neat

# Constants
g = 9.81  # Acceleration due to gravity (m/s^2)
L = 1.0   # Length of the pendulum (m)
b = 0.5 # Damping coefficient
dt = 0.01  # Time step (s)
M = 1       # Mass of Pendulum Ball

acc_offset = 10

# Initial conditions of pendulum
theta = np.pi / 4 # np.pi / 4  # Initial angle (radians)
omega = 0  # Initial angular velocity (rad/s)

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((1280,720))
clock = pygame.time.Clock()

# Initial set-up
# get starting mouse position
mouse_pos = pygame.mouse.get_pos()
x_velocity = 0
y_velocity = 0
x_velocity_current = 0
score = 0

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (70,130,180)
RED = (255, 0, 0)
CHINESE_WHITE = (220, 237, 230)
GREY = (128,128,128) 

# Initial conditions of pivot
pivot_x = screen.get_width()/2  # Initial x position of the pivot
pivot_y = screen.get_height()/2 - 100  # Y position of the pivot (fixed)

# player area
play_area = pygame.Surface((screen.get_width()-100, 100))
play_area_rect = play_area.get_rect(topleft = (50, screen.get_height()/2 + 100))
play_area.fill(BLACK)

dial = pygame.Surface((80, 96))
dial_rect = dial.get_rect(midtop = (screen.get_width()/2, screen.get_height()/2+102))
dial.fill(CHINESE_WHITE)

# score area
score_area = pygame.Surface((screen.get_width()-100, 100))
score_area_rect = score_area.get_rect(topleft = (50, 100))
# score_area.fill(CHINESE_WHITE)
score_area.fill(WHITE)

# text
font = pygame.font.Font('freesansbold.ttf', 14)
title = font.render("The Pendulum", True, BLACK)
# score_inst = font.render("Points :)", True, GREY)
score_inst = font.render("Points :)", True, WHITE)


# Equations of motion for the pendulum - returns the angular acceleration
def derivatives(theta, omega, x_acc):
    theta_acc = (3/2) * ((g * np.sin(-theta) - x_acc * np.cos(-theta))/L) - b * omega
    print("x acceleration:", theta_acc)
    return theta_acc

# Additional Initialization
mouse_down = False

# Main Game Loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        if event.type == pygame.MOUSEBUTTONUP:
            mouse_down = False
            x_velocity_current = 0
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_down = True

    # position of mouse 
    mouse_pos_current = pygame.mouse.get_pos()
    collide = pygame.Rect.collidepoint(play_area_rect, mouse_pos_current)

    if collide:
        if mouse_down:
            x_velocity_current = mouse_pos_current[0] - mouse_pos[0]   
            
            dail_rect = dial_rect.move_ip(x_velocity_current, 0)

            if not(pygame.Rect.contains(play_area_rect, dial_rect)):
                dail_rect = dial_rect.move_ip(-x_velocity_current, 0)
                x_velocity_current = 0
    else:
        x_velocity_current = 0
        
    
    # acceleration of cart
    acceleration = (x_velocity_current * acc_offset - x_velocity * acc_offset)

    x_velocity = x_velocity_current

    print("x velocity:", x_velocity)
    mouse_pos = mouse_pos_current

    pivot_x += x_velocity


    # Update the physics of the pendulum
    theta_acc = derivatives(theta, omega, acceleration)
    omega += theta_acc * dt 
    theta += omega * dt
    print("Angular Speed:", omega)
    # Clear the screen
    screen.fill(WHITE)


    # Draw player area
    screen.blit(play_area, play_area_rect)
    # Draw score area
    screen.blit(score_area, score_area_rect)

    # Draw the pivot
    pygame.draw.circle(screen, BLACK, (pivot_x, pivot_y), 5)

    # Draw the pendulum
    pendulum_x = pivot_x + int(L * 100 * np.sin(theta))
    pendulum_y = pivot_y + int(L * 100 * np.cos(theta))
    pygame.draw.line(screen, BLACK, (pivot_x, pivot_y), (pendulum_x, pendulum_y), 2)
    circle_rect = pygame.draw.circle(screen, BLUE, (pendulum_x, pendulum_y), 10)

    # Check for score
    score_collide = pygame.Rect.contains(score_area_rect, circle_rect)

    if score_collide:
        score += 1

    # Draw Dial
    screen.blit(dial, dial_rect)

    # Draw text
    score_text = font.render("Score: " + str(int(score/10)), True, BLACK)
    screen.blit(title, (50,50))
    screen.blit(score_text, (screen.get_width()-200, 50))
    screen.blit(score_inst, (70, 120))

    # Update the display
    pygame.display.flip()
    clock.tick(100)
