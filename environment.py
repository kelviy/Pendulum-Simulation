import pygame
import pendulum
import reporter
import numpy as np
import neat
import os
import NEAT
import neat.parallel
import math

class Environment:

    def __init__(self, screen_width = 1280, screen_height = 720):
        self.screen_width = screen_width                    # Play Screen Width
        self.screen_height = screen_height                  # Play Screen Height

        # Initialization
        self.dt = 0.01                                      # Time step (s)
        self.acc_offset = 10                             # Acceleration Offset Scale Factor

        self.x_velocity = 0
        self.x_acceleration = 0

        self.score = 0
        self.time = 0

        # Pendulum Initialization Parameters (..., ..., # Initial angle (radians), Initial angular velocity (rad/s))
        self.pendulum_obj = pendulum.Pendulum(screen_width, screen_height, 1, 0 , 0)
        
        self.reporter = reporter.Reporter(self.dt, self)
        

    def run(self):
        # Pygame setup
        pygame.init()
        screen = pygame.display.set_mode((self.screen_width,self.screen_height))
        clock = pygame.time.Clock()

        # Colors
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        BLUE = (70,130,180)
        RED = (255, 0, 0)
        CHINESE_WHITE = (220, 237, 230)
        GREY = (128,128,128) 

        # player area
        play_area = pygame.Surface((screen.get_width()-100, 100))
        play_area_rect = play_area.get_rect(topleft = (50, screen.get_height()/2 + 100))
        play_area.fill(BLACK)

        # player area dial
        dial = pygame.Surface((80, 96))
        dial_rect = dial.get_rect(midtop = (screen.get_width()/2, screen.get_height()/2+102))
        dial.fill(CHINESE_WHITE)

        # score area
        score_area = pygame.Surface((screen.get_width()-100, 100))
        score_area_rect = score_area.get_rect(topleft = (50, 100))
        score_area.fill(WHITE)      # to make invisible

        # text
        font = pygame.font.Font('freesansbold.ttf', 14)
        title = font.render("The Pendulum", True, BLACK)

        # get starting mouse position
        mouse_pos = pygame.mouse.get_pos()
        mouse_down = False
        running = True
        x_velocity_current = 0
        flag = False

        # Main Game Loop
        while True:
            flag = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.MOUSEBUTTONUP:
                    mouse_down = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_down = True

            # mouse tracking
            mouse_pos_current = pygame.mouse.get_pos()
            collide = pygame.Rect.collidepoint(play_area_rect, mouse_pos_current)

            if collide:
                if mouse_down:
                    x_velocity_current = mouse_pos_current[0] - mouse_pos[0]
                    
                    # moves dial
                    dial_rect.move_ip(x_velocity_current, 0)

                    if not(pygame.Rect.contains(play_area_rect, dial_rect)):
                        # reverse movement
                        dial_rect.move_ip(-x_velocity_current, 0)
                        x_velocity_current += x_velocity_current* -0.4
                        flag = True
                else:
                    x_velocity_current += x_velocity_current * -0.4
                    flag = True
            else:
                x_velocity_current += x_velocity_current* -0.4
                flag = True
                

            # acceleration of cart
            self.x_acceleration = (x_velocity_current - self.x_velocity) * self.acc_offset

            self.x_velocity = x_velocity_current
            mouse_pos = mouse_pos_current

            # Update the physics of the pendulum
            if (flag == False):
                self.pendulum_obj.pivot_x += self.x_velocity    

            self.pendulum_obj.update_physics(self.x_acceleration, self.dt)

            # record variables
            self.reporter.record()

            # Start Drawing
            screen.fill(WHITE)

            # Draw player area
            screen.blit(play_area, play_area_rect)
            # Draw score area
            screen.blit(score_area, score_area_rect)

            # Draw the pivot
            pivot_loc = self.pendulum_obj.get_pivot_loc()
            pendulum_loc = self.pendulum_obj.get_pendulum_loc()

            pygame.draw.circle(screen, BLACK, pivot_loc, 5)

            # Draw the pendulum
            pygame.draw.line(screen, BLACK, pivot_loc, pendulum_loc, 2)
            circle_rect = pygame.draw.circle(screen, BLUE, pendulum_loc, 10)

            # Check for score
            score_collide = pygame.Rect.contains(score_area_rect, circle_rect)

            if score_collide:
                self.score += 1

            # Draw Dial
            screen.blit(dial, dial_rect)

            # Draw text
            score_text = font.render("Score: " + str(int(self.score/10)), True, BLACK)
            screen.blit(title, (50,50))
            screen.blit(score_text, (screen.get_width()-200, 50))
            time_text = font.render("Time: " + str(int(self.time)) + " seconds", True, BLACK)
            screen.blit(time_text, (self.screen_width/2-50, 50))
            
            # update time
            self.time += self.dt
            # Update the display
            pygame.display.flip()

            self.dt = clock.tick(100) / 1000


    def plot(self):
        self.reporter.plot()
    
    def scoreCheck(self, velocity):
        velocity[0] = math.floor(velocity[0] * 10)


        if not(90 <= self.pendulum_obj.pivot_x + velocity[0] <=  1190):
            velocity[0] = 0

        
        self.pendulum_obj.pivot_x += velocity[0]

        # acceleration of cart
        self.x_acceleration = (velocity[0] - self.x_velocity) * self.acc_offset
        
        self.x_velocity = velocity[0]
        # Update the physics of the pendulum

        self.pendulum_obj.update_physics(self.x_acceleration, self.dt)
        # record variables
        #self.reporter.record()

        angle = self.pendulum_obj.theta % (np.pi * 2)

        # update time
        self.time += self.dt

        if np.pi - 0.3 <= angle <= np.pi + 0.3:
            #return 1/(self.pendulum_obj.omega ** 2) - (self.pendulum_obj.pivot_x) * 0.1
            return 1
        return 0
        
        # if (self.pendulum_obj.omega == 0):
        #     return -1 + score
        # else:
        #     return - abs(self.pendulum_obj.omega) * 0.4 + score

    def observe(self):
        return (self.pendulum_obj.pivot_x, self.pendulum_obj.theta, self.pendulum_obj.omega)

    def return_state(self):
        return (self.pendulum_obj.pivot_x,
                self.x_velocity,
                self.x_acceleration,
                self.pendulum_obj.theta,
                self.pendulum_obj.omega,
                self.pendulum_obj.derivatives(self.x_acceleration),
                self.time,
                self.score)


    def run_model(self, winner):
        # Pygame setup
        pygame.init()
        screen = pygame.display.set_mode((self.screen_width,self.screen_height))
        clock = pygame.time.Clock()

        # Colors
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        BLUE = (70,130,180)
        RED = (255, 0, 0)
        CHINESE_WHITE = (220, 237, 230)
        GREY = (128,128,128) 

        # player area
        play_area = pygame.Surface((screen.get_width()-100, 100))
        play_area_rect = play_area.get_rect(topleft = (50, screen.get_height()/2 + 100))
        play_area.fill(BLACK)

        # player area dial
        dial = pygame.Surface((80, 96))
        dial_rect = dial.get_rect(midtop = (screen.get_width()/2, screen.get_height()/2+102))
        dial.fill(CHINESE_WHITE)

        # score area
        score_area = pygame.Surface((screen.get_width()-100, 100))
        score_area_rect = score_area.get_rect(topleft = (50, 100))
        score_area.fill(WHITE)      # to make invisible

        # text
        font = pygame.font.Font('freesansbold.ttf', 14)
        title = font.render("The Pendulum", True, BLACK)

        x_velocity_current = 0

        # Main Game Loop
        while True:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            output = winner.activate(self.observe())
            print(self.observe())
            print(output)
            x_velocity_current = int(output[0] * 10)

            self.pendulum_obj.pivot_x += x_velocity_current

            if not(90 <= self.pendulum_obj.pivot_x <= 1190):
                self.pendulum_obj.pivot_x -= x_velocity_current
                x_velocity_current = 0

            # moves dial
            dial_rect = dial_rect.move(x_velocity_current, 0)


                            
            # acceleration of cart
            self.x_acceleration = (x_velocity_current - self.x_velocity) * self.acc_offset

            self.x_velocity = x_velocity_current

            # Update the physics of the pendulum
            self.pendulum_obj.update_physics(self.x_acceleration, self.dt)

            # record variables
            # self.reporter.record()

            # Start Drawing
            screen.fill(WHITE)

            # Draw player area
            screen.blit(play_area, play_area_rect)
            # Draw score area
            screen.blit(score_area, score_area_rect)

            # Draw the pivot
            pivot_loc = self.pendulum_obj.get_pivot_loc()
            pendulum_loc = self.pendulum_obj.get_pendulum_loc()

            pygame.draw.circle(screen, BLACK, pivot_loc, 5)

            # Draw the pendulum
            pygame.draw.line(screen, BLACK, pivot_loc, pendulum_loc, 2)
            circle_rect = pygame.draw.circle(screen, BLUE, pendulum_loc, 10)

            # Check for score
            score_collide = pygame.Rect.contains(score_area_rect, circle_rect)

            if score_collide:
                self.score += 1

            # Draw Dial
            screen.blit(dial, dial_rect)

            # Draw text
            score_text = font.render("Score: " + str(int(self.score/10)), True, BLACK)
            screen.blit(title, (50,50))
            screen.blit(score_text, (screen.get_width()-200, 50))
            time_text = font.render("Time: " + str(int(self.time)) + " seconds", True, BLACK)
            screen.blit(time_text, (self.screen_width/2-50, 50))
            
            # update time
            self.time += self.dt
            # Update the display
            pygame.display.flip()

            self.dt = clock.tick(60)/1000


def main():
    environment = Environment()
    environment.run()
    environment.plot()

def load_model():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_pendulum')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    environment = Environment()

    p = neat.Checkpointer.restore_checkpoint('models/neat-checkpoint-9')
    winner = p.run(NEAT.eval_genomes, 1)

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    environment.run_model(winner_net)
    
def test_scorer():
    envi = Environment()
    
    while True:
        print(envi.observe())
        user_input= eval(input("Enter Velocity: "))
        lis = [user_input]
        print(envi.scoreCheck(lis))

if __name__ == "__main__":
    main() # <-- Runs the pendulum simulation only
    # load_model() # <-- Loads from latest save, trains for one iteration
    #test_scorer() # <-- to test observer method
