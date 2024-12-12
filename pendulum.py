import pygame
from pygame import gfxdraw
from math import sin, cos, pi, atan2
from random import randint

class Scene:
    def __init__(self, width=1024, height=1024, agent=None):
        self.agent = agent
        self.width = width
        self.height = height
        self.fps = 60
        self.update_time = 0.1
        self.bg_color = (8, 8, 8)
        self.g = 10

        pygame.init()
        self.window = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Pendulum')
        self.clock = pygame.time.Clock()

        self.trace_surface = pygame.Surface((self.width, self.height))
        self.trace_surface.set_colorkey((0, 0, 0))
        self.platform = Platform(self.width // 2, self.height // 2)

        self.pendulum = Pendulum(self.platform, scene=self, length=100)
        self.ball = Ball(self, self.pendulum, trace=False, radius=10)

        #self.pendulum_1 = Pendulum(self.ball, scene=self,)
        #self.ball_1 = Ball(self, self.pendulum_1, trace=True)

        #self.pendulum_0 = Pendulum(self.platform, scene=self,)
        #self.ball_0 = Ball(self, self.pendulum_0, trace=False)

        #self.pendulum_2 = Pendulum(self.ball_0, scene=self)
        #self.ball_2 = Ball(self, self.pendulum_2, trace=True)

        #self.pendulum_3 = Pendulum(self.ball_2, scene=self)
        #self.ball_3 = Ball(self, self.pendulum_3, trace=True)


        #self.objects = [self.pendulum_0, self.ball_0, self.pendulum_2, self.ball_2, self.pendulum_3, self.ball_3]
        #self.objects = [self.pendulum, self.ball, self.pendulum_1, self.ball_1, self.pendulum_0, self.ball_0, self.pendulum_2, self.ball_2]
        #self.objects = [self.pendulum, self.ball, self.pendulum_1, self.ball_1]
        self.objects = [self.pendulum, self.ball]

        self.running = True

    def reset(self):
        self.objects = []

        self.platform = Platform(self.width // 2 + randint(-500, 500), self.height // 2)
        self.pendulum = Pendulum(self.platform, scene=self, length=100, theta=(pi/4)*randint(-10,10))
        self.ball = Ball(self, self.pendulum, trace=False, radius=10)

        self.objects = [self.pendulum, self.ball]

    def update_platform(self, event):
        if self.agent != 'ai':
            if event.pos[0] < 0:
                self.platform.x_center = 0
            elif event.pos[0] > self.width:
                self.platform.x_center = self.width
            else:
                self.platform.x_center = event.pos[0]

        self.platform.update_position()

    def update_frame(self):
        self.platform.draw(self.window)

        pygame.draw.line(surface=self.window, color=(128, 128, 128), start_pos=(0, self.platform.y_center),
                         end_pos=(self.width, self.platform.y_center), width=2)


        for obj in self.objects:
            obj.update_position()
            obj.draw()

    def update_ui(self):
        self.window.fill(self.bg_color)
        self.window.blit(self.trace_surface, (0, 0))
        self.update_frame()
        pygame.display.flip()

    def tick(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.MOUSEMOTION:
                self.update_platform(event)

        if self.agent == 'ai':
            self.clock.tick()
        else:
            self.clock.tick(self.fps)

    def next_frame(self):
        self.tick()
        self.update_ui()


    def loop(self):
        while self.running:
            self.next_frame()
        pygame.quit()

    def get_state(self):
        platform_x = self.platform.x_center
        platform_x_prev = self.platform.x_prev
        theta = self.pendulum.theta
        theta_prev = self.pendulum.theta_prev
        return platform_x, platform_x_prev, theta, theta_prev

    def reward(self):
        return abs(self.pendulum.theta)-pi/2

    def quit(self):
        self.running = False


class Platform:
    def __init__(self, x_center, y_center, width=100, height=10):
        self.width = width
        self.height = height
        self.x_center = x_center
        self.y_center = y_center
        self.x_prev = self.x_center
        self.rect = pygame.Rect(self.x_center - self.width // 2, self.y_center - self.height // 2,
                                self.width, self.height)

    def update_position(self):
        self.x_prev = self.x_center
        self.rect.x = self.x_center - self.width // 2


    def draw(self, surface):
        pygame.draw.rect(surface, (128, 128, 128), self.rect)


class Ball:
    def __init__(self, scene, pivot_object, radius=7, color=(250, 250, 250), mass=5, trace=False):
        self.scene = scene
        self.time = self.scene.update_time
        self.pivot_object = pivot_object
        self.radius = radius
        self.color = color
        self.mass = mass

        self.trace = trace

        self.x_center = self.pivot_object.x_end
        self.y_center = self.pivot_object.y_end

        self.x_prev = self.x_center
        self.y_prev = self.y_center

        self.force_center = 0
        self.force_norm = 0

        self.force_g = self.mass * self.scene.g
        self.theta = self.pivot_object.theta

        self.force = 0
        self.theta_f = 0

    def update_position(self):

        force_center = self.force_g * cos(self.pivot_object.theta) + self.force*cos(self.theta_f + self.pivot_object.theta)
        force_tang = self.force_g * sin(self.pivot_object.theta) + self.force*sin(self.theta_f + self.pivot_object.theta)

        self.x_prev = self.x_center
        self.y_prev = self.y_center

        self.pivot_object.force_end_center = force_center
        self.pivot_object.force_end_tang = force_tang

        self.x_center = self.pivot_object.x_end
        self.y_center = self.pivot_object.y_end

        self.pivot_object.ball_mass = self.mass


    def draw(self):
        pygame.gfxdraw.aacircle(self.scene.window, int(self.x_center), int(self.y_center), self.radius, self.color)
        pygame.gfxdraw.filled_circle(self.scene.window, int(self.x_center), int(self.y_center), self.radius, self.color)

        if self.trace:
            pygame.draw.rect(self.scene.trace_surface, self.color, pygame.Rect(self.x_center, self.y_center, 1, 1))


class Pendulum:
    def __init__(self, pivot_object, length=50, scene=None, ball=None, theta=0):
        self.color = (250, 250, 250)
        self.scene = scene
        self.pivot_object = pivot_object
        self.time = scene.update_time

        self.length = length
        self.ball_mass = ball.mass if ball is not None else 1
        self.x_start = self.pivot_object.x_center
        self.y_start = self.pivot_object.y_center
        self.theta = theta
        self.theta_prev = theta
        self.damping = 1

        self.x_end = self.x_start + self.length * sin(self.theta)
        self.y_end = self.y_start + self.length * cos(self.theta)

        self.x_prev_end = self.x_end
        self.y_prev_end = self.y_end
        self.x_prev_start = self.x_start
        self.y_prev_start = self.y_start

        self.force_end_center = 0
        self.force_end_tang = 0
        self.v = 0

    def update_position(self):

        # save previous coord
        self.x_prev_end = self.x_end
        self.y_prev_end = self.y_end
        self.x_prev_start = self.pivot_object.x_center
        self.y_prev_start = self.pivot_object.y_center

        # calculate delta theta
        delta_v = (self.force_end_tang * self.time) / self.ball_mass
        self.v -= delta_v
        self.v *= self.damping
        delta_s = self.v * self.time
        delta_theta = atan2(delta_s, self.length)

        self.x_start = self.pivot_object.x_center
        self.y_start = self.pivot_object.y_center
        self.x_end = self.x_start + self.length * sin(self.theta)
        self.y_end = self.y_start + self.length * cos(self.theta)

        self.theta_prev = self.theta
        self.theta = atan2(self.x_prev_end - self.x_start, self.y_prev_end - self.y_start)

        self.theta += delta_theta

        self.x_end = self.x_start + self.length * sin(self.theta)
        self.y_end = self.y_start + self.length * cos(self.theta)

        self.pivot_object.theta_f = self.theta
        self.pivot_object.force = self.force_end_center


    def draw(self):
        pygame.draw.line(self.scene.window, (255, 255, 255), (self.x_start, self.y_start),
                         (self.x_end, self.y_end), 3)


if __name__ == '__main__':
    scene = Scene()
    scene.loop()


