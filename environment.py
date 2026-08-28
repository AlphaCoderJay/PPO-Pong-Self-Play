import pygame
import random

pygame.init()

# -------------------------
# Settings
# -------------------------

WIDTH = 800
HEIGHT = 500

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ping Pong")

clock = pygame.time.Clock()

FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# -------------------------
# Game objects
# -------------------------




font = pygame.font.Font(None, 64)

class Direction(int):
    UP = 0
    DOWN = 1



class PingPongEnv:
    # -------------------------
    # Reset ball
    # -------------------------

    def __init__(self,w=800,h=500, render = False):
        self.left_score = 0
        self.right_score = 0
        self.w = w
        self.h = h

        self.render = render
        if self.render:
            self.screen = pygame.display.set_mode((self.w ,self.h))
            pygame.display.set_caption("Ping Pong")
            self.clock = pygame.time.Clock()

        self.PADDLE_WIDTH = 15
        self.PADDLE_HEIGHT = 100
        self.PADDLE_SPEED = 6

        self.BALL_SIZE = 15
        self.BALL_SPEED = 5

        self.left_paddle = pygame.Rect(
            30,
            self.h // 2 - self.PADDLE_HEIGHT // 2,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )

        self.right_paddle = pygame.Rect(
            self.w - 30 - self.PADDLE_WIDTH,
            self.h // 2 - self.PADDLE_HEIGHT // 2,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )

        self.ball = pygame.Rect(
            self.w // 2 - self.BALL_SIZE // 2,
            self.h // 2 - self.BALL_SIZE // 2,
            self.BALL_SIZE,
            self.BALL_SIZE
        )

        # Ball velocity
        self.ball_x_velocity = self.BALL_SPEED
        self.ball_y_velocity = random.choice([-self.BALL_SPEED, self.BALL_SPEED])


    def reset_ball(self,direction):


        self.ball.center = (self.w // 2, self.h // 2)

        self.ball_x_velocity = self.BALL_SPEED * direction
        self.ball_y_velocity = random.choice([-self.BALL_SPEED, self.BALL_SPEED])

    def reset(self):
        self.direction = 0

        self.left_score = 0
        self.right_score = 0


        self.left_paddle = pygame.Rect(
            30,
            self.h // 2 - self.PADDLE_HEIGHT // 2,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )

        self.right_paddle = pygame.Rect(
            self.w - 30 - self.PADDLE_WIDTH,
            self.h // 2 - self.PADDLE_HEIGHT // 2,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )

        self.ball = pygame.Rect(
            self.w // 2 - self.BALL_SIZE // 2,
            self.h // 2 - self.BALL_SIZE // 2,
            self.BALL_SIZE,
            self.BALL_SIZE
        )


    def close(self):
        pygame.quit()



    def get_player_observation(self,player):
        if player == "left_paddle":
            me = self.left_paddle
            opponent = self.right_paddle

        else:
            me = self.right_paddle
            opponent = self.left_paddle


        observation = [
            me.y/self.h,
            opponent.y / self.h,
            self.ball.x / self.w,
            self.ball.y / self.h,
            self.ball_x_velocity / self.BALL_SPEED,
            self.ball_y_velocity / self.BALL_SPEED,


        ]
        return observation


    # -------------------------
    # Main game loop
    # -------------------------
    def step(self,left_paddle_action, right_paddle_action):
 
        # -------------------------
        # Keyboard input
        # -------------------------

        game_over = False

        # Left paddle - W/S
        if left_paddle_action == 0:
            self.left_paddle.y -= self.PADDLE_SPEED
            self.direction = Direction.UP

        if left_paddle_action == 1:
            self.left_paddle.y += self.PADDLE_SPEED
            self.direction = Direction.DOWN

        # Right paddle - UP/DOWN
        if right_paddle_action == 0:
            self.right_paddle.y -= self.PADDLE_SPEED
            self.direction = Direction.UP

        if right_paddle_action == 1:
            self.right_paddle.y += self.PADDLE_SPEED
            self.direction = Direction.DOWN

        # -------------------------
        # Keep paddles on screen
        # -------------------------

        if self.left_paddle.top < 0:
            self.left_paddle.top = 0

        if self.left_paddle.bottom > self.h:
            self.left_paddle.bottom = self.h

        if self.right_paddle.top < 0:
            self.right_paddle.top = 0

        if self.right_paddle.bottom > self.h:
            self.right_paddle.bottom = self.h

        # -------------------------
        # Move ball
        # -------------------------

        self.ball.x += self.ball_x_velocity
        self.ball.y += self.ball_y_velocity

        # -------------------------
        # Ball hits top/bottom
        # -------------------------

        if self.ball.top <= 0:
            self.ball.top = 0
            self.ball_y_velocity *= -1

        if self.ball.bottom >= self.h:
            self.ball.bottom = self.h
            self.ball_y_velocity *= -1

        # -------------------------
        # Ball hits paddles
        # -------------------------

                # -------------------------
        # Scoring
        # -------------------------
        right_paddle_reward = 0.0
        left_paddle_reward = 0.0

        if self.ball.colliderect(self.left_paddle) and self.ball_x_velocity < 0:

            self.ball.left = self.left_paddle.right
            left_paddle_reward += 1.0
            self.ball_x_velocity *= -1


        if self.ball.colliderect(self.right_paddle) and self.ball_x_velocity > 0:

            self.ball.right = self.right_paddle.left
            right_paddle_reward += 1.0
            self.ball_x_velocity *= -1
           




        if self.ball.left <= 0:

            self.right_score += 1
            #right_paddle_reward += 1.5
            #left_paddle_reward -= -0.5

            self.reset_ball(1)

        if self.ball.right >= self.w:

            self.left_score += 1
            #left_paddle_reward += 1.5
            #right_paddle_reward -= 0.5

            self.reset_ball(-1) 


        if self.render:
            self._update_ui()
            self.clock.tick(60)


        if self.right_score >= 3 and self.left_score < 3:
            #right_paddle_reward += 2.0
            #left_paddle_reward -= 2.0
            game_over = True
            self.reset()

        elif self.left_score >= 3 and self.right_score < 3:
            #left_paddle_reward += 2.0
            #right_paddle_reward -= 2.0
            game_over = True
            self.reset()


        return left_paddle_reward,right_paddle_reward,game_over #i didnt add obs, will update seperately

    def _update_ui(self):

            self.screen.fill(BLACK)

            # Middle line
            pygame.draw.line(
                self.screen,
                WHITE,
                (self.w // 2, 0),
                (self.w // 2, self.h),
                2
            )

            # Paddles
            pygame.draw.rect(self.screen, WHITE, self.left_paddle)
            pygame.draw.rect(self.screen, WHITE, self.right_paddle)

            # Ball
            pygame.draw.rect(self.screen, WHITE, self.ball)

            # Score
            left_text = font.render(str(self.left_score), True, WHITE)
            right_text = font.render(str(self.right_score), True, WHITE)

            self.screen.blit(
                left_text,
                (self.w // 2 - 100, 30)
            )

            self.screen.blit(
                right_text,
                (self.w // 2 + 70, 30)
            )

            pygame.display.flip()

            clock.tick(60)


