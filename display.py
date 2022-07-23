from calendar import c
import pygame
from pygame import gfxdraw
import numpy as np

def _render(self, mode="human"):

    self.screen_width = 600
    self.screen_height = 400

    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    GRAY = (200, 200, 200)

    pygame.display.init()
    self.screen = pygame.display.set_mode(
        (self.screen_width, self.screen_height)
    )
    if self.clock is None:
        self.clock = pygame.time.Clock()

    world_width = self.x_threshold * 2
    scale = self.screen_width / world_width
    tablewidth = 10.0
    tableheight = 6.0

    if self.state is None:
        return None

    x = self.state

    self.surf = pygame.Surface((self.screen_width, self.screen_height))
    self.surf.fill((255, 255, 255))

    #Table
    l, r, t, b = -tablewidth / 2, tablewidth / 2, tableheight / 2, -tableheight / 2
    table_coords = [(l, b), (l, t), (r, t), (r, b)]
    table_coords = [(c[0] + self.screen_width/2, c[1] + self.screen_height/2) for c in table_coords]
    gfxdraw.aapolygon(self.surf, table_coords, GREEN)
    gfxdraw.filled_polygon(self.surf, table_coords, GREEN)

    #Community 
    x = (tablewidth / 2) - 100
    font = pygame.font.SysFont(None, 24)
    for i in range(5):
        if np.sum(community[i,:])==0:
            break
        card = Card(community[i,:])
        cardtext = card.value_as_string + '\n' 
        cardtext += card.suit_as_string 
        if card.suit % 2 == 0:
            img = font.render(cardtext, True, BLACK)
        else:
            img = font.render(cardtext, True, RED)
        self.screen.blit(img, (x, self.screen_height/2))
        x += 40

    self.surf = pygame.transform.flip(self.surf, False, True)
    self.screen.blit(self.surf, (0, 0))
    if mode == "human":
        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()

    elif mode in {"rgb_array", "single_rgb_array"}:
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )