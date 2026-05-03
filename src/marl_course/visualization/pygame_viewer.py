"""Pygame viewer backed by the same sprite renderer used for GIF output."""
from __future__ import annotations

from marl_course.visualization.gif import ansi_to_rgb_array


class PygameGridViewer:
    """Tiny pygame viewer for classroom projection.

    The live viewer intentionally delegates all grid drawing to
    `ansi_to_rgb_array`. This keeps `--live` and generated GIFs visually
    identical: same sprites, same colors, same legend, same progress bars.
    """

    def __init__(self, tile_size: int = 48):
        """Initialize pygame lazily and store tile size for render scaling."""
        import numpy as np
        import pygame

        self.np = np
        self.pygame = pygame
        self.tile_size = tile_size
        self.screen = None
        pygame.init()

    def draw_from_ansi(self, ansi: str, title: str = "MARL Course Games", agent_labels: list[tuple[str, str]] | None = None) -> None:
        """Draw one ANSI frame using the shared RGB sprite renderer."""
        frame = ansi_to_rgb_array(ansi, tile_size=self.tile_size, np=self.np, agent_labels=agent_labels)
        height, width, _channels = frame.shape
        if self.screen is None or self.screen.get_size() != (width, height):
            self.screen = self.pygame.display.set_mode((width, height))
        self.pygame.display.set_caption(title)

        # pygame.surfarray expects (width, height, channels), while numpy image
        # frames are stored as (height, width, channels).
        surface = self.pygame.surfarray.make_surface(self.np.transpose(frame, (1, 0, 2)))
        self.screen.blit(surface, (0, 0))
        self.pygame.display.flip()

        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                raise KeyboardInterrupt

    def close(self) -> None:
        """Close pygame window and release resources."""
        if self.screen is not None:
            self.pygame.quit()
            self.screen = None
