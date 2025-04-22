import pygame
import time

class KeyboardController:
    def __init__(self):
        pygame.init()
        self.keys_pressed = set()

    def get_transmitter_values(self):
        pygame.event.pump()
        keys = pygame.key.get_pressed()

        # Simulate axis-like controls (WASD & Arrows)
        axes = {
            "L1": float(keys[pygame.K_w]) - float(keys[pygame.K_x]),  # Forward/backward
            "L2": float(keys[pygame.K_d]) - float(keys[pygame.K_a]),  # Left/right
            "R1": float(keys[pygame.K_UP]) - float(keys[pygame.K_DOWN]),  # Up/down
            "R2": float(keys[pygame.K_RIGHT]) - float(keys[pygame.K_LEFT]),  # Tilt left/right
            "TL": float(keys[pygame.K_q]) - float(keys[pygame.K_e]),  # Extra axis
            "TR": float(keys[pygame.K_z]) - float(keys[pygame.K_c]),  # Extra axis
        }

        # Simulate buttons
        buttons = {
            "A": keys[pygame.K_k],
            "B": keys[pygame.K_l],
            "X": keys[pygame.K_j],
            "Y": keys[pygame.K_i],
            "LB": keys[pygame.K_1],
            "RB": keys[pygame.K_2],
            "Back": keys[pygame.K_BACKSPACE],
            "Start": keys[pygame.K_RETURN],
            "Guide": keys[pygame.K_TAB],
            "KU": keys[pygame.K_t],
            "KD": keys[pygame.K_g],
            "KL": keys[pygame.K_f],
            "KR": keys[pygame.K_h],
        }

        selected_segment = None
        if keys[pygame.K_1]:
            selected_segment = 0
        elif keys[pygame.K_2]:
            selected_segment = 1
        elif keys[pygame.K_3]:
            selected_segment = 2

        return axes, buttons, selected_segment

def main():
    controller = KeyboardController()

    running = True
    current_segment = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
               event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        axes, buttons, seg = controller.get_transmitter_values()
        if seg is not None:
            current_segment = seg

        print(f"Using segment: {current_segment+1}")
        print("Axes:", axes)
        print("Buttons:", buttons)

        time.sleep(0.1)

    pygame.quit()