import pygame
import time

class Joystick:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            print("No joystick detected")
            self.joystick = None
        else:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()

    def get_transmitter_values(self):
        pygame.event.pump()

        if not self.joystick:
            return None  # Handle case where joystick is not connected

        # Axis values (Thumbsticks & Triggers)
        axes = {
            "L1": -self.joystick.get_axis(1),
            "L2": self.joystick.get_axis(0),
            "R1": -self.joystick.get_axis(3),
            "R2": self.joystick.get_axis(2),
            "TL": self.joystick.get_axis(4),
            "TR": self.joystick.get_axis(5),
        }

        # Button values
        buttons = {
            "A": self.joystick.get_button(0),
            "B": self.joystick.get_button(1),
            "X": self.joystick.get_button(2),
            "Y": self.joystick.get_button(3),
            "LB": self.joystick.get_button(9),
            "RB": self.joystick.get_button(10),
            "Back": self.joystick.get_button(4),
            "Start": self.joystick.get_button(6),
            "LStick": self.joystick.get_button(7),
            "RStick": self.joystick.get_button(8),
            "Guide": self.joystick.get_button(5),
            "KU": self.joystick.get_button(11),
            "KD": self.joystick.get_button(12),
            "KL": self.joystick.get_button(13),
            "KR": self.joystick.get_button(14),
        }

        # Trigger rumble effect if "B" button is pressed
        selected_segment = None


            
        if buttons["Start"] == 1: 
            selected_segment = 0
            self.joystick.rumble(1, 1, 200)

        if buttons["X"] == 1:
            selected_segment = 1
            self.joystick.rumble(1, 1, 400)
           
        if buttons["B"] == 1:
            selected_segment = 2
            self.joystick.rumble(1, 1, 800)
            

        
        # if keys[pygame.K_1]:
        #     selected_segment = 0
        # elif keys[pygame.K_2]:
        #     selected_segment = 1
        # elif keys[pygame.K_3]:
        #     selected_segment = 2

        return axes, buttons, selected_segment


    def remap_value(self, value, old_min, old_max, new_min, new_max):
        # Handle cases where value is outside old range
        if value < old_min:
            value = old_min
        elif value > old_max:
            value = old_max
        old_range = old_max - old_min
        new_range = new_max - new_min
        scaled_value = (value - old_min) / old_range
        new_value = scaled_value * new_range + new_min
        return new_value


def main():
  joystick = Joystick()

  running = True
  while running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False

    axes, buttons, _ = joystick.get_transmitter_values()
    
    print(axes)
    print(buttons)
    # print(trigger)
    # if throttle is not None:
    #     # ... your code here to process joystick values ...

    # else:
    #     print("Failed to initialize joystick")
    #     break
    time.sleep(0.1)
  pygame.quit()

if __name__ == "__main__":
  main()
