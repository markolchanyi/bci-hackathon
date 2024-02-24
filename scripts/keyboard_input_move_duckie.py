import time
from typing import Tuple, List
from duckietown.sdk.robots.duckiebot import DB21J
from duckietown.sdk.types import LEDsPattern, RGBAColor
from pynput import keyboard

SIMULATED_ROBOT_NAME: str = "map_0/vehicle_0"
REAL_ROBOT_NAME: str = "rover"

# Initialize the robot
robot: DB21J = DB21J(SIMULATED_ROBOT_NAME, simulated=True)  # change accordingly

# Define global variables for speeds and movement flags
speeds = [0.0, 0.0]
moving_forward = False
moving_backward = False


def update_speeds():
    global speeds, moving_forward, moving_backward
    if moving_forward:
        base_speed = 0.4
        speeds = [base_speed, base_speed]
    elif moving_backward:
        base_speed = -0.4
        speeds = [base_speed, base_speed]

    robot.motors.publish(tuple(speeds))


def on_press(key):
    global speeds, moving_forward, moving_backward
    try:
        if key.char == 'a':  # Turn left
            speeds[0] -= 0.1
            speeds[1] += 0.1
        elif key.char == 'd':  # Turn right
            speeds[0] += 0.1
            speeds[1] -= 0.1
    except AttributeError:
        if key == keyboard.Key.up:  # Move forward
            moving_forward = True
        elif key == keyboard.Key.down:  # Move backward
            moving_backward = True
    update_speeds()


def on_release(key):
    global speeds, moving_forward, moving_backward
    if key == keyboard.Key.esc:
        # Stop listener
        return False
    try:
        if key.char == 'a' or key.char == 'd':
            # Reset turning adjustments
            if moving_forward:
                speeds = [0.4, 0.4]
            elif moving_backward:
                speeds = [-0.4, -0.4]
            else:
                speeds = [0.0, 0.0]
    except AttributeError:
        if key == keyboard.Key.up:
            moving_forward = False
        elif key == keyboard.Key.down:
            moving_backward = False
        if not moving_forward and not moving_backward:
            speeds = [0.0, 0.0]  # Stop movement
    update_speeds()


listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# Move wheels
robot.motors.start()
stime: float = time.time()
while time.time() - stime < 10:
    time.sleep(0.25)
robot.motors.stop()
print("Stopped.")

# LED lights show
frequency: float = 1.4
off: RGBAColor = (0, 0, 0, 0.0)
amber: RGBAColor = (1, 0.7, 0, 1.0)
lights_on: LEDsPattern = LEDsPattern(front_left=amber, front_right=amber, rear_right=amber, rear_left=amber)
lights_off: LEDsPattern = LEDsPattern(front_left=off, front_right=off, rear_right=off, rear_left=off)
pattern: List[LEDsPattern] = [lights_on, lights_off]
robot.lights.start()
stime: float = time.time()
i: int = 0
while time.time() - stime < 8:
    lights: LEDsPattern = pattern[i % 2]
    robot.lights.publish(lights)
    time.sleep(1. / frequency)
    i += 1
robot.lights.stop()
print("Stopped.")
