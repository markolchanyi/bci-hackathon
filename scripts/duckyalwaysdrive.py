import time
from typing import Tuple, List
from duckietown.sdk.robots.duckiebot import DB21J
from duckietown.sdk.types import LEDsPattern, RGBAColor
from pynput import keyboard

SIMULATED_ROBOT_NAME: str = "map_0/vehicle_0"
REAL_ROBOT_NAME: str = "rover"

robot: DB21J = DB21J(SIMULATED_ROBOT_NAME, simulated=True)  # change accordingly

# Define global variables for speeds and movement flags
speeds = [0,0];
straight = [1,1]
pressed = False;

# def update_speeds(spd):
#     global speeds
#     speeds = spd;
#     return

def accelerate(max=1):
    global speeds
    for i in range(10):
        speeds[0] = ((i+1)/10)*max;
        speeds[1] = ((i+1)/10)*max;
        robot.motors.publish(tuple(speeds))
        time.sleep(0.25)
    print('done')
    return


def on_press(key):
    #global speeds, moving_forward, moving_backward
    global speeds, pressed
    if pressed:
        print('held')
        return
    pressed = True;
    if key == keyboard.Key.left:
        speeds[0] -= 0.8;
        speeds[1] -= 0.5;
    elif key == keyboard.Key.right:
        speeds[1] -= 0.8;
        speeds[0] -= 0.5;   
    elif key == keyboard.Key.down:
        speeds[0] = -straight[0];
        speeds[1] = -straight[1];
    return


def on_release(key):
    #global speeds, moving_forward, moving_backward
    global speeds, pressed
    print('release')
    pressed = False;
    if key == keyboard.Key.left:
        speeds[0] = straight[0];
        speeds[1] = straight[1];
    elif key == keyboard.Key.right:
        speeds[0] = straight[0];
        speeds[1] = straight[1];
    elif key == keyboard.Key.down:
        speeds[0] = straight[0];
        speeds[1] = straight[1];
    return


listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

robot.motors.start()
speeds = straight.copy()
robot.motors.publish(tuple(speeds))
time.sleep(0.25)

stime: float = time.time()
while time.time() - stime < 10:
    robot.motors.publish(tuple(speeds))
    print(speeds)
    time.sleep(0.25)

listener.stop()

