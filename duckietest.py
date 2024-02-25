import time
from typing import Tuple, List
from duckietown.sdk.robots.duckiebot import DB21J
from duckietown.sdk.types import LEDsPattern, RGBAColor

SIMULATED_ROBOT_NAME: str = "map_0/vehicle_0"
REAL_ROBOT_NAME: str = "rover"

robot: DB21J = DB21J(SIMULATED_ROBOT_NAME, simulated=True)  # change accordingly


# move wheels
speeds = (-0.5, -0.5)
robot.motors.start()
# robot.motors.stop()
stime: float = time.time()
while time.time() - stime < 4:
    robot.motors.publish(speeds)
    time.sleep(0.25)
    print("speedy")
robot.motors.stop()
print("Stopped.")


# # LED lights show
# frequency: float = 1.4
# off: RGBAColor = (0, 0, 0, 0.0)
# amber: RGBAColor = (1, 0.7, 0, 1.0)
# lights_on: LEDsPattern = LEDsPattern(front_left=amber, front_right=amber, rear_right=amber, rear_left=amber)
# lights_off: LEDsPattern = LEDsPattern(front_left=off, front_right=off, rear_right=off, rear_left=off)
# pattern: List[LEDsPattern] = [lights_on, lights_off]
# robot.lights.start()
# stime: float = time.time()
# i: int = 0
# while time.time() - stime < 8:
#     lights: LEDsPattern = pattern[i % 2]
#     robot.lights.publish(lights)
#     time.sleep(1. / frequency)
#     i += 1
# robot.lights.stop()
# print("Stopped.")      
