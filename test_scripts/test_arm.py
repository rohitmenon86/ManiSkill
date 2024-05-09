#!/usr/bin/env python
import sys
import os

# Add the root directory of your package to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# imports your robot and registers it
from mani_skill.agents.robots import PandaWristCam, MapperArm, Panda
# imports the demo_robot example script and lets you test your new robot
import mani_skill.examples.demo_robot as demo_robot_script
demo_robot_script.main()