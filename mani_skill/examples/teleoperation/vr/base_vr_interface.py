class VRSimTeleopInterface:
    """
    Base class for implementing VR headsets for teleoperation in simulation
    """

    def calibrate_ee(self):
        """
        Run any code to calibrate such that the desired robot end-effector is placed directly at the controller position.

        This is called each time the environment is reset
        """
