import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class BasicRewardWithPitchRollPenaltyShrinkingBoundaries(BaseRLAviary):
    """Single agent RL problem: hover at position."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 target_xyzs=np.array([0, 0, 1]),
                 target_rpys=np.array([0, 0, 1.7]),
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.INIT_XYZS = initial_xyzs
        self.TARGET_POS = target_xyzs
        self.TARGET_ORIENTATION = target_rpys
        self.EPISODE_LEN_SEC = 15
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )

    ################################################################################

    def _target_error(self, state):
        return (np.linalg.norm(self.TARGET_POS - state[0:3]) +
                np.linalg.norm(self.TARGET_ORIENTATION - state[7:10]))

    def _is_away_from_exploration_area(self, state):
        return (np.linalg.norm(self.INIT_XYZS[0][0:2] - state[0:2]) >
                np.linalg.norm(self.INIT_XYZS[0][0:2] - self.TARGET_POS[0:2]) + 0.05 or
                state[2] > self.TARGET_POS[2] + 0.0125)

    def _is_closed(self, state):
        return np.linalg.norm(state[0:3] - self.TARGET_POS[0:3]) < 0.025

    def _performance(self, state):
        if self._is_closed(state) and state[7]**2 + state[8]**2 < 0.001:
            return 2

        return -(state[7]**2 + state[8]**2)

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        ret = (25 - 20 * self._target_error(state) -
               100 * (1 if self._is_away_from_exploration_area(state) else -0.2) +
               20 * self._performance(state) -
               18 * (state[13]**2 + state[14]**2 + state[15]**2))
        return ret

    ################################################################################

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_POS - state[0:3]) < .02 and state[7]**2 + state[8]**2 < 0.0005:
            return True

        return False
        
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        state = self._getDroneStateVector(0)
        if (np.linalg.norm(self.INIT_XYZS[0][0:2] - state[0:2]) >
                np.linalg.norm(self.INIT_XYZS[0][0:2] - self.TARGET_POS[0:2]) + 0.1 or
                state[2] > self.TARGET_POS[2] + 0.025 or
                abs(state[7]) > .15 or abs(state[8]) > .15):
            return True

        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True

        return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42}  # Calculated by the Deep Thought supercomputer in 7.5M years