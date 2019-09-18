# class that maintains information of the current state of the ATGM Turret
import threading

class ATGMState(object):
    # constructor
    def __init__(self):
        self._azimuth_steps = 0
        self._elevation_steps = 0
        self._zoom = 1
        self._azi_lock = threading.Lock()
        self._ele_lock = threading.Lock()
        self._zoom_lock = threading.Lock()

    # hard reset
    def reset(self):
        self._azimuth_steps = 0
        self._elevation_steps = 0
        self._zoom = 1

    @property
    def Azimuth_steps(self):
        with self._azi_lock:
            return self._azimuth_steps

    @Azimuth_steps.setter
    def Azimuth_steps(self, value):
        with self._azi_lock:
            self._azimuth_steps = value

    @property
    def Elevation_steps(self):
        with self._ele_lock:
            return self._elevation_steps

    @Elevation_steps.setter
    def Elevation_steps(self, value):
        with self._ele_lock:
            self._elevation_steps = value

    @property
    def Zoom(self):
        with self._zoom_lock:
            return self._zoom

    @Zoom.setter
    def Zoom(self, value):
        with self._zoom_lock:
            self._zoom = value
