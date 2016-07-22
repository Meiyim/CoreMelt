import numpy as np


class PressureVessle(object):
    currentTime = 98.
    currentWater = 0.0
    waterHistory = None
    powerHistory = None

    @classmethod
    def timePush(cls,dt):
        cls.currentTime += dt
        cls.waterHistory = np.interp(np.array(cls.currentTime),cls.waterHistory,cls.powerHistory)

class RodType:
    fuel = 1
    gray = 2
    black = 3
    empty = 4


class RodUnit(object):
    def __init__(self, _id, nh, nr, pos, add):
        self.index = _id
        assert isinstance(pos, tuple)
        self.position = np.array(pos)
        self.address = add
        self.T = np.zeros((nh, nr))
        self.bT = np.zeros(nh)
        self.radialPowerFactor = 0.0  # main power
        self.axialPowwerFactor = np.zeros(nh) #axial power
        self.neighbour = {}
        self.type = RodType.empty
        # ksp stuff
