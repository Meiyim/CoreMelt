import numpy as np


class RodUnit(object):
    def __init__(self):
        self.index = 0
        self.position = (0., 0.)
        self.T = None
        self.bT = None
        self.address = None
        self.neighbour = None
        #ksp stuff

    def __init__(self, _id, nh, nr, pos, add):
        self.__init__()
        self.index = _id
        self.position = pos
        self.address = add
        self.T = np.zeros((nh, nr))
        self.bT = np.zeros(nh)
