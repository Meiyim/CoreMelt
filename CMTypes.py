import numpy as np
import math


class PressureVessle(object):
    currentTime = 98.
    currentWater = 0.0
    currentPower = 0.0
    waterHistory = None #type: np.ndarray
    powerHistory = None #type: np.ndarray

    @classmethod
    def now(cls):
       return cls.currentWater, cls.currentPower

    @classmethod
    def timePush(cls, dt):
        cls.currentTime += dt
        cls.currentWater = np.interp(np.array(cls.currentTime), cls.waterHistory[0], cls.waterHistory[1])
        cls.currentPower = np.interp(np.array(cls.currentTime), cls.powerHistory[0], cls.powerHistory[1])


class RodType:
    fuel = 1
    gray = 2
    black = 3
    empty = 4


class MaterialProterty:
    def __init__(self,name,v1,v2,v3,v4,v5,v6,v7,v8):
        self.name = name
        self.lamdaIn = v1
        self.lamdaOut = v2
        self.rouIn = v3
        self.rouOut = v4
        self.cpIn = v5
        self.cpOut = v6
        self.meltingPointIn = v7
        self.meltingPointOut = v8


class RodUnit(object):
    def __init__(self, _id, nh, nrin, nr, pos, add, rin, rout, l):
        self.index = _id
        assert isinstance(pos, tuple)
        self.position = np.array(pos, dtype=np.float64)
        self.address = add
        self.T = None   # type: np.ndarray
        self.ql = None #type:np.ndarray
        self.height = None #type: np.ndarray
        self.radialPowerFactor = 0.0  # main power
        self.axialPowwerFactor = None #type: np.ndarray
        self.neighbour = {}
        self.type = RodType.empty
        self.nH = nh
        self.nRin = nrin
        self.nR = nr
        self.inRadious = rin
        self.radious = rout
        self.gapHeatRate = l
        #material
        self.material = None # type: MaterialProterty
        # ksp stuff

    def getTecplotZone(self):
        strBuffer = 'ZONE N=%d, E=%d, VARLOCATION=([1-3]=NODAL,[4]=CELLCENTERED) DATAPACKINIG=BLOCK, ZONETYPE=FEBRICK\n' \
                % ((self.nH + 1) * 4 , self.nH)#type: str
        center = self.position
        rad = self.radious / math.sqrt(2)
        # print points
        basePoint = [(center[0] + rad,center[1] + rad), #1
                     (center[0] + rad,center[1] - rad), #2
                     (center[0] - rad,center[1] - rad), #3
                     (center[0] - rad,center[1] + rad)] #4
        cord = np.zeros((self.nH+1,3))
        space = self.height[1] - self.height[0]
        for i,h in enumerate(self.height):
            for point in basePoint:
                cord[i,0] = point[0]
                cord[i,1] = point[1]
                cord[i,2] = h -space/2
        for point in basePoint:
            cord[self.nH,0] = point[0]
            cord[self.nH,1] = point[1]
            cord[self.nH,2] = self.height[-1] + space / 2

        strBuffer += ' '.join(map(lambda val:str(val),cord[:,0]))
        strBuffer += '\n'
        strBuffer += ' '.join(map(lambda val:str(val),cord[:,1]))
        strBuffer += '\n'
        strBuffer += ' '.join(map(lambda val:str(val),cord[:,2]))
        strBuffer += '\n'

        # print vars
        vars = np.zeros(self.nH)
        for i,temperatures in enumerate(self.T):
           vars[i] = temperatures.mean()
        strBuffer += ' '.join(map(lambda val:str(val),vars))
        strBuffer += '\n'
        # print connectivities
        basePoint = [1,2,3,4,5,6,7,8]
        conn = np.zeros((self.nH,8))
        for h in range(0,self.nH):
            for i in range(0,8):
                conn[h,i] = basePoint[i] + 4 * h
        for line in conn:
            strBuffer += ' '.join(map(lambda val:str(val),line))
            strBuffer += '\n'

        return strBuffer
