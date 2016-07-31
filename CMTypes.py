import numpy as np
import math
import struct
from petsc4py import PETSc


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


class PETScWrapper:
    def __init__(self,n,imax,jmax):
        self.A = PETSc.Mat().createAIJ([n,n],nnz = 5)
        self.n = n
        self.imax = imax
        self.jmax = jmax

    def getMat(self):
        return self.A.duplicate(copy=True)

    def to1d(self,i,j):
        if i==0 or i == self.imax:
            return None
        if j==0 or j == self.jmax:
            return None
        return j*self.imax+i

    def fillTemplateBlack(self,lamda,rout,hspace,rGrid):
        #type (int,int,int,float,float,float,float,float,float)
        imax = self.imax
        jmax = self.jmax
        rspace = rGrid[0] - rGrid[1]

        def giveVal(idir,i,j):
            A = 0.0
            lam = 0.0
            dis = 0.0
            if idir == 0 and idir == 2: #north and south
                dis = hspace
                A   = math.pi * 2 * rGrid[i] * rspace
                lam = lamda
            elif idir == 1: #east
                dis = rGrid[j+1] - rGrid[j]
                A   = math.pi * 2 * rGrid[i] * hspace
                lam = lamda
            elif idir == 3: #west
                dis = rGrid[j] - rGrid[j-1]
                A   = math.pi * 2 * rGrid[i] * hspace
                lam = lamda
            return 0. - lam * A / dis

        for j in xrange(0,jmax):
            for i in xrange(0,imax):
                row = self.to1d(i,j)
                cols=[None,None,None,None] # north, east, south, west
                cols[0] = self.to1d(i,j+1)
                cols[1] = self.to1d(i+1,j)
                cols[2] = self.to1d(i,j-1)
                cols[3] = self.to1d(i-1,j)
                vals=[None,None,None,None] # north ,east, south, west
                vals[0] = giveVal(0,i,j)
                vals[1] = giveVal(1,i,j)
                vals[2] = giveVal(2,i,j)
                vals[3] = giveVal(3,i,j)
                vals = filter(lambda i,val : val if cols[i] is None else None, vals)
                center  = 0. - reduce(lambda lhs,rhs : 0. if lhs is not None and rhs is not None else lhs + rhs,vals)
                cols = filter(lambda val: val is not None, cols)
                vals = filter(lambda val: val is not None, vals)
                self.A.setValue(row,row,center)       #diagnal
                self.A.setValues([row],cols,vals)     #off-diagnal
        self.A.assemblyBegin()
        self.A.assemblyEnd()

    def fillTemplatefuel(self,ibound,lamdaIn,lamdaOut,rin,rout,h,hspace,rGrid):
        #type (int,int,int,float,float,float,float,float,float)
        imax = self.imax
        jmax = self.jmax
        rInSpace = rGrid[1] - rGrid[0]
        rOutSpace = rGrid[-1] - rGrid[-2]
        def giveVal(idir,i,j):
            if self.to1d(i,j) is None:
                return None
            A = 0.0
            lam = 0.0
            dis = 0.0
            if idir == 0: #north
                dis = hspace
                if i<ibound:
                    lam = lamdaIn
                    A   = math.pi * 2 * rGrid[i] * rInSpace
                else:
                    lam = lamdaOut
                    A   = math.pi * 2 * rGrid[i] * rOutSpace
            elif idir == 1: #east
                dis = rGrid[j+1] - rGrid[j]
                A   = math.pi * 2 * rGrid[i] * hspace
                if i<ibound:
                    lam = lamdaIn
                elif i==ibound-1:
                    lam = h * dis
                else:
                    lam = lamdaOut
            elif idir == 2: #south
                dis = hspace
                if i<ibound:
                    lam = lamdaIn
                    A   = math.pi * 2 * rGrid[i] * rInSpace
                else:
                    lam = lamdaOut
                    A   = math.pi * 2 * rGrid[i] * rOutSpace
            elif idir == 3: #west
                dis = rGrid[j] - rGrid[j-1]
                A   = math.pi * 2 * rGrid[i] * hspace
                if i<ibound:
                    lam = lamdaIn
                elif i==ibound:
                    lam = h * dis
                else:
                    lam = lamdaOut
            return 0. - lam * A / dis

        for i in xrange(0,imax):
            for j in xrange(0,jmax):
                row = self.to1d(i,j)
                cols=[None,None,None,None] # north, east, south, west
                cols[0] = self.to1d(i,j+1)
                cols[1] = self.to1d(i+1,j)
                cols[2] = self.to1d(i,j-1)
                cols[3] = self.to1d(i-1,j)
                vals=[None,None,None,None] # north ,east, south, west
                vals[0] = giveVal(0,i,j)
                vals[1] = giveVal(1,i,j)
                vals[2] = giveVal(2,i,j)
                vals[3] = giveVal(3,i,j)
                vals = filter(lambda i,val : val if cols[i] is None else None, vals)
                center  = 0. - reduce(lambda lhs,rhs : 0. if lhs is not None and rhs is not None else lhs + rhs,vals)
                cols = filter(lambda val: val is not None, cols)
                vals = filter(lambda val: val is not None, vals)
                self.A.setValue(row,row,center)       #diagnal
                self.A.setValues([row],cols,vals)     #off-diagnal
        self.A.assemblyBegin()
        self.A.assemblyEnd()


class RodUnit(object):
    def __init__(self, _id, nh, nrin, nr, pos, add, rin, rout, l):
        assert isinstance(pos, tuple)
        self.position = np.array(pos, dtype=np.float64)
        self.T = None   # type: np.ndarray
        self.heatCoef = None # type: np.ndarray
        self.qbound = None #type:np.ndarray qsouce inflow == Positive, outflow == negative
        self.qup = None    # type: np.ndarray
        self.qdown = None  # type: np.ndarray
        self.qsource = None #type: np.ndarray
        self.height = None #type: np.ndarray
        self.rgrid  = None #type: np.ndarray
        self.axialPowerFactor = None #type: np.ndarray
        self.type = RodType.empty
        self.index = _id
        self.address = add
        self.nH = nh
        self.nRin = nrin
        self.nR = nr
        self.neighbour = {}
        self.radialPowerFactor = 0.0  # main power
        self.inRadious = rin
        self.radious = rout
        self.gapHeatRate = l
        #material
        self.material = None # type: MaterialProterty
        # ksp stuff

    def saveToFile(self,restartFile): #save the non-Numpy propertyies
        def getindex(rod):
            if rod is None:
                return -1
            else:
                return rod.index
        bytes = struct.pack('15i',self.index,
                            self.address[0],self.address[1],self.address[2],
                            self.nH, self.nR, self.nRin,
                            getindex(self.neighbour['xy+']),
                            getindex(self.neighbour['x+y+']),
                            getindex(self.neighbour['x+y']),
                            getindex(self.neighbour['x+y-']),
                            getindex(self.neighbour['xy-']),
                            getindex(self.neighbour['x-y-']),
                            getindex(self.neighbour['x-y']),
                            getindex(self.neighbour['x-y+']),
                            )
        restartFile.write(bytes)
        bytes = struct.pack('10d',self.radialPowerFactor,
                                 self.inRadious,
                                 self.radious,
                                 self.gapHeatRate ,
                                 self.material.lamdaIn,
                                 self.material.lamdaOut,
                                 self.material.rouIn,
                                 self.material.rouOut,
                                 self.material.cpIn,
                                 self.material.cpOut
                            )
        restartFile.write(bytes)


    def getTecplotZone(self):
        strBuffer = 'ZONE N=%d, E=%d, VARLOCATION=([1-3]=NODAL,[4]=CELLCENTERED) DATAPACKING=BLOCK, ZONETYPE=FEBRICK\n' \
                % ((self.nH + 1) * 4 , self.nH)#type: str
        center = self.position
        rad = self.radious / math.sqrt(2)
        # print points
        basePoint = [(center[0] + rad,center[1] + rad), #1
                     (center[0] + rad,center[1] - rad), #2
                     (center[0] - rad,center[1] - rad), #3
                     (center[0] - rad,center[1] + rad)] #4
        cord = np.zeros(((self.nH+1)*4,3))
        space = self.height[1] - self.height[0]
        def indexIncreaser(start,end):
            while start != end:
                yield  start
                start+=1

        gen = indexIncreaser(0,(self.nH+1)*4)
        for i,h in enumerate(self.height):
            for point in basePoint:
                ivert = gen.next()
                cord[ivert,0] = point[0]
                cord[ivert,1] = point[1]
                cord[ivert,2] = h -space/2
        for point in basePoint:
            ivert = gen.next()
            cord[ivert,0] = point[0]
            cord[ivert,1] = point[1]
            cord[ivert,2] = self.height[-1] + space / 2


        strBuffer += ' '.join(map(lambda (i,val):str(val) if i%5!=4 else str(val)+'\n' ,enumerate(cord[:,0])))
        strBuffer += '\n'
        strBuffer += ' '.join(map(lambda (i,val):str(val) if i%5!=4 else str(val)+'\n' ,enumerate(cord[:,1])))
        strBuffer += '\n'
        strBuffer += ' '.join(map(lambda (i,val):str(val) if i%5!=4 else str(val)+'\n' ,enumerate(cord[:,2])))
        strBuffer += '\n'

        # print vars
        vars = np.zeros(self.nH)
        for i,temperatures in enumerate(self.T):
           vars[i] = temperatures.mean()
        strBuffer += ' '.join(map(lambda (i,val):str(val) if i%5!=4 else str(val) + '\n',enumerate(vars)))
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

