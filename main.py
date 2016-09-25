# first CXY Core Melt Program
# 2016.7.22
from mpi4py import MPI
import CMTypes as Types
import utility as uti
import numpy as np
import Sim as simulator
import initializer as initor
import sys
import re
import petsc4py
petsc4py.init(sys.argv)
comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
my_size = comm.Get_size()

def build_rod_units(nr, nz, filename):
    uti.mpi_print('%s', 'building rods...', my_rank )
    # type: (int, int, str) -> Types.RodUnit
    rods = []
    pattern = re.compile(r'\S+')
    _id = 0;
    for line in open(filename, 'r'):
        _list = pattern.findall(line)
        # should check?
        newRod = Types.RodUnit(_id, nz, nr-10, nr,
                               (float(_list[0]), float(_list[1])),              # position
                               (int(_list[2]), int(_list[3]), int(_list[4])),   # hang lie zu
                                0.00836/2, 0.0095/2,                            # radious
                                5768                                            #material stuff: gap heat transfer
                               )
        _id  += 1
        rods.append(newRod)


    # config rod status
    grayAssemblyList = [4,20,31,33]
    blackAssemblyList = [2,5,9,11,13,16,18,22,25,27,29,35,40,42,45,47]
    center = [(9, 9)]
    specialPosi = [(3, 6), (3, 9),(3,12),(4,4),(4,14),(6,3),(6,6),(6,9),
                   (6,12),(6,15),(9,3),(9,6),(9,12),(9,15),
                   (12,3),(12,6),(12,9),(12,12),(12,15),
                   (14,4),(14,14),(15,6),(15,9),(15,12),
                   ]
    for rod in rods:
        iAss = rod.address[2]
        iRowCol = (rod.address[0], rod.address[1])
        if iAss in grayAssemblyList:
            if iRowCol in center:
                rod.type = Types.RodType.empty
            elif iRowCol in specialPosi:
                rod.type = Types.RodType.gray
            else:
                rod.type = Types.RodType.fuel
        elif iAss in blackAssemblyList:
            if iRowCol in center:
                rod.type = Types.RodType.empty
            elif iRowCol in specialPosi:
                rod.type = Types.RodType.black
            else:
                rod.type = Types.RodType.fuel
        else:
            if iRowCol in center:
                rod.type = Types.RodType.empty
            elif iRowCol in specialPosi:
                rod.type = Types.RodType.empty
            else:
                rod.type = Types.RodType.fuel

    # config neighbour
    rodMap = {}
    for rod in rods:
        assert rodMap.get(rod.address) is None
        rodMap[rod.address] = rod

    def findRod(add, myPosi):
        # type: (tuple, np.array) -> Types.RodUnit
        add = list(add)
        assert isinstance(myPosi, np.ndarray)
        CRITICAL = 0.2**2 #assembly length 0.214
        MAX_COLROW = 17
        MAX_ASSEMBLY =52
        neighbourRod = rodMap.get(tuple(add))
        if neighbourRod is not None:
            # rod in the same assembly
            pass
        else:
            on_bound = False
            icol = add[0]
            icol = (icol-1) % MAX_COLROW + 1
            irow = add[1]
            irow = (irow-1) % MAX_COLROW + 1
            if icol != add[0]:
                on_bound = True
                add[0] = icol
            if irow != add[1]:
                on_bound = True
                add[1] = irow

            if on_bound :
                dis = {}
                for iAss in range(1, MAX_ASSEMBLY + 1):
                    add[2] = iAss
                    thisrod = rodMap.get(tuple(add))
                    if thisrod is None:
                        continue
                    distance = np.sum((thisrod.position - myPosi) ** 2)
                    dis[iAss] = distance

                iAss, _min = min(dis.items(),key=lambda e:e[1])
                if _min < CRITICAL:
                    add[2] = iAss
                    assert rodMap.get(tuple(add)) is not None
                    neighbourRod =  rodMap[tuple(add)]

        if neighbourRod is not None:
            if neighbourRod.type is Types.RodType.empty:
                neighbourRod = None

        return neighbourRod

        #if neighbourRod is None:
        #    print 'cant find rod'
        #else:
        #    print 'rod founded'
        #return neighbourRod

    for rod in rods:
        iRowCol = rod.address
        rod.neighbour['xy+'] = findRod((iRowCol[0], iRowCol[1] + 1, iRowCol[2]), rod.position)
        rod.neighbour['x+y+'] = findRod((iRowCol[0] + 1, iRowCol[1] + 1, iRowCol[2]), rod.position)
        rod.neighbour['x+y'] = findRod((iRowCol[0] + 1, iRowCol[1], iRowCol[2]), rod.position)
        rod.neighbour['x+y-'] = findRod((iRowCol[0] + 1, iRowCol[1] - 1, iRowCol[2]), rod.position)
        rod.neighbour['xy-'] = findRod((iRowCol[0], iRowCol[1] - 1, iRowCol[2]), rod.position)
        rod.neighbour['x-y-'] = findRod((iRowCol[0] - 1, iRowCol[1] - 1, iRowCol[2]), rod.position)
        rod.neighbour['x-y'] = findRod((iRowCol[0] - 1, iRowCol[1], iRowCol[2]), rod.position)
        rod.neighbour['x-y+'] = findRod((iRowCol[0] - 1, iRowCol[1] + 1, iRowCol[2]), rod.position)

    return rods, rodMap


def init_heat_generation_rate(rods,rodsMap, nz, coreHeight, filename):
    uti.mpi_print('%s', 'initialing heat rate ...', my_rank )
    f = open(filename, 'r')
    # radial_distribution
    distribution = []
    f.readline()
    pattern = re.compile(r'\S+')
    patternEnd = re.compile(r'\$AxialPower\s*')
    for line in f:
        if patternEnd.match(line):
            break
        _list = pattern.findall(line)
        distribution.append(float(_list[1]))
    assert len(distribution) == 52
    distribution = np.array(distribution)
    distribution /= distribution.sum()
    distribution /= 4 # quarter core considered
    for add,rod in rodsMap.items():
        iAss = add[2]
        rod.radialPowerFactor = distribution[iAss-1] / (17*17) #per rod
    # axial distribution
    height = []
    distribution = []
    patternEnd = re.compile(r'\$WaterHistory\s*')
    for line in f:
        if patternEnd.match(line):
            break
        _list = pattern.findall(line)
        height.append(float(_list[0]) )
        distribution.append(float(_list[1]))
    height = np.array(height)
    distribution = np.array(distribution)
    space = coreHeight / nz
    hgrid = np.linspace(0. + space, coreHeight - space, nz)
    axialDistribution = np.interp(hgrid, height, distribution)
    axialDistribution /= axialDistribution.sum()
    for rod in rods:
        rod.axialPowerFactor = axialDistribution
        rod.height = hgrid
    del height
    del distribution

    # water history
    time = []
    water = []
    patternEnd = re.compile(r'\$PowerHistory\s*')
    for line in f:
        if patternEnd.match(line):
            break
        _list = pattern.findall(line)
        time.append(float(_list[0]))
        water.append(float(_list[1]))
    Types.PressureVessle.waterHistory = np.array((time, water))
    del water

    # power history
    time = []
    power = []
    for line in f:
        _list = pattern.findall(line)
        time.append(float(_list[0]))
        power.append(float(_list[1])* 3.150e9) # decay heat power: 3150e6
    Types.PressureVessle.powerHistory = np.array((time, power))
    del time
    del power
    # init other stuff
    for rod in rods:
        assert isinstance(rod,Types.RodUnit)
        if rod.type is Types.RodType.fuel:
            rInSpace = rod.inRadious / rod.nRin
            rOutSpace = (rod.radious - rod.inRadious) / (rod.nR - rod.nRin)
            rInGrid = np.linspace(0.+rInSpace/2, rod.inRadious - rInSpace/2, rod.nRin)
            rOutGrid = np.linspace(rod.inRadious + rOutSpace/2, rod.radious - rOutSpace/2, rod.nR - rod.nRin)
            rod.rgrid = np.hstack((rInGrid,rOutGrid))
        else:
            rspace = rod.radious/rod.nR
            rod.rgrid  = np.linspace(0.+rspace, rod.radious - rspace, rod.nR)

def clean_rod_units(rods,rodsMap):
    cleaned_rod = filter(lambda rod: rod.type is not Types.RodType.empty, rods)
    return cleaned_rod, None
if __name__ == "__main__":
    nz = 100
    nr = 30
    coreHeight = 3.657
    mask = {
        0 : [1,2,3,4,5,6,7,8,
               9,10,11,12,13,14,15,16],
        1 : [ 17, 18 ,19, 20, 21, 22, 23, 24, 
                25, 26, 27, 28, 29, 30, 31],
        2 : [32, 33, 34, 35, 36, 37, 38, 
              39, 40,41 ,42 ,43, 44],
        3: [45, 46, 47, 48, 49, 50,
              51, 52,]

    }
    assert len(mask) ==  my_size
    rodUnits, rodsMap = build_rod_units(nr, nz, 'rod_position.dat')
    init_heat_generation_rate(rodUnits,rodsMap, nz, coreHeight,'heat_rate.dat')
    rodUnits, rodsMap = clean_rod_units(rodUnits,rodsMap)
    simulator.config_material(rodUnits)
    initor.set_initial(rodUnits,0.0 ,10,373) #start time , delat T, Tfluid

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    rodUnits, bound_array = initor.set_mask(rank, rodUnits, mask)
    
    fuelTemplate, blackTemplate, rhs = initor.initPetscTemplate(rodUnits)
    simulator.installPETScTemplate(fuelTemplate, blackTemplate, rhs)
    #simulator.ready_to_solve(rodUnits)
    #start solve
    comm.barrier()
    simulator.start(rodUnits, bound_array, mask, 5000, 1)
    print 'done'

