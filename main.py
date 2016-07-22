# first CXY Core Melt Program
# 2016.7.22
import CMTypes as Types
import numpy as np
import Sim as simulator
import re


def build_rod_units(nr, nz, filename):
    # type: (int, int, str) -> Types.RodUnit
    def id_generator():
        ret = 0
        while True:
            yield ret
            ret += 1

    rods = []
    pattern = re.compile(r'\S*')
    for line in open(filename, 'r'):
        _list = pattern.findall(line)
        _gen = id_generator()
        # should check?
        newRod = Types.RodUnit(_gen.next(), nr, nz, (_list[0], _list[1]), (_list[2], _list[3], _list[4]))
        rods.append(newRod)

    # config rod status
    fuelAssemblyList = []
    grayAssemblyList = []
    blackAssemblyList = []
    center = [(9, 9)]
    specialPosi = [(1, 1), (2, 2)]
    for rod in rods:
        iAss = rod.address[2]
        iRowCol = (rod.address[0], rod.address[1])
        if iAss in fuelAssemblyList:
            if iRowCol in center:
                rod.status = Types.RodType.empty
            elif iRowCol in specialPosi:
                rod.status = Types.RodType.empty
            else:
                rod.status = Types.RodType.fuel
        elif iAss in grayAssemblyList
            if iRowCol in center:
                rod.status = Types.RodType.empty
            elif iRowCol in specialPosi:
                rod.status = Types.RodType.gray
            else:
                rod.status = Types.RodType.fuel
        elif iAss in blackAssemblyList
            if iRowCol in center:
                rod.status = Types.RodType.empty
            elif iRowCol in specialPosi:
                rod.status = Types.RodType.black
            else:
                rod.status = Types.RodType.fuel
        else:
            assert False

    # config neighbour
    rodMap = {}
    for rod in rods:
        rodMap[rod.address] = rod

    def findRod(add, myPosi):
        # type: (tuple, np.array) -> Types.RodUnit
        add = list(add)
        assert isinstance(myPosi, np.array)
        CRITICAL = 10.
        ID_END = 17
        ID_BEG = 1
        neighbourRod = rodMap.get(add)
        if neighbourRod is not None:
            # rod in the same assembly
            pass
        else:
            if add[0] == ID_BEG:
                add[0] = ID_END
            if add[1] == ID_BEG:
                add[1] = ID_END
            if add[0] == ID_END:
                add[0] = ID_BEG
            if add[1] == ID_END:
                add[1] = ID_BEG
            dis = []
            for iAss in range(ID_BEG, ID_END + 1):
                add[2] = iAss
                thisrod = neighbourRod.get(add)
                assert thisrod is not None
                distance = np.sum((thisrod.position - myPosi) ** 2)
                dis.append(distance)

            _min = min(dis)
            if _min < CRITICAL:
                iAss = dis.index(_min)
                add[2] = iAss
                assert rodMap.get(tuple(add)) is not None
                neighbourRod =  rodMap[tuple(add)]

        if neighbourRod.type is Types.RodType.empty:
            neighbourRod = None
        return neighbourRod

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
    f = open(filename, 'r')
    # radial_distribution
    distribution = []
    f.readline()
    pattern = re.compile(r'\S*')
    for line in f:
        if line == '$AxialPower':
            break
        _list = pattern.findall(line)
        distribution.append(_list[1])
    assert len(distribution) == 52
    distribution = np.array(distribution)
    distribution /= distribution.max()
    for add,rod in rodsMap.iter():
        iAss = add[2]
        rod.radialPowerDistribution = distribution[iAss]
    # axial distribution
    height = []
    distribution = []
    for line in f:
        if line == '$WaterHistory':
            break
        _list = pattern.findall(line)
        height.append(_list[0])
        distribution.append(_list[1])
    height = np.array(height)
    distribution = np.array(distribution)
    hgrid = np.linspace(0., coreHeight, nz)
    axialDistribution = np.interp(hgrid, height, distribution)
    for rod in rods:
        rod.axialPowwerFactor = axialDistribution
    del height
    del distribution

    # water history
    time = []
    water = []
    for line in f:
        if line == '$PowerHistory':
            break
        _list = pattern.findall(line)
        time.append(_list[0])
        water.append(_list[1])
    Types.PressureVessle.waterHistory = np.array((time, water))
    del water

    # power history
    time = []
    power = []
    for line in f:
        _list = pattern.findall(line)
        time.append(_list[0])
        power.append(_list[1])
    Types.PressureVessle.powerHistory = np.array((time, power))
    del time
    del power

def clean_rod_units(rods,rodsMap):
    cleaned_rod = filter(lambda rod: rod.type is Types.RodType.empty, rods)
    return cleaned_rod, None

if __name__ == "__main__":
    nz = 100
    nr = 30
    coreHeight = 3657
    rodUnits, rodsMap = build_rod_units(nr, nz, 'rod_position.dat')
    init_heat_generation_rate(rodUnits,rodsMap, nz, coreHeight,'heat_rate.dat')
    rodUnits, rodsMap = clean_rod_units(rodUnits,rodsMap)

    simulator.start()
    print 'done'

