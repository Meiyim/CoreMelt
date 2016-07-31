import CMTypes as Types
import Sim as simulator
from petsc4py import PETSc
import math
import numpy as np


def set_initial(rods,tstart,deltaT,Tf):
    # type: (list,float,float, float) -> None
    print 'allocate & initialing temperature field'
    assert isinstance(rods,list)
    Types.PressureVessle.currentTime = tstart
    Types.PressureVessle.timePush(0)
    nowWater, nowPower = Types.PressureVessle.now()
    for rod in rods:
        assert isinstance(rod,Types.RodUnit)
        rodPower = rod.radialPowerFactor * nowPower
        rod.qsource = rod.axialPowerFactor * rodPower
        print rod.qsource
        Trows = []
        vspace = ( rod.height[-1] - rod.height[0] ) / (rod.height.shape[0]-1)
        #print 'donging rod %s' % str(rod.address)
        for ih in range(0,rod.nH):
            L = rod.height[ih] + 1 # TODO L should NOT be ZERO
            q = rod.qsource[ih] / vspace
            h = simulator.calcBoilHeatTransferRate(simulator.calGr(deltaT,L),1,1,L) #assuming deltaT == 10
            Tco = Tf + q / (math.pi * 2 * rod.radious * h)
            Tci = Tco + q / (2 * math.pi * rod.material.lamdaOut) * math.log(rod.radious/rod.inRadious)
            To  = Tci + q / (math.pi * rod.inRadious * 2 *rod.gapHeatRate )
            print '4 key temp for rod %d-%d-%d: flux: %f, Tco: %f, Tci: %f, To:%f Tf: %f' % (rod.address + (q,Tco,Tci,To,Tf) )
            tIn = np.linspace(To,To,rod.nRin)
            tOut= np.linspace(Tci,Tco,rod.nR - rod.nRin)
            Trows.append( np.hstack((tIn,tOut)) )
        Trows = tuple(Trows)
        rod.T = np.vstack(Trows)
        rod.qbound = np.zeros(rod.nH)
        rod.qup = np.zeros(rod.nR)
        rod.qdown = np.zeros(rod.nR)
        rod.heatCoef = np.zeros(rod.nH)
        assert rod.T.shape == (rod.nH,rod.nR)


def initPetscTemplate(rods):
    #type: (list)->Types.PETScWrapper,Types.PETScWrapper
    fuleRodSample = None
    blackRodSample = None
    for rod in rods:
        if rod.type is Types.RodType.fuel:
            fuleRodSample = rod
        else:
            blackRodSample = rod
            break
    assert isinstance(fuleRodSample,Types.RodUnit)
    assert isinstance(blackRodSample,Types.RodUnit)
    assert fuleRodSample.type is Types.RodType.fuel
    fueltemp = Types.PETScWrapper(fuleRodSample.nH*fuleRodSample.nR,fuleRodSample.nR,fuleRodSample.nH)
    blacktemp = Types.PETScWrapper(blackRodSample.nH*blackRodSample.nR,blackRodSample.nR,blackRodSample.nH)
    fueltemp.fillTemplatefuel(fuleRodSample.nRin, fuleRodSample.material.lamdaIn, fuleRodSample.material.lamdaOut, fuleRodSample.inRadious,
                              fuleRodSample.radious, fuleRodSample.gapHeatRate, fuleRodSample.height[1]-fuleRodSample.height[0], fuleRodSample.rgrid)
    blacktemp.fillTemplateBlack(blackRodSample.material.lamdaIn,blackRodSample.radious,blackRodSample.height[1]-blackRodSample.height[0], blackRodSample.rgrid)

    return fueltemp, blacktemp, PETSc.Vec().createSeq(fuleRodSample.nH*fuleRodSample.nR)



