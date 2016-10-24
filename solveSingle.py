import sys
import math
import CMTypes as Types
import numpy as np
import petsc4py
from petsc4py import PETSc
petsc4py.init(sys.argv)

def calc_fuel_temperature(rod,Tf,dt, wrapper,verbose=False): #currently  only 2
    #type: (Types.RodUnit) -> None
    A    = wrapper.getMat()
    b    = PETSc.Vec().createSeq(rod.nR * rod.nH)
    b.zeroEntries()
    xsol = PETSc.Vec().createSeq(rod.nR * rod.nH)
    for j in xrange(0, rod.nH):
        for i in xrange(0, rod.nR):
            row = j*rod.nR + i
            xsol.setValue(row,rod.T[j,i])
    # out bound --- 3rd condition
    outsideAera = math.pi * 2 * rod.radious * ( rod.height[1] - rod.height[0] )
    for j in xrange(0, rod.nH):
        i = rod.nR - 1
        row = j*rod.nR + i
        b.setValue(row, rod.heatCoef[j] * Tf * outsideAera  - rod.qbound[j] ) # qbound move to right hand side ...
        A.setValue(row, row, rod.heatCoef[j] * outsideAera, addv = True)
    #upbound --- 2nd condition
    for i in xrange(0, rod.nR):
        j = rod.nH - 1
        row = j*rod.nR + i
        b.setValue(row, 0. - rod.qup[i] ,addv = True) # qflux move to right hand side
    #down bound --- 2nd condition
    for i in xrange(0, rod.nR):
        j = 0
        row = j*rod.nR + i
        b.setValue(row, 0. - rod.qdown[i], addv = True) # q flux move to right hand side
    #set body source
    spaceIn = rod.rgrid[1] - rod.rgrid[0]
    spaceOut = rod.rgrid[-1] - rod.rgrid[-2]
    hspace  = rod.height[1] - rod.height[0]
    for j in xrange(0, rod.nH):
        for i in xrange(0, rod.nRin):
            volumn = rod.rgrid[i] * 2 * math.pi * spaceIn * hspace
            heatGenerationRate = volumn * rod.qsource[j]
            #print heatGenerationRate
            transidentCoef = volumn * rod.material.cpIn * rod.material.rouIn / dt
            row = j*rod.nR + i
            b.setValue(row, heatGenerationRate + transidentCoef * rod.T[j,i], addv = True)  # heatGenerationRate > 0 on right hand side
            A.setValue(row,row, transidentCoef, addv = True )#transient  term
    for j in xrange(0, rod.nH):
        for i in xrange(rod.nRin, rod.nR):
            volumn = rod.rgrid[i] * 2 * math.pi * spaceOut * hspace
            transidentCoef = volumn * rod.material.cpOut * rod.material.rouOut / dt
            row = j * rod.nR + i
            b.setValue(row, transidentCoef * rod.T[j,i], addv = True)
            A.setValue(row, row, transidentCoef, addv =  True)

    A.assemblyBegin()
    b.assemblyBegin()
    xsol.assemblyBegin()
    A.assemblyEnd()
    b.assemblyEnd()
    xsol.assemblyEnd()

    petsc_ksp = PETSc.KSP().create()
    petsc_ksp.setType(PETSc.KSP.Type.CG)
    pc = petsc_ksp.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    petsc_ksp.setPC(pc)

    petsc_ksp.setInitialGuessNonzero(False)
    petsc_ksp.setOperators(A)
    petsc_ksp.setTolerances(rtol=1.e-6,max_it=1000)
    petsc_ksp.solve(b,xsol)

    if petsc_ksp.getConvergedReason() < 0:
        raise ValueError, 'iteration not converged in %d-%d-%d' % rod.address
    else:
        if verbose:
            print 'iteration converged in %d step' % petsc_ksp.getIterationNumber()

    raw_arr = xsol.getArray()
    for row,val in enumerate(raw_arr):
        j = row / rod.nR
        i = row % rod.nR
        rod.T[j,i] = val
    if verbose:
        sys.stdout.write('rod [%d %d %d] T center %f, fuelOut %f, cladOut %f, qbound:  %f, qline % f, headCoef %f, mass %f\n' % (rod.address + rod.getSummary()))
    return rod.T[rod.nH/2, :]

def solveByEmpirical(Tf, hConv, hGap, lamdaIn, lamdaOut, nRin, nR,rin, r, qsource ):
    qline = math.pi * rin ** 2 * qsource
    Tco = Tf + qline / (math.pi * 2 * r * hConv)
    Tci = Tco + qline / (2 * math.pi * lamdaOut) * math.log(r / rin)
    To = Tci + qline/(math.pi * rin * 2 * hGap)
    return np.hstack((np.linspace(To, To, nRin), np.linspace(Tci, Tco, nR - nRin) ))

def main():
    gapHeatRate = 10.0
    qsource = 10000.0
    hside = 10000.0
    nrin = 10
    nr = 20
    nh = 10
    r = 2.0
    rin = 1.0
    h = 1.0
    rod = Types.RodUnit(0, nh, nrin, nr, (0., 0), (0, 0, 0), rin, r, gapHeatRate)
    mat = Types.MaterialProterty('test', 10.0, 10.0, 1000., 1000., 4100, 4100, 10000, 10000)
    rod.material = mat

    rod.qup = np.zeros(nr)
    rod.qdown = np.zeros(nr)
    rod.qbound = np.zeros(nh)
    rod.heatCoef = np.zeros(nh)
    rod.heatCoef[:] = hside
    rod.T = np.zeros((nh, nr))
    rod.qsource = np.zeros(nh)
    rod.qsource[:] = qsource

    #geometry config
    rInSpace = rod.inRadious / rod.nRin
    rOutSpace = (rod.radious - rod.inRadious) / (rod.nR - rod.nRin)
    rInGrid = np.linspace(0.+rInSpace/2, rod.inRadious - rInSpace/2, rod.nRin)
    rOutGrid = np.linspace(rod.inRadious + rOutSpace/2, rod.radious - rOutSpace/2, rod.nR - rod.nRin)
    rod.rgrid = np.hstack((rInGrid,rOutGrid))
    rod.height = np.linspace(0., h, nh)

    wrapper = Types.PETScWrapper(nr * nh, nr, nh)
    wrapper.fillTemplatefuel(nrin, rod.material.lamdaIn,
                                   rod.material.lamdaOut,
                                   rod.nRin, rod.nR, 
                                   rod.gapHeatRate, 
                                   rod.height[1] - rod.height[0],
                                   rod.rgrid )

    print calc_fuel_temperature(rod, 373, 1e30, wrapper, True)
    print solveByEmpirical(373, hside, gapHeatRate, 10.0, 10.0, rod.nRin, rod.nR, rod.inRadious, rod.radious, qsource)


main()
