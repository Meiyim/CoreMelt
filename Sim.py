import numpy as np
import math
import CMTypes as Types
from petsc4py import PETSc

fuelTemplate  = None # type: Types.PETScWrapper
blackTemplate = None # type: Types.PETScWrapper
petsc_rhs          = None # type: PETSc.Vec
petsc_ksp     = None # type: PETSc.KSP

def installPETScTemplate(fuel,black,b):
    global fuelTemplate,blackTemplate,petsc_rhs,petsc_ksp
    fuelTemplate  = fuel
    blackTemplate = black
    petsc_rhs      = b
    petsc_ksp = PETSc.KSP().create()
    petsc_ksp.setType(PETSc.KSP.Type.CG)
    pc = petsc_ksp.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    petsc_ksp.setPC(pc)


def config_material(rods):
    #type(Types.RodUnits) -> ddNone
    print 'configuring material'
    for rod in rods:
        if rod.type is Types.RodType.black or rod.type is Types.RodType.gray:
            rod.material = Types.MaterialProterty('gray/black',25,25,7020,7020,835,835,1600,1600)
            rod.radialPowerFactor = 0.0 #black & gray rod cannot generate heat...
        elif rod.type is Types.RodType.fuel:
            mixratio1 = 92.35 /(92.35 + 2.606)
            mixratio2 = 2.606 /(92.35 + 2.606)
            rod.material = Types.MaterialProterty('fuel',
            5.3*mixratio1+25*mixratio2,  2.09,
            8740*mixratio1+7020*mixratio2, 5990,
            535*mixratio1+835*mixratio2, 645,
            3113*mixratio1+1600*mixratio2,2911)
        else:
            assert False

def calGr(dT,L):
    beta = 0.434
    g = 9.8
    niu = 2.82e-4
    rou = 985.4
    return g*beta*dT*(L**3) /((niu/rou)**2)

def calcBoilHeatTransferRate(Gr,Prf,Prw,L):
    mul = Gr*Prf
    lamda = 0.683
    Nu = 0.0
    if mul<10e3:
        print 'Gr, Pr didnt confront Correlation\n Pr * Gr == %f!' % mul
        assert False
    if 10e3 < mul < 10e10:
        Nu = 0.6 * (mul)**0.25 * (Prf/Prw) ** (0.25)
    if mul >=10e10:
        Nu = 0.15 * (mul)**0.333 * (Prf/Prw) ** (0.25)
    return  Nu * lamda / L



def ready_to_solve(rods):
    #type: (Types.RodUnits) -> None
    #print to tecplot
    print 'preparing for solve'
    _file = open('tec.dat','w')
    _file.write('variables="x","y","z","t"\n')
    for rod in rods:
        buffer = rod.getTecplotZone()
        _file.write(buffer)
    _file.close()
    print 'ready to solve'
    return


def save_restart_file(rods):
    #type: (list) -> None
    restartFile = open('restart','wb')
    position = []
    T = []
    qb = []
    qu = []
    qd = []
    hc = []
    qs = []
    h = []
    rg = []
    ap = []
    for rod in rods:
        assert isinstance(rod,Types.RodUnit)
        rod.saveToFile(restartFile)
        position.append(rod.position)
        T.append(rod.T)
        qb.append(rod.qbound)
        qu.append(rod.qup)
        qd.append(rod.qdown)
        hc.append(rod.heatCoef)
        qs.append(rod.qsource)
        h.append(rod.height)
        rg.append(rod.rgrid)
        ap.append(rod.axialPowerFactor)
    position = np.vstack(position)
    T        = np.vstack(T)
    qb       = np.vstack(qb)
    qu       = np.vstack(qu)
    qd       = np.vstack(qd)
    hc       = np.vstack(hc)
    qs       = np.vstack(qs)
    h        = np.vstack(h)
    rg       = np.vstack(rg)
    ap       = np.vstack(ap)
    np.savez('restart.npz',pos=position,T=T,qb=qb, qu = qu, qd = qd,qs=qs, hc = hc, h=h, rg=rg, ap=ap)


def calc_rod_bound(rod,Tf):
    #type: (Types.RodUnit, float) -> None
    SIGMA = 1.0e-12
    EPSILONG = 1.0e-3
    RADIOUS = rod.radious
    SPACE = 0.016
    ROD_SPACE = rod.height[1] - rod.height[0]
    Xangle_Area = {
                    'xy+'   : (0.145, RADIOUS*2                     * ROD_SPACE ),
                    'x+y+'  : (0.105, (SPACE-RADIOUS)*math.sqrt(2)  * ROD_SPACE ),
                    'x+y'   : (0.145, RADIOUS*2                     * ROD_SPACE ),
                    'x+y-'  : (0.105, (SPACE-RADIOUS)*math.sqrt(2)  * ROD_SPACE ),
                    'xy-'   : (0.145, RADIOUS*2                     * ROD_SPACE ),
                    'x-y-'  : (0.105, (SPACE-RADIOUS)*math.sqrt(2)  * ROD_SPACE ),
                    'x-y'   : (0.145, RADIOUS*2                     * ROD_SPACE ),
                    'x-y+'  : (0.105, (SPACE-RADIOUS)*math.sqrt(2)  * ROD_SPACE ),
                  }

    for ih,L in enumerate(rod.height):
        L+=1.0 #TODO to prevent zero height ...
        selfT = rod.T[ih,-1] #outside wall Temp ... no extrapolation now
        deltaT = selfT - Tf
        #print deltaT
        h = calcBoilHeatTransferRate(calGr(deltaT,L),1,1,L) #assuming deltaT == 10
        rod.heatCoef = h
        #qConvection = h * (selfT - Tf)
        qRadiation = 0.0
        for dir,neighbourRod in rod.neighbour.items():
            qRadiation = Xangle_Area[dir][1] * Xangle_Area[dir][0] * (selfT - neighbourRod.T[ih,-1]) * SIGMA * EPSILONG
        rod.qbound[ih] = qRadiation

    for ir, R in enumerate(rod.rgrid): # currently adiabetic up/down
        rod.qup[ir] = 0.
        rod.qdown[ir] = 0.


def calc_fuel_temperature(rod,Tf): #currently  only 2
    #type: (Types.RodUnit) -> None
    A    = fuelTemplate.getMat()
    b    = petsc_rhs.duplicate()
    b.zeroEntries()
    xsol = petsc_rhs.duplicate()
    for j in xrange(0., rod.nH):
        for i in xrange(0., rod.nR):
            row = j*rod.nR + i
            xsol.setValue(row,rod.T[j,i])

    # out bound --- 3rd condition
    for j in xrange(0, rod.nH):
        i = rod.nR - 1
        row = j*rod.nR + i
        b.setValue(row, rod.heatCoef[j] * Tf  - rod.qbound[j] ) # qbound move to right hand side ...
        A.setValue(row, row, rod.heatCoef[j], addv = True)
    #upbound --- 2nd condition
    for i in xrange(0, rod.nR):
        j = rod.nH - 1
        row = j*rod.nR + i
        b.setValue(row, 0. - rod.qup[j] ) # qflux move to right hand side
    #down bound --- 2nd condition
    for i in xrange(0, rod.nR):
        j = 0
        row = j*rod.nR + i
        b.setValue(row, 0. - rod.qdown[j] ) # q flux move to right hand side
    #set body source
    spaceIn = rod.rgrid[1] - rod.rgrid[0]
    hspace  = rod.height[1] - rod.height[0]
    for j in xrange(0, rod.nH):
        for i in xrange(0, rod.nRin):
            volumn = rod.rgrid[i] * 2 * math.pi * spaceIn * hspace
            heatGenerationRate = volumn * rod.qsource[i]
            row = j*rod.nR + i
            b.setValue(row, heatGenerationRate, addv = True)  # heatGenerationRate > 0 on right hand side
    A.assemblyBegin()
    b.assemblyBegin()
    xsol.assemblyBegin()
    A.assemblyEnd()
    b.assemblyEnd()
    xsol.assemblyEnd()

    petsc_ksp.setOperators(A)
    petsc_ksp.setTolerences(rtol=1.e-6,max_it=1000)
    petsc_ksp.solve(b,xsol)
    raw_arr = xsol.getArray()
    print 'petsc solve done for rod: %d-%d-%d -average T: %f' % (rod.address + (sum(raw_arr)/len(raw_arr),))
    for row,val in enumerate(raw_arr):
        j = row / rod.nR
        i = row % rod.nR
        rod.T[j,i] = val






def calc_other_temperature(rod,Tf): #currently  only 2
    #type: (Types.RodUnit) -> None
    A    = blackTemplate.getMat()
    b    = petsc_rhs.duplicate()
    b.zeroEntries()
    xsol = petsc_rhs.duplicate()
    for j in xrange(0., rod.nH):
        for i in xrange(0., rod.nR):
            row = j*rod.nR + i
            xsol.setValue(row,rod.T[j,i])

    # out bound --- 3rd condition
    for j in xrange(0, rod.nH):
        i = rod.nR - 1
        row = j*rod.nR + i
        b.setValue(row, rod.heatCoef[j] * Tf  - rod.qbound[j] ) # qbound move to right hand side ...
        A.setValue(row, row, rod.heatCoef[j], addv = True)
    #upbound --- 2nd condition
    for i in xrange(0, rod.nR):
        j = rod.nH - 1
        row = j*rod.nR + i
        b.setValue(row, 0. - rod.qup[j] ) # qflux move to right hand side
    #down bound --- 2nd condition
    for i in xrange(0, rod.nR):
        j = 0
        row = j*rod.nR + i
        b.setValue(row, 0. - rod.qdown[j] ) # q flux move to right hand side

    A.assemblyBegin()
    b.assemblyBegin()
    xsol.assemblyBegin()
    A.assemblyEnd()
    b.assemblyEnd()
    xsol.assemblyEnd()

    petsc_ksp.setOperators(A)
    petsc_ksp.setTolerences(rtol=1.e-6,max_it=1000)
    petsc_ksp.solve(b,xsol)
    raw_arr = xsol.getArray()
    print 'petsc solve done for rod: %d-%d-%d -average T: %f' % (rod.address + (sum(raw_arr)/len(raw_arr),))
    for row,val in enumerate(raw_arr):
        j = row / rod.nR
        i = row % rod.nR
        rod.T[j,i] = val

    ndim = rod.nH

def start(rods,timeLimit, dt):
    #type: (list,float,float) -> None
    print 'starting simulation'

    while Types.PressureVessle.currentTime <= timeLimit:
        print 'solving time %f' % Types.PressureVessle.currentTime
        print 'saving...'
        save_restart_file(rods)
        Types.PressureVessle.timePush(dt)
        nowWater, nowPower = Types.PressureVessle.now()
        print 'calculating heat souce and temp bound'
        for rod in rods:
            h = (rod.height[1]-rod.height[0]) * rod.height.shape[0]
            volumn = math.pi * rod.radious**2 * h
            rodPower = rod.radialPowerFactor * nowPower
            rod.qsource = rod.axialPowerFactor * rodPower / volumn
            calc_rod_bound(rod,373) #last parameter is fluid temp
            if rod.type is Types.RodType.fuel:
                calc_fuel_temperature(rod,373)   #a PETSc impementation
            else:
                calc_other_temperature(rod,373)  #a PETSc impementation

        print 'buillding matrix...'
        print 'solving matrix...'

    print  'similation done'

