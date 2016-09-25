import numpy as np
import math
import time
import sys
import CMTypes as Types
import utility as uti
import initializer as initor
from tqdm import tqdm
from mpi4py import MPI
from petsc4py import PETSc

fuelTemplate  = None # type: Types.PETScWrapper
blackTemplate = None # type: Types.PETScWrapper
petsc_rhs          = None # type: PETSc.Vec
petsc_ksp     = None # type: PETSc.KSP
comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()

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
    dT = abs(dT)
    #beta = 0.434
    beta = 7.52e-4
    g = 9.8
    niu = 2.82e-4
    rou = 985.4
    if dT < 1.e-10 or L < 1.e-10:
        print 'Gr is zero or negative: dt: %f, L: %f' %(dT,L)
    ret = g*beta*dT*(L**3) /((niu/rou)**2) 
    return  ret

def calSteamGr(dT,L):
    dT = abs(dT)
    beta = 0.00268
    g = 9.8
    niu = 12.37e-6
    rou = 0.598
    if dT < 1.e-10 or L < 1.e-10:
        print 'Gr is zero or negative: dt: %f, L: %f' %(dT,L)
    return g*beta*dT*(L**3) /((niu/rou)**2) 

def calcSteamHeatTransferRate(Gr,Prf,Prw,L):
    mul = Gr*Prf
    lamda = 0.0250
    Nu = 0.0
    if mul<1e3:
        pass
        #uti.mpi_print('Gr, Pr steam  didnt confront Correlation\n Pr * Gr == %f!' , mul, my_rank)
    if 1e3 < mul < 1e10:
        Nu = 0.6 * (mul)**0.25 * (Prf/Prw) ** (0.25)
    if mul >=1e10:
        Nu = 0.15 * (mul)**0.333 * (Prf/Prw) ** (0.25)
    #return 500.0
    #return  0.05  * Nu * lamda / L
    return  0.1 * Nu * lamda / L

def calcBoilHeatTransferRate(Gr,Prf,Prw,L):
    mul = Gr*Prf
    lamda = 0.683
    Nu = 0.0
    if mul<1e3:
        uti.mpi_print('Gr, Pr didnt confront Correlation\n Pr * Gr == %f!' , mul, my_rank)
        assert False
    if 1e3 < mul < 1e10:
        Nu = 0.6 * (mul)**0.25 * (Prf/Prw) ** (0.25)
    if mul >=1e10:
        Nu = 0.15 * (mul)**0.333 * (Prf/Prw) ** (0.25)
    #return 500.0
    #return  0.05  * Nu * lamda / L
    return  0.1 * Nu * lamda / L

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


def calc_rod_bound(rod,Tf,nowWater):
    #type: (Types.RodUnit, float, float) -> None
    SIGMA = 5.67e-8
    EPSILONG = 0.7
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

    selfT = rod.getSurface()
    deltaT = selfT - Tf
    for ih,L in enumerate(rod.height):
        L+=1.0 #TODO to prevent zero height ...
        #print deltaT
        if L < nowWater:
            h = calcBoilHeatTransferRate(calGr(deltaT[ih],L),1.75,1.75,L) #assuming deltaT == 10
            rod.heatCoef[ih] = h
        else:
            h = calcSteamHeatTransferRate(calSteamGr(deltaT[ih], L - nowWater), 1.003, 1.003, L - nowWater)
            rod.heatCoef[ih] = h 
        #qConvection = h * (deltaT[ih])
        qRadiation = 0.0
        for dir,neighbourRod in rod.neighbour.items():
            if neighbourRod is None:
                continue
            surface = neighbourRod.getSurface()
            qRadiation += Xangle_Area[dir][1] * Xangle_Area[dir][0] * (selfT[ih] - surface[ih]) * SIGMA * EPSILONG
        rod.qbound[ih] = qRadiation

    for ir, R in enumerate(rod.rgrid): # currently adiabetic up/down
        rod.qup[ir] = 0.
        rod.qdown[ir] = 0.


def calc_fuel_temperature(rod,Tf,dt,verbose=False): #currently  only 2
    #type: (Types.RodUnit) -> None
    A    = fuelTemplate.getMat()
    b    = petsc_rhs.duplicate()
    b.zeroEntries()
    xsol = petsc_rhs.duplicate()
    for j in xrange(0, rod.nH):
        for i in xrange(0, rod.nR):
            row = j*rod.nR + i
            xsol.setValue(row, rod.T[j,i])

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
    #block melted part
    for melted_part in rod.melted:
        neighbour = [list(melted_part), list(melted_part), list(melted_part), list(melted_part)]
        neighbour[0][0] += 1
        neighbour[1][1] += 1
        neighbour[2][0] -= 1
        neighbour[3][1] -= 1
        neighbour = filter(lambda val: (0 <= val[0] < rod.nH) and (0 <= val[1] < rod.nR), neighbour)
        neighbour = map(lambda (j,i): j*rod.nR + i ,neighbour)
        row = melted_part[0] * rod.nR + melted_part[1]
        A.setValue(row,row,1.,addv = False) # diagnal == 1
        vals = [0.] * 4
        A.setValues(row, neighbour, vals, addv = False) # clear off-diagnal
        A.setValues(neighbour, row, vals, addv = False)
        b.setValue(row, 0., addv = False) # right hand side

    A.assemblyBegin()
    b.assemblyBegin()
    xsol.assemblyBegin()
    A.assemblyEnd()
    b.assemblyEnd()
    xsol.assemblyEnd()

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
        sys.stdout.write('rod %d, %d, %d, T center %f, fuelOut %f, cladOut %f, qbound:  %f, qline % f, headCoef %f\n' % (rod.address + rod.getSummary()))

def calc_other_temperature(rod, Tf, dt,verbose=False): #currently  only 2
    #type: (Types.RodUnit) -> None
    A    = blackTemplate.getMat()
    b    = petsc_rhs.duplicate()
    b.zeroEntries()
    xsol = petsc_rhs.duplicate()
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
        b.setValue(row, 0. - rod.qup[i], addv = True ) # qflux move to right hand side
    #down bound --- 2nd condition
    for i in xrange(0, rod.nR):
        j = 0
        row = j*rod.nR + i
        b.setValue(row, 0. - rod.qdown[i], addv = True) # q flux move to right hand side

    rspace = rod.rgrid[1] - rod.rgrid[0]
    hspace = rod.height[1] - rod.height[0]
    #apply transient term
    for j in xrange(0, rod.nH):
        for i in xrange(0, rod.nR):
            volumn = rod.rgrid[i] * 2 * math.pi * rspace * hspace
            transidentCoef = volumn * rod.material.cpOut * rod.material.rouOut / dt
            row = j * rod.nR + i
            b.setValue(row, transidentCoef * rod.T[j,i], addv = True)
            A.setValue(row, row, transidentCoef, addv =  True)

    #block melted part
    for melted_part in rod.melted:
        neighbour = [list(melted_part), list(melted_part), list(melted_part), list(melted_part)]
        neighbour[0][0] += 1
        neighbour[1][1] += 1
        neighbour[2][0] -= 1
        neighbour[3][1] -= 1
        neighbour = filter(lambda val: (0 <= val[0] < rod.nH) and (0 <= val[1] < rod.nR), neighbour)
        neighbour = map(lambda (j,i): j*rod.nR + i ,neighbour)
        row = melted_part[0] * rod.nR + melted_part[1]
        A.setValue(row,row,1.,addv = False) # diagnal == 1
        vals = [0.] * 4
        A.setValues(row, neighbour, vals, addv = False) # clear off-diagnal
        A.setValues(neighbour, row, vals, addv = False)
        b.setValue(row, 0., addv = False) # right hand side

    A.assemblyBegin()
    b.assemblyBegin()
    xsol.assemblyBegin()
    A.assemblyEnd()
    b.assemblyEnd()
    xsol.assemblyEnd()

    petsc_ksp.setOperators(A)
    petsc_ksp.setTolerances(rtol=1.e-6,max_it=1000)
    petsc_ksp.solve(b,xsol)
    raw_arr = xsol.getArray()

    #check if converge
    if petsc_ksp.getConvergedReason() < 0:
        raise ValueError, 'iteration not converged in %d-%d-%d' % rod.address
    else:
        if verbose:
            print 'iteration converged in %d step' % petsc_ksp.getIterationNumber()
    for row,val in enumerate(raw_arr):
        j = row / rod.nR
        i = row % rod.nR
        rod.T[j,i] = val

    if verbose :
        sys.stdout.write('rod %d, %d, %d, T center %f, fuelOut %f, cladOut %f, qbound:  %f, qline % f, headCoef %f\n' % (rod.address + rod.getSummary()))


def set_melt_for_black(rod):
    #type (Types.RodUnit) -> None
    CRITICAL = 2000
    for j in xrange(0,rod.nH):
        for i in xrange(0,rod.nR):
            if (j,i) in rod.melted:
                continue
            if rod.T[j,i] > CRITICAL:  #fuel cell part
                rod.melted.append((j,i))

def set_melt_for_fuel(rod): # find the melte part ...
    #type (Types.RodUnit) -> None
    FUEL_CRITICAL = 3000
    CLAD_CRITICAL = 2000
    for j in xrange(0,rod.nH):
        for i in xrange(0,rod.nR):
            if (j,i) in rod.melted:
                continue
            if i < rod.nRin and rod.T[j,i] > FUEL_CRITICAL:  #fuel cell part
                rod.melted.append((j,i))
            elif i >= rod.nRin and rod.T[j,i] > CLAD_CRITICAL: # clad
                rod.melted.append((j,i))

def calc_rod_source(rod,nowPower):
    #type (Types.RodUnit,float)->(None)
    h = (rod.height[1]-rod.height[0])
    volumn = math.pi * rod.inRadious**2 * h
    rodPower = rod.radialPowerFactor * nowPower
    #print rodPower, nowPower
    rod.qsource = rod.axialPowerFactor * rodPower / volumn
    assert rod.qsource.shape[0] == rod.nH

def summarize(rods):
    #type: (list)
    center = [0,(0,)]
    qline = [0,(0,)]
    qbound = [0,(0,)]
    hcoef = [0,(0,)]
    #fuelgap = []
    #cladout = []
    for rod in rods:
        ret = rod.getSummary()
        if ret[0] > center[0]:
            center[0] = ret[0]
            center[1] = rod.address
        if ret[3] > qbound[0]:
            qbound[0] = ret[3]
            qbound[1] = rod.address
        if ret[4] > qline[0]:
            qline[0] = ret[4]
            qline[1] = rod.address
        if ret[5] > hcoef[0]:
            hcoef[0] = ret[5]
            hcoef[1] = rod.address
    uti.mpi_print('max: T %f, qbound %f, qline %f,hcoef %f -- [%d %d %d]', (center[0], qbound[0], qline[0], hcoef[0]) + center[1], my_rank) 
    #uti.mpi_print ('max rod temperature %e -[%d, %d, %d]', ((center[0],)+ center[1]), comm_rank )

def syncBoundary(rods, bound_array, mask, verbose=False):
    assert isinstance(rods, list)
    assert isinstance(bound_array, dict)
    assert len(bound_array) <= 8 and len(bound_array) >= 1
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    diagnal_rank = filter(lambda (rank, arr): arr.shape[0] == 1, bound_array.items())
    diagnal_rank = map(lambda (r,arr) : r, diagnal_rank)
    comm.Barrier()
    #copy local temp to buffer
    for rod in rods:
        for direction, neighbourRod in rod.neighbour.items():
            if neighbourRod is None:
                continue
            that_rank = initor.get_rank(mask, neighbourRod.address[2])
            if that_rank in diagnal_rank:
                neighbourRod.getSurface()[:] = rod.getSurface()[:]
            elif that_rank in bound_array.keys():
                #if direction.count('+') + direction.count('-') == 1:
                neighbourRod.getSurface()[:] = rod.getSurface()[:]
                    
    for to_rank, bound in  bound_array.items():
        if to_rank >= comm_size: #to enforce partial calculation...
            continue
        comm.Bsend(bound, dest = to_rank, tag = comm_rank) 
        if verbose:
            print '%d ---> %d : %d' % (comm_rank, to_rank, len(bound))
    for from_rank, bound in bound_array.items():
        if from_rank >= comm_size: #to enforce partial calculation...
            continue
        comm.Recv(bound, source = from_rank, tag = from_rank)
        if verbose:
            print '%d <--- %d : %d' % (comm_rank, from_rank, len(bound))

def start(rods, bound_array, mask, timeLimit, dt):
    #type: (list, dict,float,float) -> None
    uti.root_print('%s', 'start simulation', my_rank)
    uti.root_print('%s', 'the first steady status', my_rank)
    buff = np.zeros(99999999 * len(bound_array.values()))
    MPI.Attach_buffer(buff)
    comm.Barrier()
    Types.PressureVessle.timePush(0.0)
    nowWater, nowPower = Types.PressureVessle.now()
    for rod in rods:
        calc_rod_source(rod,nowPower)
        calc_rod_bound(rod,373,nowWater)
    #for rod in tqdm(rods): #update T
    for rod in rods: #update T
        if rod.type is Types.RodType.fuel:
            #calc_fuel_temperature(rod,373,1.e30, False)
            calc_fuel_temperature(rod,373,1.e30, False)
        else:
            #calc_other_temperature(rod,373,1.e30, False)
            calc_other_temperature(rod,373,1.e30, False)
    print 'xxxxxxxxxxxxxxxxxxx %d ' % my_rank
    syncBoundary(rods, bound_array, mask)
    print 'xxxxxxxxxxxxxxxxxxx %d ' % my_rank
    uti.root_print('%s', 'finish calculating steady status', my_rank)
    #begin time iteration
    summarize(rods)
    open('rod_1.dat','w').write(rods[0].get2DTec() )
    step_counter = 0
    while Types.PressureVessle.currentTime <= timeLimit:
        uti.root_print('%s... now %s', ('saving', time.strftime('%Y-%m-%d %X', time.localtime())), my_rank)
        #save_restart_file(rods)
        Types.PressureVessle.timePush(dt)
        uti.root_print('solving time %f', Types.PressureVessle.currentTime, my_rank)
        nowWater, nowPower = Types.PressureVessle.now()
        uti.root_print('%s', 'calculating heat souce and temp bound', my_rank)
        for rod in rods:
            calc_rod_source(rod,nowPower)
            calc_rod_bound(rod,373,nowWater) #last parameter is fluid temp
        #for rod in tqdm(rods): #update T
        for rod in rods: #update T
            if rod.type is Types.RodType.fuel:
                calc_fuel_temperature(rod, 373, dt)   #a PETSc impementation
                set_melt_for_fuel(rod)
            else:
                calc_other_temperature(rod, 373, dt)  #a PETSc impementation
                set_melt_for_black(rod)
        if step_counter % 10:
            syncBoundary(rods, bound_array, mask)
        summarize(rods)
        step_counter += 1
    uti.root_print('%s', 'simulation done', my_rank)
    MPI.Detach_buffer()

