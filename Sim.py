import numpy as np
import math
import time
import struct
import sys
import CMTypes as Types
import utility as uti
import initializer as initor
from mpi4py import MPI
from petsc4py import PETSc

fuelTemplate  = None # type: Types.PETScWrapper
blackTemplate = None # type: Types.PETScWrapper
petsc_rhs          = None # type: PETSc.Vec
petsc_ksp     = None # type: PETSc.KSP
comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
my_size = comm.Get_size()

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
    uti.mpi_print('%s', 'configuring material', my_rank)
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
    return   Nu * lamda / L

def calcBoilHeatTransferRate(Gr,Prf,Prw,L):
    mul = Gr*Prf
    lamda = 0.683
    Nu = 0.0
    if mul<1e3:    
        pass	
        #uti.mpi_print('Gr, Pr didnt confront Correlation\n Pr * Gr == %f!' , mul, my_rank)
    if 1e3 < mul < 1e10:
        Nu = 0.6 * (mul)**0.25 * (Prf/Prw) ** (0.25)
    if mul >=1e10:
        Nu = 0.15 * (mul)**0.333 * (Prf/Prw) ** (0.25)
    #return 500.0
    #return  0.05  * Nu * lamda / L
    return  Nu * lamda  / L

def save_tec_2d(rod, now):
    open('tec/rod_%s_at%ds.dat' % (str(rod.address), int(now)), 'w').write(rod.get2DTec() )

def save_tec(rods):
    #type: (Types.RodUnits) -> None
    #print to tecplot
    if my_rank == 0:
        title = 'tec/head_tec.dat'
    else:
        title = 'tec/tec_%d.dat' % my_rank
    _file = open(title ,'w')
    if my_rank == 0:
        _file.write('variables="x","y","z","t"\n')
    for rod in rods:
        buffer = rod.getTecplotZone()
        _file.write(buffer)
    _file.close()
    return


def save_restart_file(t, rods):
    title = 'sav/rank_%d.npy' % my_rank
    _f = open(title,'wb')
    data = struct.pack('f',t)
    _f.write(data)
    for rod in rods:
        np.save(_f, rod.T)

def load_restart_file(rods):
    title = 'sav/rank_%d.npy' % my_rank
    try:
        _f = open(title,'r')
        data = _f.read(struct.calcsize('f'))
        t, = struct.unpack('f', data)
        for rod in rods:
            rod.T[:,:] = np.load(_f)[:,:]
        return t
    except IOError :
        uti.mpi_print('%s', 'new start', my_rank)
    return 

def calc_rod_bound(rod,Tf,nowWater):
    #type: (Types.RodUnit, float, float) -> None
    RADIOUS = rod.radious
    SPACE = 0.016
    ROD_SPACE = rod.height[1] - rod.height[0]
    Xangle_Area = {
                    'xy+'   : (0.145, RADIOUS*2                     * ROD_SPACE ), #angle coef, surface 
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
        #TODO to prevent zero height ...
        rod.qbound[ih] = 0.0
        rod.heatCoef[ih] = 0.0
        h = 0.0
        area = math.pi * ROD_SPACE * RADIOUS * 2 
        if L < nowWater:#fluid
            h = calcBoilHeatTransferRate(calGr(deltaT[ih], L + 1.0), 1.75, 1.75, L + 1.0 ) #assuming deltaT == 10
            #rod.qbound[ih] += h * area * (0. - deltaT[ih])
            rod.heatCoef[ih] += h
        else: #gas
            h = calcSteamHeatTransferRate(calSteamGr(deltaT[ih], L - nowWater + 1.0), 1.003, 1.003, L - nowWater + 1.0)
            #rod.qbound[ih] += h * area * (0. - deltaT[ih])
            rod.heatCoef[ih] += h 
        #qConvection = h * (deltaT[ih])
        qRadiation = 0.0
        #convection to neighbour
        #radiation
        for dir,neighbourRod in rod.neighbour.items():
            if neighbourRod is None:
                continue
            surface = neighbourRod.getSurface()
            if L >= nowWater:
                rod.qbound[ih] += 0.01 * h *  (selfT[ih] - surface[ih] ) # neighbour  convective considered
            qRadiation += Xangle_Area[dir][1] * Xangle_Area[dir][0] * (selfT[ih] ** 4 - surface[ih] ** 4 ) * \
                          Types.Constant.SIGMA * \
                          Types.Constant.EPSILONG * \
                          Types.Constant.RADIO_ANGLE_AMPLIFIER
        rod.qbound[ih] += qRadiation
        #uti.mpi_print('qbound %d %e', (ih, rod.qbound[ih]), my_rank)
        #assert isinstance(rod.qbound[ih], float)
        #assert isinstance(rod.heatCoef[ih], float)

    for ir, R in enumerate(rod.rgrid): # currently adiabetic up/down
        rod.qup[ir] = 0.
        rod.qdown[ir] = 0.

def build_basic_temperature(A, b, dt, Tf, rod):
    # out bound --- 3rd condition
    outsideAera = math.pi * 2 * rod.radious * ( rod.height[1] - rod.height[0] )
    surface = rod.getSurface()
    for j in xrange(0, rod.nH):
        i = rod.nR - 1
        row = j*rod.nR + i
        b.setValue(row, (rod.heatCoef[j] * Tf  - rod.qbound[j] ) * outsideAera, addv = True ) # qbound move to right hand side ...
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

def build_melt_temperature(A, b, rod):
    set_melted = False
    for melted_part in rod.melted.status:
        set_melted = True
        neighbour = [list(melted_part), list(melted_part), list(melted_part), list(melted_part)]
        neighbour[0][0] += 1
        neighbour[1][1] += 1
        neighbour[2][0] -= 1
        neighbour[3][1] -= 1
        neighbour = filter(lambda val: (0 <= val[0] < rod.nH) and (0 <= val[1] < rod.nR), neighbour)
        neighbour = map(lambda (j,i): j*rod.nR + i ,neighbour)
        row = melted_part[0] * rod.nR + melted_part[1]
        A.setValue(row,row,1.,addv = False) # diagnal == 1
        vals = [0.] * len(neighbour) 
        A.setValues(row, neighbour, vals, addv = False) # clear off-diagnal
        A.setValues(neighbour, row, vals, addv = False)
        b.setValue(row, 0., addv = False) # right hand side
    if set_melted :
        A.assemblyBegin()
        b.assemblyBegin()
        A.assemblyEnd()
        b.assemblyEnd()

def petsc_solve(A, b, xsol, ksp):
    ksp.setInitialGuessNonzero(True)
    ksp.setOperators(A)
    ksp.setTolerances(rtol=1.e-6, max_it=1000)
    ksp.solve(b, xsol)

    if ksp.getConvergedReason() < 0:
        raise ValueError, 'iteration not converged'
    #else:
       # print 'iteration converged in %d step' % ksp.getIterationNumber()
    return xsol.getArray()

def calc_fuel_temperature(rod,Tf,dt, verbose=False): #currently  only 2
    #type: (Types.RodUnit) -> None
    A    = fuelTemplate.getMat()
    b    = petsc_rhs.duplicate()
    b.zeroEntries()
    xsol = petsc_rhs.duplicate()
    for j in xrange(0, rod.nH):
        for i in xrange(0, rod.nR):
            row = j*rod.nR + i
            xsol.setValue(row, rod.T[j,i])
    build_basic_temperature(A, b, dt, Tf, rod)
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

    A.assemblyBegin()
    b.assemblyBegin()
    xsol.assemblyBegin()
    A.assemblyEnd()
    b.assemblyEnd()
    xsol.assemblyEnd()
    #block melted part
    build_melt_temperature(A, b, rod)

    raw_arr = petsc_solve(A, b, xsol, petsc_ksp)
    for row,val in enumerate(raw_arr):
        j = row / rod.nR
        i = row % rod.nR
        rod.T[j,i] = val
    if verbose:
        sys.stdout.write('rod [%d %d %d] T center %f, fuelOut %f, cladOut %f, qbound:  %f, qline % f, headCoef %f, mass %f\n' % (rod.address + rod.getSummary()))
   
def calc_other_temperature(rod, Tf, dt, verbose=False): #currently  only 2
    #type: (Types.RodUnit) -> None
    A    = blackTemplate.getMat()
    b    = petsc_rhs.duplicate()
    b.zeroEntries()
    xsol = petsc_rhs.duplicate()
    for j in xrange(0, rod.nH):
        for i in xrange(0, rod.nR):
            row = j*rod.nR + i
            xsol.setValue(row,rod.T[j,i])
    build_basic_temperature(A, b, dt, Tf, rod)
    A.assemblyBegin()
    b.assemblyBegin()
    xsol.assemblyBegin()
    A.assemblyEnd()
    b.assemblyEnd()
    xsol.assemblyEnd()

    #block melted part
    build_melt_temperature(A, b, rod)

    raw_arr = petsc_solve(A, b, xsol, petsc_ksp)
    for row,val in enumerate(raw_arr):
        j = row / rod.nR
        i = row % rod.nR
        rod.T[j,i] = val

    if verbose :
        sys.stdout.write('rod [%d %d %d] T center %f, fuelOut %f, cladOut %f, qbound:  %f, qline % f, headCoef %f, mass %f\n' % (rod.address + rod.getSummary()))

def set_melt(rod): # find the melte part ...
    #type (Types.RodUnit) -> None
    FUEL_CRITICAL = rod.material.meltingPointIn
    CLAD_CRITICAL = rod.material.meltingPointOut
    for j in xrange(0,rod.nH):
        for i in xrange(0,rod.nR):
            if (j,i) in rod.melted.status:
                continue
            if i < rod.nRin and rod.T[j,i] > FUEL_CRITICAL:  #fuel cell part
                rod.melted.melt_mass += rod.get_volumn(j, i) * rod.material.rouIn
                rod.T[j, i] = 0.0
                rod.melted.status.append((j,i))
            elif i >= rod.nRin and rod.T[j,i] > CLAD_CRITICAL: # clad
                rod.melted.melt_mass += rod.get_volumn(j, i) * rod.material.rouOut
                rod.T[j, i] = 0.0
                rod.melted.status.append((j,i))

def calc_rod_source(rod,nowPower):
    #type (Types.RodUnit,float)->(None)
    h = (rod.height[1]-rod.height[0])
    volumn = math.pi * rod.inRadious**2 * h
    rodPower = rod.radialPowerFactor * nowPower
    #print rodPower, nowPower
    rod.qsource = rod.axialPowerFactor * rodPower / volumn
    assert rod.qsource.shape[0] == rod.nH

def summarize(now, rods):
    #type: (list)
    center = [-1,(0,0,0)]
    '''
    qline = [0, (0,0,0)]
    qbound = [0,(0,0,0)]
    hcoef = [0, (0,0,0)]
    mass = [0,  (0,0,0)]
    '''
    #fuelgap = []
    #cladout = []
    hot_rod = None
    n = len(rods)
    static = {'T':np.zeros(n),
              'qbound':np.zeros(n),
              'qline':np.zeros(n),
              'hcoef':np.zeros(n),
              'mass':np.zeros(n),
    }
    counter = 0
    for rod in rods:
        ret = rod.getSummary()
        static['T'][counter] = ret[0]
        static['qbound'][counter] = ret[3]
        static['qline'][counter] = ret[4]
        static['hcoef'][counter] = ret[5]
        static['mass'][counter] = ret[6]
        counter += 1
        if ret[0] > center[0]:
            center[0] = ret[0]
            center[1] = rod.address
            hot_rod = rod
    uti.mpi_print('time %e max: T %f, qbound %e, qline %f, hcoef %f, mass %e -- [%d %d %d]', (now, 
                static['T'].max(), static['qbound'].mean(), static['qline'].mean(), static['hcoef'].mean(), static['mass'].sum()) + center[1], my_rank) 
    return hot_rod
    #uti.mpi_print ('max rod temperature %e -[%d, %d, %d]', ((center[0],)+ center[1]), comm_rank )

def syncBoundary(rods, bound_array, mask, boundary_assembly_rank, barrel_buf,verbose=False):
    assert isinstance(rods, list)
    assert isinstance(bound_array, dict)
    assert len(bound_array) <= 8 and len(bound_array) >= 1
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank() 
    comm_size = comm.Get_size()
    diagnal_rank = filter(lambda (rank, arr): arr.shape[0] == 1, bound_array.items())
    diagnal_rank = map(lambda (r,arr) : r, diagnal_rank)
    comm.Barrier()
    uti.mpi_print('%s', 'rod syncing', my_rank)
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
        if verbose:
            uti.mpi_print('%d ---> %d : %d' , (comm_rank, to_rank, len(bound)), my_rank)
        comm.Bsend(bound, dest = to_rank, tag = comm_rank) 
    reqs = []
    for from_rank, bound in bound_array.items():
        if from_rank >= comm_size: #to enforce partial calculation...
            continue
        req = comm.Irecv(bound, source = from_rank, tag = from_rank)
        reqs.append(req)
        if verbose:
            uti.mpi_print( '%d <--- %d : %d', (comm_rank, from_rank, len(bound)), my_rank)
     # outside
    if my_rank in boundary_assembly_rank:
        send_buf = np.zeros(rods[0].nH)
        send_count = 0
        for rod in rods:
            for dir, neighbour in rod.neighbour.items():
                if neighbour is None and (dir == 'x+y' or dir == 'xy+'):
                    send_count += 1
                    send_buf += rod.getSurface() 
        send_buf /= float(send_count)
        comm.Bsend(send_buf, dest = my_size - 1, tag = my_rank)
        req = comm.Irecv(barrel_buf, source = my_size - 1, tag = my_size - 1)
        reqs.append(req)
    MPI.Request.Waitall(reqs)

def start(rods, bound_array, mask, boundary_assembly_rank, timeLimit, dt):
    #type: (list, dict,float,float) -> None
    uti.root_print('%s', 'start simulation', my_rank)
    uti.root_print('%s', 'the first steady status', my_rank)
    barrel_buf = None
    if my_rank in boundary_assembly_rank:
        barrel_buf = np.zeros(rods[0].nH);
    buff = np.zeros(9999999  )
    MPI.Attach_buffer(buff)
    syncBoundary(rods, bound_array, mask, boundary_assembly_rank, barrel_buf, True)
    
    Types.PressureVessle.timePush(0.0)
    nowWater, nowPower = Types.PressureVessle.now()
    for rod in rods:
        calc_rod_source(rod, nowPower)
        calc_rod_bound(rod, 373, nowWater)
    #for rod in tqdm(rods): #update T
    for rod in rods: #update T 
        if rod.type is Types.RodType.fuel:
            calc_fuel_temperature(rod,373,1.e30, False)
            #calc_fuel_temperature(rod, 373, 1.e1, nowWater, False)
        else:
            calc_other_temperature(rod,373,1.e30, False)
            #calc_other_temperature(rod, 373, 1.e1, nowWater ,False)
    uti.root_print('%s', 'finish calculating steady status', my_rank)
    #begin time iteration
    summarize(0.0, rods)
    step_counter = 0
    restart_time = load_restart_file(rods)
    #uti.mpi_print('%s',str(bound_array), my_rank)
    syncBoundary(rods, bound_array, mask, boundary_assembly_rank, barrel_buf, True)
    if restart_time is not None:
        Types.PressureVessle.currentTime = restart_time
        uti.mpi_print('restarting from time %f, end time %f', (Types.PressureVessle.currentTime, timeLimit), my_rank)
    while Types.PressureVessle.currentTime <= timeLimit:
        uti.root_print('%s... now %s', ('calcing', time.strftime('%Y-%m-%d %X', time.localtime())), my_rank)
        Types.PressureVessle.timePush(dt)
        nowWater, nowPower = Types.PressureVessle.now()
        uti.root_print('solving time %f, water level %f', (Types.PressureVessle.currentTime, nowWater), my_rank)
        uti.root_print('%s', 'calculating heat souce and temp bound', my_rank)
        for rod in rods:
            calc_rod_source(rod, nowPower)
            calc_rod_bound(rod, 373, nowWater) #last parameter is fluid temp
        #for rod in tqdm(rods): #update T
        for rod in rods: #update T
            if rod.type is Types.RodType.fuel:
                calc_fuel_temperature(rod, 373, dt )   #a PETSc impementation
                #set_melt_for_fuel(rod)
            else:
                calc_other_temperature(rod, 373, dt)  #a PETSc impementation
                #set_melt_for_black(rod)
            set_melt(rod)
        if step_counter % 10 == 0:
            syncBoundary(rods, bound_array, mask, boundary_assembly_rank, barrel_buf)
        hot_rod = summarize(Types.PressureVessle.currentTime ,rods)
        if step_counter % 100 == 0:
            save_tec(rods)
            save_tec_2d(hot_rod, Types.PressureVessle.currentTime)
        if step_counter % 500 == 0:
            save_restart_file(Types.PressureVessle.currentTime, rods)
        step_counter += 1
    uti.root_print('%s', 'simulation done', my_rank)
    MPI.Detach_buffer()

