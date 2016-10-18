import numpy as np
import CMTypes as Types
import Sim
from petsc4py import PETSc
from mpi4py import MPI

barrel_template =  None
rhs_template = PETSc.Vec()

petsc_ksp = PETSc.KSP().create()
petsc_ksp.setType(PETSc.KSP.Type.CG)
pc = petsc_ksp.getPC()
pc.setType(PETSc.PC.Type.ILU)
petsc_ksp.setPC(pc)

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
my_size = comm.Get_size()

def Barrel(Types.RodUnit):
    def  __init__(self, nh, nr, h, rin, rout):
        super(Barrel, self).__init__(-1, nh, 0, nr,(0., 0.), (0, 0, 0), rin, rout, 0.0)
        self.qinbound = np.zeros(nh)
        self.hinCoef = np.zeros(nh)
        self.qbound = np.zerso(nh)
        self.qup = np.zeros(nr)
        self.qdown = np.zeros(nr)
        self.T = np.zeros(nh, nr)
        self.height = np.linspace(0., h, nh)

    def getInSurface(self):
        return self.T[:,0]

    def get_in_area(self):
        return (self.height[1] - self.height[0]) * math.pi * 2 * self.inRadious)

def set_barrel_material(bar):
    assert isinstance(bar, Barrel)
    ber.material = Types.MaterialProterty(
            'Barrel',
            )

def calc_barrel_bound(bar, Tf, nowWater, rod_temp):
    assert isinstance(bar, Barrel)
    assert isinstance(rod_temp, np.ndarray)
    nh = bar.nH
    surface = bar.getInSurface()
    deltaT = tf - surface
    SPACE = bar.height[1] - bar.height[0]
    for i, L in enumerate(bar.height):
        # h in 
        if bar.height[i] < nowWater:
            h = Sim.calcBoilHeatTransferRate(Sim.calGr(deltaT[i], L + 1.0), 1.75, 1.75, L + 1.0)
            bar.hinCoef[i] += h
        else:
            h = Sim.calcSteamHeatTransferRate(Sim.calSteamGr(deltaT[i], L - nowWater + 1.0), 1.003, 1.003, L - nowWater + 1.0)
            bar.hinCoef[i] += h
        #q in 
        qrad = SPACE * 
               Types.Constant.EPSILONG *
               Types.Constant.SIGMA * 
               Types.Constant.RADIO_ANGLE_AMPLIFIER *
               (rod_temp[i] ** 4 - surface[i] ** 4) 
        bar.qinbound[i] = qrad
    #h out
    bar.heatCoef[:] = 0.0
    #q out
    bar.qbound[:] = 0.0

def syncBarrel(barrel_mask, bar, temps): #only called on barrel
    assert isinstance(barrel_mask, list)
    assert isinstance(temps, np.ndarray)
    recv_list = range(0, my_rank)
    nh = bar.nH
    buf = np.zeros((len(recv_list), nh))
    reqs = []
    for recv_rank in recv_list:
        req = comm.Irecv(buf[recv_rank, :], source = recv_rank, tag = recv_rank)
        reqs.append(req)
    MPI.Request.Waitall(reqs)
    for h in xrange(0, nh):
        temps[h] = buf[:, h].mean()

def calc_barrel_temperature(bar, Tf, dt):
    A = barrel_template.getMat()
    b = rhs_template.duplicate()
    b.zeroEntries()
    xsol = rhs_template.duplicate()
    for j in xrange(0, bar.nH):
        for i in xrange(0, bar.nR):
            row = j*rod.nR + i
            xsol.setValue(row,rod.T[j,i])
    Sim.build_basic_temperature(A, b, dt, bar)
    insideArea = bar.get_in_area()
    for j in xrange(0, bar.nH):
        i = 0
        row = j*bar.nR + i
        b.setValue(row, rod.hinCoefp[j] * Tf * insideArea - rod.qinbound[j], addv = True)
        A.setValue(row, row, rod.hinCoef[j] * insideArea, addv = True)
    A.assemblyBegin()
    b.assemblyBegin()
    xsol.assemblyBegin()
    A.assemblyEnd()
    b.assemblyEnd()
    xsol.assemblyEnd()
 
    Sim.build_melt_temperature(A, b, bar)
    raw_arr = Sim.petsc_solve(A, b, petsc_ksp)
    for row,val in enumerate(raw_arr):
        j = row / rod.nR
        i = row % rod.nR
        rod.T[j,i] = val
    if verbose :
        sys.stdout.write('rod %d, %d, %d, T center %f, fuelOut %f, cladOut %f, qbound:  %f, qline % f, headCoef %f\n' % (rod.address + rod.getSummary()))

def barrel_init(nh, nr, h, rin, rout):
    bar = Barrel(nh, nr, h, rin, rout)
    set_barrel_material(bar)

    petsc_warpper = PETScWrapper(nh * nr, nr, nh)
    hspace = bar.height[1] - bar.height[0]
    rgrid = bar.rgrid[1] - bar.rgrid[0]
    petsc_warpper.fillTemplateBlack(bar.material.lamdaIn, r, rgrid)
    global barrel_template
    barrel_template = petsc_warpper
    return bar

