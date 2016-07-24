import numpy as np
import math
import CMTypes as Types


def config_material(rods):
    #type(Types.RodUnits) -> ddNone
    print 'configuring material'
    for rod in rods:
        if rod.type is Types.RodType.black or rod.type is Types.RodType.gray:
            rod.material = Types.MaterialProterty('gray/black',25,25,7020,7020,835,835,1600,1600)
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


def set_initial(rods,tstart,deltaT,Tf):
    # types: (Types.RodUnits, float) -> None
    print 'initialing temperature field'
    assert isinstance(rods,list)
    Types.PressureVessle.currentTime = tstart
    Types.PressureVessle.timePush(0)
    nowWater, nowPower = Types.PressureVessle.now()
    for rod in rods:
        assert isinstance(rod,Types.RodUnit)
        rodPower = rod.radialPowerFactor * nowPower
        rod.ql = rod.axialPowwerFactor * rodPower
        Trows = []
        #print 'donging rod %s' % str(rod.address)
        for ih in range(0,rod.nH):
            L = rod.height[ih] + 1 # TODO L should NOT be ZERO
            q = rod.ql[ih]
            h = calcBoilHeatTransferRate(calGr(deltaT,L),1,1,L) #assuming deltaT == 10
            Tco = Tf + q / (math.pi * 2 * rod.radious * h)
            Tci = Tco + q / (2 * math.pi * rod.material.lamdaOut) * math.exp(rod.radious/rod.inRadious)
            To  = Tci + q / (math.pi * rod.inRadious * 2 *rod.gapHeatRate )
            tIn = np.linspace(To,To,rod.nRin)
            tOut= np.linspace(Tci,Tco,rod.nR - rod.nRin)
            Trows.append( np.hstack((tIn,tOut)) )
        Trows = tuple(Trows)
        rod.T = np.vstack(Trows)
        assert rod.T.shape == (rod.nH,rod.nR)

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


def start():
    print 'start'
