# Aiming for a module for zombie state computation
import math, numpy, scipy
from functools import reduce
from pyscf import gto, scf, ao2mo
from pyscf import tools
from pyscf import symm

def make(norb):
    # Returns a completely empty zombie state (all zeros)
    zz = numpy.zeros((norb,2))
    return zz

def populate(zz,occ):
    zz[:,:] = 0.
    for iorb in range(len(zz[:,0])):
        zz[iorb,occ[iorb]] = 1.
    return zz

def new(norb,occ):
    zz = make(norb)
    zz = populate(zz,occ)
    return zz

def new_ran(norb):
    rantemp = 2.0*math.pi*numpy.random.random((norb))
    zz = make(norb)
    zz[:,0] = numpy.cos(rantemp)
    zz[:,1] = numpy.sin(rantemp)
    return zz
    
def overlap(z1,z2):
    # coefficients can be real or complex
    temp = 1.0
    for iorb in range(len(z1[:,0])):
        tt = z1[iorb,0].conjugate()*z2[iorb,0] \
             + z1[iorb,1].conjugate()*z2[iorb,1]
        temp = temp*tt
    return temp

def overlap_f(z1,z2):
    # coefficients can be real or complex
    temp = 1.0
    for iorb in range(len(z1[:,0])):
        tt = z1[iorb,0].conjugate()*z2[iorb,0] \
             + z1[iorb,1].conjugate()*z2[iorb,1]
        if tt == 0.0:
            return 0.0
        temp = temp*tt
    return temp


def cr(z1,iorb):
    # Creation operator on orbital iorb
    z1[iorb,1] = z1[iorb,0]
    z1[iorb,0] = 0.0
    z1[:iorb,1] *= -1.0
    return z1

def an(z1,iorb):
    # Annihilation operator on orbital iorb
    z1[iorb,0] = z1[iorb,1]
    z1[iorb,1] = 0.0
    z1[:iorb,1] *= -1.0
    return z1

def num(z1,iorb):
    # Number operator on a specific orbital iorb
    z1[iorb,0] = 0.0
    return z1

def numo(z1,z2):
    # Calculates <z1| sum ni |z2>
    norb = len(z1[:,0])
    temp = 0.0
    for iorb in range(norb):
        zz = numpy.copy(z2)
        zzi = num(zz,iorb)
        temp += overlap(z1,zzi)
    return temp

def numm(z1):
    # Calculates <z1| sum ni |z1>
    return numo(z1,z1)

def numf(z1,z2):
    # faster algorithm for application of number operator
    # needs adaptation for complex calculation
    norb = len(z1[:,0])
    cc = numpy.zeros((norb))
    dd = numpy.zeros((norb))
    mult = numpy.zeros((norb))
    multb = numpy.zeros((norb))
    for iorb in range(norb):
        mult[iorb] = z1[iorb,1].conjugate()*z2[iorb,1]
        multb[iorb] = mult[iorb] + z1[iorb,0].conjugate()*z2[iorb,0]
    cc[0] = multb[0]
    dd[-1] = multb[-1]
    for iorb in range(1,norb):
        cc[iorb] = cc[iorb-1]*multb[iorb]
    for iorb in range(norb-2,-1,-1):# I think indices are correct
        dd[iorb] = dd[iorb+1]*multb[iorb]
    temp = mult[0]*dd[1]
    #print('mult',mult)
    #print('cc',cc)
    #print('dd',dd)
    for iorb in range(1,norb-1):
        temp += cc[iorb-1]*mult[iorb]*dd[iorb+1]
        #print(iorb,temp,mult[iorb])
    temp += cc[norb-2]*mult[-1]
    return temp

def nsq(z1,z2,norb):
    "N squared, used numo to cope with complex"
    temp = 0.0
    for iorb in range(norb):
        zt = numpy.copy(z2)
        zt = num(zt,iorb)
        temp += numo(z1,zt)
    return temp

def isdet(zom,norb):
    """Checks if a given zombie state
    is equal to a single determinant
    and the entries for each spinorbital are either 0 and 1 or 
    0 and -1"""
    for iorb in range(norb):
        if zom[iorb,0] == 0.0:
            if zom[iorb,1] in [1.0,-1.0]:
                pass
            else:
                return False
        elif zom[iorb,0] in [1.0,-1.0]:
            if zom[iorb,1] == 0.0:
                pass
            else:
                return False
        else:
            return False
        #if (zom[iorb,0] == 0.0 and zom[iorb,1] == 1.0) or \
        #   (zom[iorb,0] == 1.0 and zom[iorb,1] == 0.0):
        #    pass
        #else:
        #    return False
    return True

def iszero(z1):
    """Determines if a given zombie state vanishes
    Returns True if state vanishes
    False otherwise"""
    # coefficients can be real or complex
    for iorb in range(len(z1[:,0])):
        tt = z1[iorb,0].conjugate()*z1[iorb,0] \
             + z1[iorb,1].conjugate()*z1[iorb,1]
        if tt == 0.0:
            return True
    return False

def numtodet(i,norb):
    """Turns an integer number 0 <= j <= 2**norb-1
    Into an arrange length norb with orbital occupancy
    by converting the integer into binary"""
    if i >= 2**norb:
        raise ValueError('i too big')
    bini = numpy.zeros((norb),dtype = int)
    it = i
    for j in range(norb):
        bini[j] = it%2
        it -= bini[j]
        it /= 2
    return bini

def dettonum(bini,norb):
    """Turns an integer array of 0s and 1s length norb
    into its corresponding binary number
    The reverse of numtodet"""
    twop = numpy.zeros((norb),dtype = int)
    for i in range(norb):
        twop[i] = 2**i
    return numpy.dot(bini,twop)

def sz(zom1,zom2,norb):
    temp = 0.0
    for iorb in range(0,norb,2):
        zomt = numpy.copy(zom2)
        zomt = an(zomt,iorb)
        zomt = cr(zomt,iorb)
        temp += 0.5*overlap(zom1,zomt)
    for iorb in range(1,norb,2):
        zomt = numpy.copy(zom2)
        zomt = an(zomt,iorb)
        zomt = cr(zomt,iorb)
        temp -= 0.5*overlap(zom1,zomt)
    return temp

def szf(z1,z2,norb):
    """Fast algorithm for application of sz operator
    O(N) steps"""
    if norb%2 != 0:
        raise ValueError('Even norb required')
    cc = numpy.zeros((norb))
    dd = numpy.zeros((norb))
    mult = numpy.zeros((norb))
    multb = numpy.zeros((norb))
    for iorb in range(norb):
        mult[iorb] = z1[iorb,1].conjugate()*z2[iorb,1]
        multb[iorb] = mult[iorb] + z1[iorb,0].conjugate()*z2[iorb,0]
    cc[0] = multb[0]
    dd[-1] = multb[-1]
    for iorb in range(1,norb):
        cc[iorb] = cc[iorb-1]*multb[iorb]
    for iorb in range(norb-2,-1,-1):
        dd[iorb] = dd[iorb+1]*multb[iorb]
    temp = mult[0]*dd[1]
    for iorb in range(1,norb-1):
        temp += cc[iorb-1]*mult[iorb]*dd[iorb+1]*(-1)**iorb
    temp -= cc[norb-2]*mult[-1]
    return 0.5*temp


# Further spin functions
def splusf(zs,k):
    """Faster version of s+ acting on spatial orbital k"""
    zt = numpy.copy(zs)
    zt[2*k,1] = zt[2*k,0]
    zt[2*k,0] = 0.0
    zt[2*k+1,0] = zt[2*k+1,1]
    zt[2*k+1,1] = 0.0
    return zt

def sminf(zs,k):
    """Faster version of s- acting on spatial orbital k"""
    zt = numpy.copy(zs)
    zt[2*k,0] = zt[2*k,1]
    zt[2*k,1] = 0.0
    zt[2*k+1,1] = zt[2*k+1,0]
    zt[2*k+1,0] = 0.0
    return zt

def spsmf(zs1, zs2, norb):
    Kmax = int(norb/2)
    temp = 0.0
    for isp in range(Kmax):
        zst = numpy.copy(zs2)
        zst = sminf(zst,isp)
        #print(zst)
        for jsp in range(Kmax):
            zst2 = numpy.copy(zst)
            zst2 = splusf(zst2,jsp)
            ov = overlap_f(zs1,zst2)
            temp=temp+ov
            #print(ov)
    return temp

def sz2(zs1,zs2,norb):
    """Computing <zs1 | S_z^2 | zs2 >"""
    temp = 0.0
    for iorb in range(norb):
        for jorb in range(norb):
            zs2t = numpy.copy(zs2)
            zs2t = num(zs2t,jorb)
            zs2t = num(zs2t,iorb)
            temp += overlap_f(zs1,zs2t)*(-1)**(iorb+jorb)
    return temp*0.25

def sz2f(zs1,zs2,norb):
    """Computing <zs1 | S_z^2 | zs2 >
    O(M^2) not O(M^3)"""
    temp = 0.0
    for iorb in range(norb):
        zs2t = numpy.copy(zs2)
        zs2t = num(zs2t,iorb)
        temp += szf(zs1,zs2t,norb)*(-1)**iorb
    return temp*0.5

def spsm2(zs1,zs2,norb):
    kmax=int(norb/2)
    zs3=[]
    tot=0.0
    for isp in range(kmax):
        zs3.append(sminf(zs2,isp))
    for isp in range(kmax):
        zsc=zs3[isp]
        cc=numpy.zeros((kmax))
        d = numpy.zeros((kmax))    
        for jsp in range(kmax):
            a=2*jsp
            b=(2*jsp)+1
            cc[jsp]=(zs1[a,1].conjugate()*zsc[a,1]+zs1[a,0].conjugate()*zsc[a,0])*(zs1[b,1].conjugate()*zsc[b,1]+zs1[b,0].conjugate()*zsc[b,0])
            d[jsp]=(zs1[a,1].conjugate()*zsc[a,0])*(zs1[b,0].conjugate()*zsc[b,1])
        for jsp in range(kmax):
            if(jsp==0):
                temp=d[0]*numpy.prod(cc[1:])
            elif(jsp==kmax-1):
                temp=numpy.prod(cc[:(kmax-1)])*d[kmax-1]
            else:
                temp=numpy.prod(cc[:jsp])*d[jsp]*numpy.prod(cc[jsp+1:])
            #print(temp)
            tot=tot+temp
    return tot


def spsmfast(zs1,zs2,norb):
    """Fastest calcualtion of <zs1 |S_+S_- |zs2>"""
    kmax=int(norb/2)
    cc = numpy.zeros((kmax))
    dd = numpy.zeros((kmax))
    ss = numpy.zeros((kmax))
    tt= numpy.zeros((kmax))
    for i in range(kmax):
        a=2*i
        b=(2*i)+1
        cc[i]=(zs1[a,1].conjugate()*zs2[a,1]+zs1[a,0].conjugate()*zs2[a,0])*(zs1[b,1].conjugate()*zs2[b,1]+zs1[b,0].conjugate()*zs2[b,0])
        dd[i]=(zs1[a,0].conjugate()*zs2[a,1])*(zs1[b,1].conjugate()*zs2[b,0])
        ss[i]=(zs1[a,1].conjugate()*zs2[a,0])*(zs1[b,0].conjugate()*zs2[b,1])
        tt[i]=(zs1[a,1].conjugate()*zs2[a,1])*(zs1[b,0].conjugate()*zs2[b,0])
    tot=0.0 
    for i in range(kmax):
        for j in range(i, kmax):
            p1=0
            p2=0
            if(i==0):
                    p1=1
            elif(i==1):
                p1=cc[0]
            else:
                p1=numpy.prod(cc[:i])
            if(j==kmax-1):
                p2=1
            elif(j==kmax-2):
                p2=cc[kmax-1]
            else:
                p2=numpy.prod(cc[j+1:])
            if(j==i):
                tot=tot+(p1*p2*tt[i])
            elif(j==i+1):
                tot=tot+(p1*p2*ss[i]*dd[j])+(p1*p2*ss[j]*dd[i])
            elif(j==i+2):
                tot=tot+(p1*p2*ss[i]*dd[j]*cc[i+1])+(p1*p2*ss[j]*dd[i]*cc[i+1])
            else:
                p3=numpy.prod(cc[i+1:j])
                tot=tot+(p1*p2*ss[i]*dd[j]*p3)+(p1*p2*ss[j]*dd[i]*p3)

    return tot

#Total spin alogrithms 
def Stot1(zs1,zs2,norb):
    return spsmf(zs1,zs2,norb) -sz(zs1,zs2,norb)+sz2(zs1,zs2,norb)

def Stot2(zs1,zs2,norb):
    return spsm2(zs1,zs2,norb) - szf(zs1,zs2,norb) + sz2f(zs1,zs2,norb)

def Stotfast(zs1,zs2,norb):
    return spsmfast(zs1,zs2,norb) -szf(zs1,zs2,norb)+sz2f(zs1,zs2,norb)



def z_an_z3(zom1,zom2,norb,vec):
    """Finding sum_k <zom1 | b_k | zom2> vec_k for all k in norb
    Dynamically handles real vs complex input data
    Faster than the first two versions"""
    vmult = numpy.multiply(zom1.conjugate(),zom2)
    if type(vmult[0,0].item()) is complex:
        gg = numpy.zeros((norb),dtype=complex)
        hh = numpy.zeros((norb),dtype=complex)
    else:
        gg = numpy.zeros((norb),dtype=float)
        hh = numpy.zeros((norb),dtype=float)        
    gg[0] = vmult[0,0] - vmult[0,1]
    gmax = norb
    for ii in range(1,norb):
        gg[ii] = gg[ii-1]*(vmult[ii,0]-vmult[ii,1])
        if gg[ii] == 0.0:
            gmax = ii
            break
    hh[-1] = vmult[-1,0] + vmult[-1,1]   
    hmin = 0
    for ii in range(norb-2,-1,-1):
        hh[ii] = hh[ii+1]*(vmult[ii,0]+vmult[ii,1])
        if hh[ii] == 0.0:
            hmin = ii
            break
    #an_out = numpy.zeros(norb)
    an = 0.0
    if gmax < hmin:
        return 0.0
    if vec[0] != 0:
        an += zom1[0,0].conjugate()*zom2[0,1]*hh[1]*vec[0]
    for ii in range(1,norb-1,1):
        if vec[ii]!=0.0:
            an += gg[ii-1]*zom1[ii,0].conjugate()*zom2[ii,1]*hh[ii+1]*vec[ii]
    if vec[-1] !=0: 
        an += gg[-2]*zom1[-1,0].conjugate()*zom2[-1,1]*vec[-1]
    return an


# Trying a zombie class
class zom:
    """Class of a single zombie state"""
    def __init__(self,norb,typ='empty',occ=None,coefs=None, \
                 ib=None,nel=None,thetas=None):
        self.norb = norb
        # Possible types if initialisation
        if typ == 'empty':
            self.zs = make(self.norb)
            self.zs[:,0] = 0.0
        elif typ == 'occ':
            self.zs = new(self.norb,occ)
        elif typ == 'ran':
            self.zs = new_ran(self.norb)
        elif typ == 'coef':
            self.zs = coefs
        elif typ == 'binary':
            bini = numtodet(ib,self.norb)
            self.zs = new(self.norb,bini)
        elif typ == 'aufbau':
            occ = numpy.zeros((self.norb),dtype=int)
            occ[:nel] = 1
            self.zs = new(self.norb,occ)
        elif typ == 'theta':
            self.zs = make(self.norb)
            self.zs[:,0] = numpy.cos(2.0*math.pi*thetas)
            self.zs[:,1] = numpy.sin(2.0*math.pi*thetas)
        else:
            raise ValueError('Invalid type')
    def ov(self):
        return overlap_f(self.zs,self.zs)
    def cr(self,iorb):
        self.zs[iorb,1] = self.zs[iorb,0]
        self.zs[iorb,0] = 0.0
        self.zs[:iorb,1] *= -1.0
    def an(self,iorb):
        self.zs[iorb,0] = self.zs[iorb,1]
        self.zs[iorb,1] = 0.0
        self.zs[:iorb,1] *= -1.0
    def num(self):
        "Number of electrons in the zombie state"
        return numf(self.zs,self.zs)
    def isdet(self):
        return isdet(self.zs,self.norb)
    def sz(self):
        return szf(self.zs,self.zs,self.norb)

# Make a class for the Hamiltonian parameters and functions?
class system:
    """Class holding the system parameters for zombie state calculation"""
    def __init__(self, norb, Hnr, H1ei, H2ei):
        self.Hnr = Hnr
        self.H1ei = H1ei
        self.H2ei = H2ei
        self.norb = norb
        if norb%2 != 0:
            raise ValueError('norb must be even')
        self.nspao = int(norb/2)
    def Ham1z(self,zom1,zom2):
        Ht1 = 0.0
        for ii in range(self.norb):
            for jj in range(self.norb):
                zomt = numpy.copy(zom2)
                zomt = an(zomt,ii)
                zomt = cr(zomt,jj)
                ov = overlap(zom1,zomt)
                # print(ii,jj,ov)
                Ht1 += ov*self.H1ei[ii,jj]
        return Ht1
    def Ham2z_v3(self,zom1,zom2):
        """Algorithm to calculate the two-electron Hamiltonian bra-ket
        Between two zombie states
        Breaking up annihilation and creation calculation
        Scales as O(M^5) but with a low prefactor
        Needs some attention to deal with complex numbers"""
        Ht2 = 0.0
        Z1ij = numpy.zeros((self.norb,self.norb,self.norb,2),dtype=complex)
        Z2lk = numpy.zeros((self.norb,self.norb,self.norb,2),dtype=complex)
        for ii in range(self.norb):
            for jj in range(self.norb):
                zomt = numpy.copy(zom1)
                zomt = an(zomt,ii)
                zomt = an(zomt,jj)
                Z1ij[ii,jj,:,:] = zomt[:,:]
        for kk in range(self.norb):
            for ll in range(self.norb):
                zomt = numpy.copy(zom2)
                zomt = an(zomt,kk)
                zomt = an(zomt,ll)
                Z2lk[ll,kk,:,:] = zomt[:,:]
        for ii in range(self.norb):
            if zom1[ii,1] == 0.0:
                continue
            for jj in range(self.norb):
                if zom1[jj,1] == 0.0:
                    continue
                for kk in range(self.norb):
                    if zom2[kk,1] == 0.0:
                        continue 
                    for ll in range(self.norb):
                        if self.H2ei[ii,jj,kk,ll] == 0.0 or zom2[ll,1] == 0.0:
                            continue
                        ov = overlap_f(Z1ij[ii,jj,:,:],Z2lk[ll,kk,:,:])
                        if ov == 0.0:
                            continue
                        Ht2 += 0.5*ov*self.H2ei[ii,jj,kk,ll]
        return Ht2
    def Ham2z_v5(self,zom1,zom2):
        Ht2 = 0.0
        if type(zom1[0,0].item()) is complex:
            Z1ij = numpy.zeros((self.norb,self.norb,self.norb,2),dtype=complex)
        else:
            Z1ij = numpy.zeros((self.norb,self.norb,self.norb,2),dtype=float)
        if type(zom2[0,0].item()) is complex:
            Z2k = numpy.zeros((self.norb,self.norb,2),dtype=complex)
        else:
            Z2k = numpy.zeros((self.norb,self.norb,2),dtype=float)
        for ii in range(self.norb):
            for jj in range(self.norb):
                zomt = numpy.copy(zom1)
                zomt = an(zomt,ii)
                zomt = an(zomt,jj)
                Z1ij[ii,jj,:,:] = zomt[:,:]
        for kk in range(self.norb):
            zomt = numpy.copy(zom2)
            zomt = an(zomt,kk)
            Z2k[kk,:,:] = zomt[:,:]
        for ii in range(self.norb):
            if zom1[ii,1] == 0.0:
                continue
            ispin = ii%2
            for jj in range(self.norb):
                if iszero(Z1ij[ii,jj,:,:]):
                    continue
                jspin = jj%2
                for kk in range(ispin,self.norb,2):
                    if zom2[kk,1] == 0.0:
                        continue
                    Ht2 += z_an_z3(Z1ij[ii,jj,:,:],Z2k[kk,:,:], \
                                        self.norb,self.H2ei[ii,jj,kk,:])
        return 0.5*Ht2
    def HTot(self,zom1,zom2):
        H1et = self.Ham1z(zom1,zom2)
        H2et = self.Ham2z_v5(zom1,zom2)
        HH = H1et + H2et + self.Hnr*overlap_f(zom1,zom2)
        return HH

# Reading in data from Dmitry's files
def readin(filename,norb):
    file = open(filename,'r')
    if norb%2 != 0:
        raise ValueError('norb must be even')
    nspao = int(norb/2)
    H1e = numpy.zeros((nspao,nspao))
    H2e = numpy.zeros((nspao,nspao,nspao,nspao))
    Hnr = 0.0
    for line in file:
        bits = line.split()
        #print(len(bits))
        en = float(bits[0])
        i = int(bits[1]) - 1
        j = int(bits[2]) - 1
        k = int(bits[3]) - 1
        l = int(bits[4]) - 1
        #print(i,j,k,l,en)
        if k != -1 and l != -1:
            H2e[i,j,k,l] = en
            H2e[j,i,k,l] = en
            H2e[i,j,l,k] = en
            H2e[j,i,l,k] = en
            H2e[k,l,i,j] = en
            H2e[l,k,i,j] = en
            H2e[k,l,j,i] = en
            H2e[l,k,j,i] = en
            # print('2e',i,j,k,l,en)
        elif k == -1 and l == -1 and i != -1 and j != -1:
            H1e[i,j] = en
            if i != j:
                H1e[j,i] = en
        elif i == -1 and j == -1 and k == -1 and l == -1:
            Hnr = en
            # print('nucr',en)
        else:
            print('error',bits)
    H1ei = numpy.zeros((norb,norb))
    H2ei = numpy.zeros((norb,norb,norb,norb))
    for i in range(nspao):
        for j in range(nspao):
            ii = i*2
            jj = j*2
            H1ei[ii,jj] = H1e[i,j] # alpha spin
            H1ei[ii+1,jj+1] = H1e[i,j] # beta spin
            for k in range(nspao):
                for l in range(nspao):
                    kk = k*2
                    ll = l*2
                    Ht = H2e[i,j,k,l]
                    H2ei[ii,kk,jj,ll] = Ht
                    H2ei[ii+1,kk,jj+1,ll] = Ht
                    H2ei[ii,kk+1,jj,ll+1] = Ht
                    H2ei[ii+1,kk+1,jj+1,ll+1] = Ht
    return Hnr, H1ei, H2ei

def spatospin(H1ea,H2ea,norb):
    H1ei = spatospin1(H1ea,norb)
    H2ei = spatospin2(H2ea,norb)
    return H1ei, H2ei

def spatospin1(H1ea,norb):
    """Converting H1ea from spatial to spin"""
    if norb%2 != 0:
        raise ValueError('norb must be even')
    nspao = int(norb/2)
    H1ei = numpy.zeros((norb,norb))
    for i in range(nspao):
        for j in range(nspao):
            ii = i*2
            jj = j*2
            H1ei[ii,jj] = H1ea[i,j] # alpha spin
            H1ei[ii+1,jj+1] = H1ea[i,j] # beta spin
    return H1ei

def spatospin2(H2ea,norb):
    """Converting H2ea from spatial orbitals to spin orbitals
    where the spatial orbitals are in the chemist's notation"""
    if norb%2 != 0:
        raise ValueError('norb must be even')
    nspao = int(norb/2)
    H2ei =  numpy.zeros((norb,norb,norb,norb))
    for i in range(nspao):
        for j in range(nspao):
            ii = i*2
            jj = j*2
            for k in range(nspao):
                for l in range(nspao):
                    kk = k*2
                    ll = l*2
                    Ht = H2ea[i,j,k,l]
                    H2ei[ii,kk,jj,ll] = Ht
                    H2ei[ii+1,kk,jj+1,ll] = Ht
                    H2ei[ii,kk+1,jj,ll+1] = Ht
                    H2ei[ii+1,kk+1,jj+1,ll+1] = Ht
    return H2ei

def pyscfint(mol,myhf, norb):
    """Obtaining one and two electron integrals from pyscf calculation
    Code adapted from George Booth"""
    # Extract AO->MO transformation matrix
    c = myhf.mo_coeff
    # Get 1-electron integrals and convert to MO basis
    h1e = reduce(numpy.dot, (c.T, myhf.get_hcore(), c))
    # Get 2-electron integrals and transform them
    eri = ao2mo.kernel(mol, c)
    # Ignore all permutational symmetry, and write as four-index tensor, in chemical notation
    eri_full = ao2mo.restore(1, eri, c.shape[1])
    # Scalar nuclear repulsion energy
    Hnuc = myhf.energy_nuc()
    # Now convert from sppatial to spin orbitals
    H1ei, H2ei = spatospin(h1e,eri_full,norb)
    return Hnuc, H1ei, H2ei

def normv(vec):
    """Normalizes a vector"""
    fac = numpy.dot(vec.conjugate(),vec)
    vec2 = vec/math.sqrt(fac)
    return vec2

def mu(zom1,zom2,Mumat,norb):
    """Computes dipole moment"""
    temp = 0.0
    for iorb in range(norb):
        for jorb in range(norb):
            zt = numpy.copy(zom2)
            zt = an(zt,iorb)
            zt = cr(zt,jorb)
            ov = overlap_f(zom1,zt)
            temp += Mumat[iorb,jorb]*ov
    return temp

def MakeMuMat(norb,Muspa):
    """Converts a top diagonal *spatial* dipole moment matrix
    To one in spin notation
    Muspa is the size norb/2, norb/2
    Only elements Muspa[i,j] where j>1 are used"""
    if norb%2 !=0 :
        raise ValueError('norb must be even')
    nspa = norb//2
    Mumat = numpy.zeros((norb,norb))
    for ispa in range(nspa):
        for jspa in range(ispa,nspa):
            if Muspa[ispa,jspa] == 0:
                continue
            ispi = ispa*2
            jspi = jspa*2
            Mumat[ispi,jspi] = Muspa[ispa,jspa]
            Mumat[ispi+1,jspi+1] = Muspa[ispa,jspa]
            Mumat[jspi,ispi] = Muspa[ispa,jspa]
            Mumat[jspi+1,ispi+1] = Muspa[ispa,jspa]
    return Mumat
