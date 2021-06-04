import math, numpy
import scipy.sparse.linalg as scisp
from matplotlib import pyplot as plt
import matplotlib as mpl
from pylab import cm
import zombie

# Reading in data
ndet = 64
norb = 10
Bigham = numpy.zeros((ndet,ndet))
Kover = numpy.zeros((ndet,ndet))

hamin = open('biassed_64BF_14_molpro_noHF.dat','r')

for line in hamin:
    bits = line.split(',')
    i1 = int(bits[0])
    i2 = int(bits[1])
    ko = float(bits[2])
    ht = float(bits[3])
    Bigham[i1,i2] = ht
    Kover[i1,i2] = ko
    if i1 != i2:
        Bigham[i2,i1] = ht
        Kover[i2,i1] = ko



print('read in')
BHr=Bigham
Kr=Kover
ndr=ndet
# Create inverse overlap matrix in the reduced basis
Kir = numpy.linalg.inv(Kr)

numpy.random.seed(1)
zstore=[]
for idet in range(ndet):
    zstore.append(zombie.zom(norb,typ='ran',ib=idet))
zrhf = zombie.zom(norb,typ='aufbau',nel=6)
dtcr = numpy.zeros((ndr))
for idet in range(ndr):
    dtcr[idet] = zombie.overlap_f(zrhf.zs,zstore[idet].zs)

dvec=numpy.dot(Kir,dtcr)
nst = 4
dvecs = numpy.zeros((ndr,nst))
dvecs[:,0]=dvec
# initialize values
for ist in range(0,nst):
    dvecs[ist:,ist] = 1.0

# Check energy
ens = numpy.einsum('ik,ij,jk->k',dvecs,BHr,dvecs)
ovis = numpy.einsum('ik,ij,jl->kl',dvecs,Kr,dvecs)
print('en', ens)
print('ovi')
print(ovis)

def gs(dvecs,Kr,nst,ndr):
    # Gram-Schmidt orthogonalization
    uecs = numpy.copy(dvecs)
    for ist in range(1,nst):
        for jst in range(0,ist):
            # dvecs[:,ist]*Kr*uvecs[:,jst]
            numer = numpy.dot(dvecs[:,ist],numpy.dot(Kr,uecs[:,jst]))
            den = numpy.dot(uecs[:,jst],numpy.dot(Kr,uecs[:,jst]))
            uecs[:,ist] -= uecs[:,jst]*numer/den
    for ist in range(nst):
        fac = numpy.dot(uecs[:,ist],numpy.dot(Kr,uecs[:,ist]))
        uecs[:,ist] *= fac**-0.5
    return uecs

dvecs = gs(dvecs,Kr,nst,ndr)
ens = numpy.einsum('ik,ij,jk->k',dvecs,BHr,dvecs)
ovis = numpy.einsum('ik,ij,jl->kl',dvecs,Kr,dvecs)
print('en', ens)
print('ovi')
print(ovis)

# Find exact eigenstate
# Firstly diagonalise Kover
Dei, Deivec = numpy.linalg.eigh(Kover)
# Now form A matrix
Amat = numpy.zeros((ndet,ndet))
for idet in range(ndet):
    Amat[:,idet] = Deivec[:,idet]/math.sqrt(Dei[idet])
Ainv = numpy.linalg.inv(Amat)
# Tmat = numpy.matmul(Amat.T,numpy.matmul(Kover,Amat))
# print(numpy.allclose(Tmat,numpy.eye(ndet)))
# Now form A.T H A
HT = numpy.matmul(Amat.T,numpy.matmul(Bigham,Amat))
Eival, Eivec = numpy.linalg.eigh(HT)
vec = Eivec[:,2]
vect = numpy.dot(Amat,vec)

# Now to propagate in imaginary time
beta = 50000
nb = 20000
KiHr = numpy.matmul(Kir,BHr)
db = beta/nb
dvecs0 = dvecs
#den = numpy.einsum('i,ij,j',dvec0,BHr,dvec0)
dvecsl = numpy.zeros((ndet,nst))
dvecsl[:ndr,:] = dvecs
ov0 = numpy.einsum('i,ij,jk->k',vect,Kover,dvecsl)
eb = numpy.zeros((nb+1,nst))
ov = numpy.zeros((nb+1,nst))
eb[0,:] = ens
ov[0,:] = ov0

nprint = 100
ng = nb // nprint + 1

print(Eival[0])
print(Eival[1])
print(Eival[2])
print(Eival[3])

for ib in range(nb):
    ddots = -numpy.dot(KiHr,dvecs)
    dvecs = dvecs + db*ddots
    dvecs = gs(dvecs,Kr,nst,ndr)
    ens = numpy.einsum('ik,ij,jk->k',dvecs,BHr,dvecs)
    ovis = numpy.einsum('ik,ij,jl->kl',dvecs,Kr,dvecs)
    eb[ib+1,:] = ens
    dvecsl = numpy.zeros((ndet,nst))
    dvecsl[:ndr,:] = dvecs
    ov[ib+1,:] = numpy.einsum('i,ij,jk->k',vect,Kover,dvecsl)
    if (ib+1)%nprint == 0:
        print( (ib+1)*db )
        print(eb[ib+1,:])
        print(ov[ib+1,:])

print(eb[nb,0])
print(eb[nb,1])
print(eb[nb,2])
print(eb[nb,3])
mpl.rcParams['font.family']='Avenir'
plt.rcParams['font.size']=18
plt.rcParams['axes.linewidth']=2
colors =cm.get_cmap('Set1',5)
fig=plt.figure(figsize=(3.37,5.055))
ax=fig.add_axes([0,0,2,1])
ax.plot(numpy.linspace(0,beta,num=nb + 1),eb[:,0],label='State 1',color=colors(0),linewidth=2)
ax.plot(numpy.linspace(0,beta,num=nb + 1),eb[:,1],label='State 2',color=colors(1),linewidth=2)
ax.plot(numpy.linspace(0,beta,num=nb + 1),eb[:,2],label='State 3',color=colors(2),linewidth=2)
ax.plot(numpy.linspace(0,beta,num=nb + 1),eb[:,3],label='State 4',color=colors(3),linewidth=2)
ax.plot(numpy.linspace(db,beta,num=nb+1),numpy.ones((nb+1))*-14.876024486659754,linewidth=2, linestyle='dashed', color=colors(4))
ax.plot(numpy.linspace(db,beta,num=nb+1),numpy.ones((nb+1))*-14.869949868878516,linewidth=2, linestyle='dashed', color=colors(4))
ax.plot(numpy.linspace(db,beta,num=nb+1),numpy.ones((nb+1))*-14.855416640029112,linewidth=2, linestyle='dashed', color=colors(4))
ax.set_xlim(0,50000)
ax.legend(loc='best')
ax.set_ylim(-14.8775,-14.8550)
ax.set_ylabel('Energy',labelpad=10)
ax.set_xlabel(r'$\mathregular{\beta}$',labelpad=10)
plt.savefig('GS_64BF.png',dpi=300, transparent=False,bbox_inches='tight')


exit()