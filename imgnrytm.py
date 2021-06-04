import math, numpy
import scipy.sparse.linalg as scisp
from matplotlib import pyplot as plt
import matplotlib as mpl
from pylab import cm
import zombie

ndet = 1024
norb = 10
Bigham = numpy.zeros((ndet,ndet))
Kover = numpy.zeros((ndet,ndet))

hamin = open('Ham.out','r')

for line in hamin:
    bits = line.split()
    i1 = int(bits[0])
    i2 = int(bits[1])
    ht = float(bits[4])
    Bigham[i1,i2] = ht
    if(i1==i2):
        Kover[i1,i2] = 1
    if i1 != i2:
        Bigham[i2,i1] = ht


nei = 10
Eival, Eivec = scisp.eigsh(Bigham,k = nei, which='SA')
print(Eival[:nei])

# RHF determinant
zrhf = zombie.zom(norb,typ='aufbau',nel=7)

zstore=[]
for idet in range(ndet):
    zstore.append(zombie.zom(norb,typ='binary',ib=idet))

dtc=numpy.zeros((ndet))
for idet in range(ndet):
    dtc[idet] = zombie.overlap_f(zrhf.zs,zstore[idet].zs)

Kinv = numpy.linalg.inv(Kover)
dvec = numpy.dot(Kinv,dtc)

# Check energy
en = numpy.einsum('i,ij,j',dvec,Bigham,dvec)
ovi = numpy.einsum('i,ij,j',dvec,Kover,dvec)
print('en', en, 'ovi', ovi)

# Find exact eigenstate
# Firstly diagonalise Kover
Dei, Deivec = numpy.linalg.eigh(Kover)
# Now form A matrix
Amat = numpy.zeros((ndet,ndet))
for idet in range(ndet):
    Amat[:,idet] = Deivec[:,idet]/math.sqrt(Dei[idet])
Ainv = numpy.linalg.inv(Amat)
Tmat = numpy.matmul(Amat.T,numpy.matmul(Kover,Amat))
# Now form A.T H A
HT = numpy.matmul(Amat.T,numpy.matmul(Bigham,Amat))
Eival, Eivec = numpy.linalg.eigh(HT)
vec = Eivec[:,2]
vect = numpy.dot(Amat,vec)

# Now to propagate in imaginary time
beta = 200.0
nb = 100
KinvH = numpy.matmul(Kinv,Bigham)
db = beta/nb
dvec0 = dvec
den = numpy.einsum('i,ij,j',dvec0,Bigham,dvec0)
ov0 = numpy.einsum('i,ij,j',vect,Kover,dvec0)
eb = numpy.zeros((nb))
ov = numpy.zeros((nb))
print('Beta |     Energy    |    E - E_FCI   |  Overlap')
string = '{:>5.0f} {:15.11f} {:15.13f} {:15.13f}'.format(0,den,den-Eival[2],ov0)
print(string)
for ib in range(nb):
    ddot = -numpy.dot(KinvH,dvec)
    dvec = dvec + db*ddot
    norm = numpy.einsum('i,ij,j',dvec,Kover,dvec)
    dvec /= norm**0.5
    den = numpy.einsum('i,ij,j',dvec,Bigham,dvec)
    eb[ib] = den
    ov[ib] = numpy.einsum('i,ij,j',vect,Kover,dvec)
    #print(ib,ib*db,eb[ib],ov[ib])
    if (ib+1)%100 == 0:
        string = '{:>5.0f} {:15.11f} {:15.13f} {:15.13f}'.format((ib+1)*db,eb[ib],eb[ib]-Eival[2],ov[ib])
        print(string)


print(Eival[0])
print(Eival[1])
print(Eival[2])
mpl.rcParams['font.family']='Avenir'
plt.rcParams['font.size']=18
plt.rcParams['axes.linewidth']=2
colors =cm.get_cmap('Set1',2)
fig=plt.figure(figsize=(3.37,5.055))
ax=fig.add_axes([0,0,2,1])
ax.plot(numpy.linspace(db,beta,num=nb),eb, linewidth=2, color=colors(0))
ax.plot(numpy.linspace(db,beta,num=nb),numpy.ones((nb))*Eival[1],linewidth=2, color=colors(1))
ax.set_xlim(0,200)
#ax.set_ylim(-14.8615,-14.8575)
ax.set_ylabel('Energy',labelpad=10)
ax.set_xlabel(r'$\mathregular{\beta}$',labelpad=10)
plt.savefig('imaginaryslater2.png',dpi=300, transparent=False,bbox_inches='tight')
#plt.show()    
exit()