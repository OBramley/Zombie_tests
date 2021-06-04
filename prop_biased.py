import math, numpy
import scipy.sparse.linalg as scisp
from matplotlib import pyplot as plt
import matplotlib as mpl
from pylab import cm
import zombie
import random
import csv


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
#print(Kover)

ndr=ndet
# RHF determinant
zrhf = zombie.zom(norb,typ='aufbau',nel=7)

numpy.random.seed(1)
zstore=[]
for idet in range(ndet):
    zstore.append(zombie.zom(norb,typ='ran',ib=idet))
print(zstore[1].zs)  
#Create the reduced basis

BHr=Bigham
Kr=Kover
Kri = numpy.linalg.inv(Kr)

# Overlap with RHF determinant
dtcr=numpy.zeros((ndr))
for idet in range(ndr):
    dtcr[idet] = zombie.overlap_f(zrhf.zs,zstore[idet].zs)
dvec = numpy.dot(Kri,dtcr)
fac = abs(numpy.einsum('i,ij,j',dvec,Kr,dvec))
dvec /= math.sqrt(fac)

# dvec=numpy.zeros((ndr))
# dvec[0]=1.0

ovr = numpy.einsum('i,ij,j',dvec,Kr,dvec)
print('ovr',ovr)
#overlap quality
# qual=numpy.dot(dvec,dtcr)
# print('qual',qual,fac**0.5)

# Check energy
en = numpy.einsum('i,ij,j',dvec,BHr,dvec)
ovi = numpy.einsum('i,ij,j',dvec,Kr,dvec)
print('en', en, 'ovi', ovi)

# dnew=clean(Kri,SPhi,dvec)
# dvec=dnew

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
print(Eival)

vec = Eivec[:,0]
vect = numpy.dot(Amat,vec)




# Now to propagate in imaginary time
beta = 2000.0
nb = 5000
KinvH = numpy.matmul(Kri,BHr)
db = beta/nb
dvec0 = dvec
den = numpy.einsum('i,ij,j',dvec0,BHr,dvec0)
dvecl=numpy.zeros((ndet))
dvecl[:ndr]=dvec
ov0 = numpy.einsum('i,ij,j',vect,Kover,dvecl)
eb = numpy.zeros((nb))
ov = numpy.zeros((nb))
# nprint=500
# ng=nb//nprint+1
# fciv=numpy.zeros(ng)
# dnew=clean(Kri,SPhi,dvec)
# dnewl=numpy.zeros((ndet))
# dnewl[:ndr]=dnew
# fciv[0]=numpy.einsum('i,ij,j',vect,Kover,dnewl)
# dvec=dnew
print('Beta |     Energy    |    E - E_FCI   |  Overlap')
string = '{:>5.0f} {:15.11f} {:15.13f} {:15.13f}'.format(0,den,den-Eival[0],ov0)
print(string)

for ib in range(nb):
    ddot = -numpy.dot(KinvH,dvec)
    dvec = dvec + db*ddot
    norm = abs(numpy.einsum('i,ij,j',dvec,Kr,dvec))
    dvec /= math.sqrt(norm)
    den = numpy.einsum('i,ij,j',dvec,BHr,dvec)
    eb[ib] = den
    dvecl=numpy.zeros((ndet))
    dvecl[:ndr]=dvec
    ov[ib] = numpy.einsum('i,ij,j',vect,Kover,dvecl)  
    if (ib+1)%100 == 0:
        string = '{:>5.0f} {:15.11f} {:15.13f} {:15.13f}'.format((ib+1)*db,eb[ib],eb[ib]-Eival[0],ov[ib])
        print(string)

print(Eival[0])
print(Eival[1])
print(Eival[2])
# dnew=clean(Kri,SPhi,dvec)
# ddot = -numpy.dot(KinvH,dvec)
# dnew=dnew +db*ddot
# norm = abs(numpy.einsum('i,ij,j',dnew,Kr,dnew))
# dnew /= math.sqrt(norm)
# den = numpy.einsum('i,ij,j',dnew,BHr,dnew)
# print(den)
# ergcopy=numpy.zeros(nb+1)
# ergcopy[:nb]=eb
# ergcopy[nb]=den
# print(ergcopy[nb])
# lazy=numpy.zeros(nb+1)
# for i in range(nb+1):
#     lazy[i]=i
# plt.plot(lazy,ergcopy)
# plt.plot(lazy,numpy.ones((nb+1))*-14.869949868878578)
# plt.plot(numpy.linspace(db,beta,num=nb),eb)
# plt.plot(numpy.linspace(db,beta,num=nb),numpy.ones((nb))*Eival[1])
# plt.ylabel('Energy')
# plt.xlabel('beta')
# plt.show()


mpl.rcParams['font.family']='Avenir'
plt.rcParams['font.size']=18
plt.rcParams['axes.linewidth']=2
colors =cm.get_cmap('Set1',2)
fig=plt.figure(figsize=(3.37,5.055))
ax=fig.add_axes([0,0,2,1])
ax.plot(numpy.linspace(db,beta,num=nb),eb, linewidth=2, color=colors(0))
ax.plot(numpy.linspace(db,beta,num=nb),numpy.ones((nb))*-14.876024486659754,linewidth=2, color=colors(1))
ax.set_xlim(0,2000)
#ax.set_ylim(-14.8615,-14.8575)
ax.set_ylabel('Energy',labelpad=10)
ax.set_xlabel(r'$\mathregular{\beta}$',labelpad=10)
plt.savefig('biased_30_prop_14_nohf.png',dpi=300, transparent=False,bbox_inches='tight')

exit()