import math, numpy
import scipy.sparse.linalg as scisp
from matplotlib import pyplot as plt
import matplotlib as mpl
from pylab import cm
import zombie
import random
import pandas as pd

ndet = 1024
norb = 10
Bigham = numpy.zeros((ndet,ndet))
Kover = numpy.zeros((ndet,ndet))

hamin = open('ran_ham.dat','r')

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


ndr=ndet
# RHF determinant
zrhf = zombie.zom(norb,typ='aufbau',nel=6)

numpy.random.seed(1)
zstore=[]
for idet in range(ndet):
    zstore.append(zombie.zom(norb,typ='ran',ib=idet))

#Create the reduced basis

# BHr=numpy.zeros((ndr,ndr))
# Kr=numpy.zeros((ndr,ndr))
BHr=Bigham
Kr=Kover
# #random sample

# #Random list of rows to pick reduced values from
# rows=random.sample(range(0,ndet),ndr)
# for val in range(0,ndr):
#     #Diag is the row number in Bigham values will be chosen from
#     diag=rows[val]
#     #Diagonal value set
#     BHr[val,val]=Bigham[diag,diag]
#     Kr[val,val]=Kover[diag,diag]
#     start=val+1
#     #Routine to fill rest of row from place ajacent to diagonal value   
#     for i in range(start,ndr):
#         pos=rows[i]
#         BHr[i,val]=Bigham[pos,diag]
#         Kr[i,val]=Kover[pos,diag]
#         BHr[val,i]=Bigham[pos,diag]
#         Kr[val,i]=Kover[pos,diag]
    
 
# Overlap with RHF determinant
dtcr=numpy.zeros((ndr))
for idet in range(ndr):
    dtcr[idet] = zombie.overlap_f(zrhf.zs,zstore[idet].zs)
Kri = numpy.linalg.inv(Kr)
dvec = numpy.dot(Kri,dtcr)
fac = numpy.einsum('i,ij,j',dvec,Kr,dvec)
dvec /= math.sqrt(fac)
ovr = numpy.einsum('i,ij,j',dvec,Kr,dvec)
print('ovr',ovr)
#overlap quality
qual=numpy.dot(dvec,dtcr)
print('qual',qual,fac**0.5)

# Check energy
en = numpy.einsum('i,ij,j',dvec,BHr,dvec)
ovi = numpy.einsum('i,ij,j',dvec,Kr,dvec)
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
nb = 2000
KinvH = numpy.matmul(Kri,BHr)
db = beta/nb
dvec0 = dvec
den = numpy.einsum('i,ij,j',dvec0,BHr,dvec0)
dvecl=numpy.zeros((ndet))
dvecl[:ndr]=dvec
ov0 = numpy.einsum('i,ij,j',vect,Kover,dvecl)
eb = numpy.zeros((nb))
ov = numpy.zeros((nb))
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

# print(Eival[1])
print(Eival[2])
# print(Eival[0])
mpl.rcParams['font.family']='Avenir'
plt.rcParams['font.size']=18
plt.rcParams['axes.linewidth']=2
colors =cm.get_cmap('Set1',2)
fig=plt.figure(figsize=(3.37,5.055))
ax=fig.add_axes([0,0,2,1])
ax.plot(numpy.linspace(db,beta,num=nb),eb, linewidth=2, color=colors(0))
ax.plot(numpy.linspace(db,beta,num=nb),numpy.ones((nb))*Eival[2],linewidth=2, color=colors(1))
ax.set_xlim(0,200)
#ax.set_ylim(-14.8615,-14.8575)
ax.set_ylabel('Energy',labelpad=10)
ax.set_xlabel(r'$\mathregular{\beta}$',labelpad=10)
plt.savefig('random_imag_full_prop.png',dpi=300, transparent=False,bbox_inches='tight')
    
exit()