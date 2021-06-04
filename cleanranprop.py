import math, numpy
import scipy.sparse.linalg as scisp
from matplotlib import pyplot as plt
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



ndr=1024
# RHF determinant
zrhf = zombie.zom(norb,typ='aufbau',nel=6)

numpy.random.seed(1)
zstore=[]
for idet in range(ndet):
    zstore.append(zombie.zom(norb,typ='ran',ib=idet))

# Create Slater determinant basis
sdstore = []
nel = 6
SPhi = numpy.zeros((ndet,ndr))
for idet in range(ndet):
    sdstore.append(zombie.zom(norb,typ='binary',ib = idet))
    if zombie.numf(sdstore[idet].zs,sdstore[idet].zs) == nel:
        for jdet in range(ndr):
            SPhi[idet,jdet] = zombie.overlap_f(sdstore[idet].zs,zstore[jdet].zs)


#create the reduced basis


# BHr=numpy.zeros((ndr,ndr))
# Kr=numpy.zeros((ndr,ndr))

#random sample
# BHr=numpy.zeros((ndr,ndr))
# Kr=numpy.zeros((ndr,ndr))

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

BHr=Bigham
Kr=Kover



#Overlap with RHF determinant
dtcr=numpy.zeros((ndr))
for idet in range(ndr):
    dtcr[idet] = zombie.overlap_f(zrhf.zs,zstore[idet].zs)
Kri = numpy.linalg.inv(Kr)

dvec = numpy.zeros((ndr))
initype='random'
if(initype=='rhf'):
    dvec = numpy.dot(Kri,dtcr)
    fac = numpy.einsum('i,ij,j',dvec,Kr,dvec)
    dvec /= math.sqrt(fac)
    ovr = numpy.einsum('i,ij,j',dvec,Kr,dvec)
    print('ovr',ovr)
    #overlap quality
    qual=numpy.dot(dvec,dtcr)
    print('qual',qual,fac**0.5)
elif(initype=='random'):
    # Try random dvec
    dvec[:] = 0.0
    dvec[1] = 1.0
else:
    exit()
# Check energy
en = numpy.einsum('i,ij,j',dvec,BHr,dvec)
ovi = numpy.einsum('i,ij,j',dvec,Kr,dvec)
print('en', en, 'ovi', ovi)


# SPhi perhaps the best way as big matrix only computed once
redv = numpy.dot(SPhi,dvec)
#redv = numpy.multiply(sdv,redv)
redv = zombie.normv(redv)
print('SPhi calc', redv[63])
#dnew = numpy.einsum('ij,kj,kl,l',Kri,SPhi,SPhi,dvec)
dnew = numpy.dot(SPhi.T,redv)
dnew = numpy.dot(Kri,dnew)
print('einsum',numpy.dot(SPhi[63,:],dnew))
print('norm',numpy.einsum('i,ij,j',dnew,Kr,dnew))
CleanMat = numpy.matmul(Kri,SPhi.T)

def clean(Kri,SPhi,dvec):
    redv = numpy.dot(SPhi,dvec)
    redv = zombie.normv(redv)
    dnew = numpy.dot(SPhi.T,redv)
    dnew = numpy.dot(Kri,dnew)
    return dnew

dnew = clean(Kri,SPhi,dvec)
print('clean',numpy.dot(SPhi[63,:],dnew))
print('norm',numpy.einsum('i,ij,j',dnew,Kr,dnew))

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
beta = 1000.0
nb = 5000
KinvH = numpy.matmul(Kri,BHr)
db = beta/nb
dvec0 = dvec
den = numpy.einsum('i,ij,j',dvec0,BHr,dvec0)
dvecl=numpy.zeros((ndet))
dvecl[:ndr]=dvec
ov0 = numpy.einsum('i,ij,j',vect,Kover,dvecl)
eb = numpy.zeros((nb+1))
ov = numpy.zeros((nb+1))
eb[0]=den
ov[0]=ov0

nprint=500
ng=nb//nprint+1
fciv=numpy.zeros(ng)
dnew=clean(Kri,SPhi,dvec)
dnewl=numpy.zeros((ndet))
dnewl[:ndr]=dnew
fciv[0]=numpy.einsum('i,ij,j',vect,Kover,dnewl)
dvec=dnew
print('Beta |     Energy    |    E - E_FCI   | FCI Overlap   | 6e Overlap w FCI')
string = '{:>5.0f} {:15.11f} {:15.12f} {:15.12f} {:15.12f}'.format(0,den,den-Eival[2],ov0,fciv[0])
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
    #print(ib,ib*db,eb[ib],ov[ib])
    if (ib+1)%nprint == 0:
        dnew = clean(Kri,SPhi,dvec)
        ig = (ib+1)//nprint
        dnewl = numpy.zeros((ndet))
        dnewl[:ndr] = dnew
        fciv[ig] = numpy.einsum('i,ij,j',vect,Kover,dnewl)
        dvec=dnew
        string = '{:>5.0f} {:+15.11f} {:+15.12f} {:+15.12f} {:+15.12f}'.format((ib)*db,eb[ib],eb[ib]-Eival[2],ov[ib],fciv[ig])
        print(string)
        # Clean the wavefunction
        


plt.plot(numpy.linspace(0,beta,num=nb+1),eb)
plt.plot(numpy.linspace(db,beta,num=nb),numpy.ones((nb))*Eival[2])
#plt.axis([0,200,-15,-14])
plt.ylabel('Energy')
plt.xlabel('beta')
plt.show()
    
exit()