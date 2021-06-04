# Diagonalising the FCI Hamiltonian
import math, numpy
import scipy.sparse.linalg as scisp
import zombie

ndet = 1024
Bigham = numpy.zeros((ndet,ndet))
hamin = open('Ham_anion.out','r')

for line in hamin:
    bits = line.split()
    i1 = int(bits[0])
    i2 = int(bits[1])
    ht = float(bits[4])
    print
    Bigham[i1,i2] = ht
    if i1 != i2:
        Bigham[i2,i1] = ht


nei = 10
Eival, Eivec = scisp.eigsh(Bigham,k = nei, which='SA')
#Eival, Eivec = numpy.linalg.eig(Bigham)
print(Eival[:nei])

tol = 0.01
for ieig in range(nei):
    idum = 1
    for idet in range(ndet):
        if abs(Eivec[idet,ieig]) > tol:
            zz = zombie.new(10,zombie.numtodet(idet,10))
            sp = zombie.szf(zz,zz,10)
            ch = zombie.numf(zz,zz) #- 7.0
            # if ch != 0.0:
            #     break
            if idum == 1:
                print('Eigenstate',ieig,'Energy',Eival[ieig])
                #break
                print('DetNo|      Occupancy      | M_s |Charge| Coefficient')
                idum = 0
            string = '{0:>5d} {1} {2:+5.1f} {3:+6.1f} {4:<+10.7f}'.\
                format(idet,zombie.numtodet(idet,10),sp,ch,Eivec[idet,ieig])
            print(string)
            #print(idet,zombie.numtodet(idet,10),Eivec[idet,ieig])
            
