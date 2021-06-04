import math, numpy
import scipy.sparse.linalg as scisp
from matplotlib import pyplot as plt
import random
import matplotlib as mpl
from pylab import cm
import zombie

ndet1 = 1024
ndet2 = 1024
# ndet3 = 1024
norb = 10
Bigham1 = numpy.zeros((ndet1,ndet1))
Kover1 = numpy.zeros((ndet1,ndet1))
Bigham2 = numpy.zeros((ndet2,ndet2))
Kover2 = numpy.zeros((ndet2,ndet2))
# Bigham5 = numpy.zeros((ndet3,ndet3))
# Kover5 = numpy.zeros((ndet3,ndet3))

hamin1 = open('Ham.out','r')

# for line in hamin1:
#     bits = line.split(',')
#     i1 = int(bits[0])
#     i2 = int(bits[1])
#     ko = float(bits[2])
#     ht = float(bits[3])
#     Bigham1[i1,i2] = ht
#     Kover1[i1,i2] = ko
#     if i1 != i2:
#         Bigham1[i2,i1] = ht
#         Kover1[i2,i1] = ko

for line in hamin1:
    bits = line.split()
    i1 = int(bits[0])
    i2 = int(bits[1])
    ht = float(bits[4])
    Bigham1[i1,i2] = ht
    if(i1==i2):
        Kover1[i1,i2] = 1
    if i1 != i2:
        Bigham1[i2,i1] = ht

hamin2 = open('ran_ham.dat','r')

for line in hamin2:
    bits = line.split(',')
    i1 = int(bits[0])
    i2 = int(bits[1])
    ko = float(bits[2])
    ht = float(bits[3])
    Bigham2[i1,i2] = ht
    Kover2[i1,i2] = ko
    if i1 != i2:
        Bigham2[i2,i1] = ht
        Kover2[i2,i1] = ko

# hamin3 = open('ran_ham.dat','r')

# for line in hamin3:
#     bits = line.split(',')
#     i1 = int(bits[0])
#     i2 = int(bits[1])
#     ko = float(bits[2])
#     ht = float(bits[3])
#     Bigham5[i1,i2] = ht
#     Kover5[i1,i2] = ko
#     if i1 != i2:
#         Bigham5[i2,i1] = ht
#         Kover5[i2,i1] = ko

# ndr1=ndet1
# ndr2=ndet2
# ndr3=50
# ndr4=30
# RHF determinant
zrhf1 = zombie.zom(norb,typ='aufbau',nel=7)
zrhf2 = zombie.zom(norb,typ='aufbau',nel=7)
# zrhf3 = zombie.zom(norb,typ='aufbau',nel=7)
# zrhf4=zrhf3
numpy.random.seed(1)
zstore1=[]
for idet in range(ndet1):
    # zstore1.append(zombie.zom(norb,typ='ran'))
    zstore1.append(zombie.zom(norb,typ='binary',ib=idet))

zstore2 = []
for idet in range(ndet2):
    zstore2.append(zombie.zom(norb,typ='ran'))

# zstore3 = []
# for idet in range(ndet3):
#     zstore3.append(zombie.zom(norb,typ='ran'))

# zstore4=zstore3

dtc1=numpy.zeros((ndet1))
for idet in range(ndet1):
    dtc1[idet] = zombie.overlap_f(zrhf1.zs,zstore1[idet].zs)

dtc2=numpy.zeros((ndet2))
for idet in range(ndet2):
    dtc2[idet] = zombie.overlap_f(zrhf2.zs,zstore2[idet].zs)

# dtc3=numpy.zeros((ndr3))
# for idet in range(ndr3):
#     dtc3[idet] = zombie.overlap_f(zrhf3.zs,zstore3[idet].zs)

# dtc4=numpy.zeros((ndr4))
# for idet in range(ndr4):
#     dtc4[idet] = zombie.overlap_f(zrhf4.zs,zstore4[idet].zs)



# Bigham3=numpy.zeros((ndr3,ndr3))
# Kover3=numpy.zeros((ndr3,ndr3))
# Bigham4=numpy.zeros((ndr4,ndr4))
# Kover4=numpy.zeros((ndr4,ndr4))

# #Random list of rows to pick reduced values from
# rows=random.sample(range(0,ndet3),ndr3)
# for val in range(0,ndr3):
#     #Diag is the row number in Bigham values will be chosen from
#     diag=rows[val]
#     #Diagonal value set
#     Bigham3[val,val]=Bigham5[diag,diag]
#     Kover3[val,val]=Kover5[diag,diag]
#     start=val+1
#     #Routine to fill rest of row from place ajacent to diagonal value   
#     for i in range(start,ndr3):
#         pos=rows[i]
#         Bigham3[i,val]=Bigham5[pos,diag]
#         Kover3[i,val]=Kover5[pos,diag]
#         Bigham3[val,i]=Bigham5[pos,diag]
#         Kover3[val,i]=Kover5[pos,diag]

# #Random list of rows to pick reduced values from
# rows=random.sample(range(0,ndet3),ndr4)
# for val in range(0,ndr4):
#     #Diag is the row number in Bigham values will be chosen from
#     diag=rows[val]
#     #Diagonal value set
#     Bigham4[val,val]=Bigham5[diag,diag]
#     Kover4[val,val]=Kover5[diag,diag]
#     start=val+1
#     #Routine to fill rest of row from place ajacent to diagonal value   
#     for i in range(start,ndr4):
#         pos=rows[i]
#         Bigham4[i,val]=Bigham5[pos,diag]
#         Kover4[i,val]=Kover5[pos,diag]
#         Bigham4[val,i]=Bigham5[pos,diag]
#         Kover4[val,i]=Kover5[pos,diag]

Kinv1 = numpy.linalg.inv(Kover1)
dvec1 = numpy.dot(Kinv1,dtc1)
Kinv2 = numpy.linalg.inv(Kover2)
dvec2 = numpy.dot(Kinv2,dtc2)
# Kinv3 = numpy.linalg.inv(Kover3)
# dvec3 = numpy.dot(Kinv3,dtc3)
# Kinv4 = numpy.linalg.inv(Kover4)
# dvec4 = numpy.dot(Kinv4,dtc4)

# dvec1=numpy.zeros((ndr1))
# dvec1[0]=1.0
# dvec2=numpy.zeros((ndr2))
# dvec2[0]=1.0


# Check energy
en1 = numpy.einsum('i,ij,j',dvec1,Bigham1,dvec1)
ovi1 = numpy.einsum('i,ij,j',dvec1,Kover1,dvec1)
print('en', en1, 'ovi', ovi1)

en2 = numpy.einsum('i,ij,j',dvec2,Bigham2,dvec2)
ovi2 = numpy.einsum('i,ij,j',dvec2,Kover2,dvec2)
print('en', en2, 'ovi', ovi2)

# en3 = numpy.einsum('i,ij,j',dvec3,Bigham3,dvec3)
# ovi3 = numpy.einsum('i,ij,j',dvec3,Kover3,dvec3)
# print('en', en3, 'ovi', ovi3)

# en4 = numpy.einsum('i,ij,j',dvec4,Bigham4,dvec4)
# ovi4 = numpy.einsum('i,ij,j',dvec4,Kover4,dvec4)
# print('en', en3, 'ovi', ovi3)

# Find exact eigenstate
# Firstly diagonalise Kover
Dei1, Deivec1 = numpy.linalg.eigh(Kover1)
Dei2, Deivec2 = numpy.linalg.eigh(Kover2)
# Dei3, Deivec3 = numpy.linalg.eigh(Kover5)
# Dei4, Deivec4 = numpy.linalg.eigh(Kover5)
# Now form A matrix
Amat1 = numpy.zeros((ndet1,ndet1))
Amat2 = numpy.zeros((ndet2,ndet2))
# Amat3 = numpy.zeros((ndet3,ndet3))
# Amat4 = numpy.zeros((ndet3,ndet3))
for idet in range(ndet1):
    Amat1[:,idet] = Deivec1[:,idet]/math.sqrt(Dei1[idet])
for idet in range(ndet2):
    Amat2[:,idet] = Deivec2[:,idet]/math.sqrt(Dei2[idet])
# for idet in range(ndet3):
#     Amat3[:,idet] = Deivec3[:,idet]/math.sqrt(Dei3[idet])
# for idet in range(ndet3):
#     Amat4[:,idet] = Deivec4[:,idet]/math.sqrt(Dei4[idet])
Ainv1 = numpy.linalg.inv(Amat1)
Tmat1 = numpy.matmul(Amat1.T,numpy.matmul(Kover1,Amat1))
Ainv2 = numpy.linalg.inv(Amat2)
Tmat2 = numpy.matmul(Amat2.T,numpy.matmul(Kover2,Amat2))
# Ainv3 = numpy.linalg.inv(Amat3)
# Tmat3 = numpy.matmul(Amat3.T,numpy.matmul(Kover5,Amat3))
# Ainv4 = numpy.linalg.inv(Amat4)
# Tmat4 = numpy.matmul(Amat4.T,numpy.matmul(Kover5,Amat4))
# Now form A.T H A
HT1 = numpy.matmul(Amat1.T,numpy.matmul(Bigham1,Amat1))
Eival1, Eivec1 = numpy.linalg.eigh(HT1)
vec1 = Eivec1[:,2]
vect1 = numpy.dot(Amat1,vec1)
HT2 = numpy.matmul(Amat2.T,numpy.matmul(Bigham2,Amat2))
Eival2, Eivec2 = numpy.linalg.eigh(HT2)
vec2 = Eivec2[:,2]
vect2 = numpy.dot(Amat2,vec2)
# HT3 = numpy.matmul(Amat3.T,numpy.matmul(Bigham5,Amat3))
# Eival3, Eivec3 = numpy.linalg.eigh(HT3)
# vec3 = Eivec3[:,2]
# vect3 = numpy.dot(Amat3,vec3)
# HT4 = numpy.matmul(Amat3.T,numpy.matmul(Bigham5,Amat3))
# Eival4, Eivec4 = numpy.linalg.eigh(HT4)
# vec4 = Eivec4[:,2]
# vect4 = numpy.dot(Amat4,vec4)

beta = 200.0
nb = 1000
KinvH1 = numpy.matmul(Kinv1,Bigham1)
db = beta/nb
dvec01 = dvec1
den1 = numpy.einsum('i,ij,j',dvec01,Bigham1,dvec01)
ov01 = numpy.einsum('i,ij,j',vect1,Kover1,dvec01)
eb1 = numpy.zeros((nb))
ov1 = numpy.zeros((nb))
KinvH2 = numpy.matmul(Kinv2,Bigham2)
dvec02 = dvec2
den2 = numpy.einsum('i,ij,j',dvec02,Bigham2,dvec02)
ov02 = numpy.einsum('i,ij,j',vect2,Kover2,dvec02)
eb2 = numpy.zeros((nb))
ov2 = numpy.zeros((nb))
# KinvH3 = numpy.matmul(Kinv3,Bigham3)
# dvec03 = dvec3
# den3 = numpy.einsum('i,ij,j',dvec03,Bigham3,dvec03)
# ov3 = numpy.einsum('i,ij,j',vect3,Kover5,dvec03)
# eb3 = numpy.zeros((nb))
# ov3 = numpy.zeros((nb))
# KinvH4 = numpy.matmul(Kinv4,Bigham4)
# dvec04 = dvec4
# den4 = numpy.einsum('i,ij,j',dvec04,Bigham4,dvec04)
# ov4 = numpy.einsum('i,ij,j',vect4,Kover5,dvec04)
# eb4 = numpy.zeros((nb))
# ov4 = numpy.zeros((nb))

for ib in range(nb):
    ddot1 = -numpy.dot(KinvH1,dvec1)
    dvec1 = dvec1 + db*ddot1
    norm1 = numpy.einsum('i,ij,j',dvec1,Kover1,dvec1)
    dvec1 /= norm1**0.5
    den1 = numpy.einsum('i,ij,j',dvec1,Bigham1,dvec1)
    eb1[ib] = den1
    ov1[ib] = numpy.einsum('i,ij,j',vect1,Kover1,dvec1)
    ddot2 = -numpy.dot(KinvH2,dvec2)
    dvec2 = dvec2 + db*ddot2
    norm2 = numpy.einsum('i,ij,j',dvec2,Kover2,dvec2)
    dvec2 /= norm2**0.5
    den2 = numpy.einsum('i,ij,j',dvec2,Bigham2,dvec2)
    eb2[ib] = den2
    ov2[ib] = numpy.einsum('i,ij,j',vect2,Kover2,dvec2)
    # ddot3 = -numpy.dot(KinvH3,dvec3)
    # dvec3 = dvec3 + db*ddot3
    # norm3 = numpy.einsum('i,ij,j',dvec3,Kover3,dvec3)
    # dvec3 /= norm3**0.5
    # den3 = numpy.einsum('i,ij,j',dvec3,Bigham3,dvec3)
    # eb3[ib] = den3
    # # ov3[ib] = numpy.einsum('i,ij,j',vect3,Kover3,dvec3)
    # ddot4 = -numpy.dot(KinvH4,dvec4)
    # dvec4 = dvec4 + db*ddot4
    # norm4 = numpy.einsum('i,ij,j',dvec4,Kover4,dvec4)
    # dvec4 /= norm4**0.5
    # den4 = numpy.einsum('i,ij,j',dvec4,Bigham4,dvec4)
    # eb4[ib] = den4
    # ov4[ib] = numpy.einsum('i,ij,j',vect4,Kover4,dvec4)

mpl.rcParams['font.family']='Avenir'
plt.rcParams['font.size']=18
plt.rcParams['axes.linewidth']=2
colors =cm.get_cmap('Set1',3)
fig=plt.figure(figsize=(3.37,5.055))
ax=fig.add_axes([0,0,2,1])
ax.plot(numpy.linspace(db,beta,num=nb),eb1, linestyle=':',  linewidth=2, color=colors(0))
ax.plot(numpy.linspace(db,beta,num=nb),eb2, alpha=0.4, linewidth=6, color=colors(1))
# ax.plot(numpy.linspace(db,beta,num=nb),eb3, linewidth=2, label='50 random Zombie states',color=colors(2))
# ax.plot(numpy.linspace(db,beta,num=nb),eb4, linewidth=2, label='30 random Zombie states',color=colors(3))
ax.plot(numpy.linspace(db,beta,num=nb),numpy.ones((nb))*-14.876024486659754,linewidth=2, color=colors(4))
# ax.plot(numpy.linspace(db,beta,num=nb),numpy.ones((nb))*-14.869949868878516,linewidth=2, color=colors(2))
ax.set_xlim(0,200)
# ax.set_ylim(-15,-12.5)
# ax.legend(loc='center right')
ax.set_ylabel('Energy',labelpad=10)
ax.set_xlabel(r'$\mathregular{\beta}$',labelpad=10)
plt.savefig('imaginaryslaterrand2.png',dpi=300, transparent=False,bbox_inches='tight')
#plt.show()    
exit()