# Generating random zombie states and the Hamiltonian between them
import math, numpy, scipy
import zombie, timeit
from pyscf import gto, scf, ao2mo
from pyscf import tools
from pyscf import symm
from matplotlib import pyplot as plt
import matplotlib as mpl
from pylab import cm

norb = 10

Hnr, H1ei, H2ei = zombie.readin('Integrals_Li2_6',norb)
Li2 = zombie.system(norb,Hnr,H1ei,H2ei)

ndet = 2**norb

numpy.random.seed(1)
zstore = []

ndet=65
skipper=0
for idet in range(ndet):
    num=numpy.zeros(norb)
    if(skipper==1):
        skipper=0
        continue
    randomizer=0#numpy.random.rand()
    for i in range(norb):
        if(i<4):
            # mu=0.25
            # sig=0.0001
            # num[i]=numpy.random.normal(mu,sig)
            num[i]=0.25
        elif((i>3)and(i<6)):
            mu=0.25
            sig=0.175
            num[i]=numpy.random.normal(mu,sig)
        elif((i>5)and(i<8)):
            mu=0
            sig=0.351
            num[i]=numpy.random.normal(mu,sig)
        else:
            mu=0
            sig=0.12
            num[i]=numpy.random.normal(mu,sig)
            # num[i]=0
    if(idet==0):
        iteration=zombie.zom(norb,typ='aufbau',nel=6)
        zstore.append(iteration)    
    else:
        iteration
        zstore.append(zombie.zom(norb,typ='theta',thetas=num))
    print(zombie.numf(iteration.zs,iteration.zs))
        # if(randomizer<0.9):
        #     #alt=[]
        #     # alt=numpy.array([num[1],num[0],num[3],num[2],num[5],num[4],num[7],num[6],num[9],num[8]])
        #     alt=numpy.array([num[0],num[1],num[2],num[3],num[5],num[4],num[7],num[6],num[9],num[8]])
        #     zstore.append(zombie.zom(norb,typ='theta',thetas=alt))
        #     skipper=1

Kover = numpy.zeros((ndet,ndet))
Ham = numpy.zeros((ndet,ndet))
# Overlap matrix
ranham = open('biassed_65BF_14_molpro.dat','w')
for idet in range(ndet):
    for jdet in range(idet,ndet):
        Kover[idet,jdet] = zombie.overlap_f(zstore[idet].zs, zstore[jdet].zs)
        Kover[jdet,idet] = Kover[idet,jdet]
        Ham[idet,jdet] = Li2.HTot(zstore[idet].zs, zstore[jdet].zs)
        Ham[jdet,idet] = Ham[idet,jdet]
        ranham.write('{:5d}, {:5d}, {:<20.15}, {:<20.15}\n'.\
                     format(idet,jdet,Kover[idet,jdet],Ham[idet,jdet]))
    # print(idet,'completed!',flush=True)                    

ranham.close()

# Firstly diagonalise Kover
Dei, Deivec = numpy.linalg.eigh(Kover)
# Now form A matrix
Amat = numpy.zeros((ndet,ndet))
for idet in range(ndet):
    Amat[:,idet] = Deivec[:,idet]/math.sqrt(Dei[idet])
Ainv = numpy.linalg.inv(Amat)
Tmat = numpy.matmul(Amat.T,numpy.matmul(Kover,Amat))
print(numpy.allclose(Tmat,numpy.eye(ndet)))
# Now form A.T H A
HT = numpy.matmul(Amat.T,numpy.matmul(Ham,Amat))

nei = 10
Eival, Eivec = numpy.linalg.eigh(HT)

tol = 0.01
for ieig in range(nei):
    string = 'Eigenstate {0:5d} Energy {1:<+15.12f}'.\
        format(ieig,Eival[ieig])
    vec = Eivec[:,ieig]
    vect = numpy.dot(Amat,vec)
    top = numpy.einsum('i,ij,j',vect,Ham,vect)
    #top = numpy.dot(vect,numpy.dot(Bigham,vect))
    bottom = numpy.einsum('i,ij,j',vect,Kover,vect)
    #bottom = numpy.dot(vect,numpy.dot(Kover,vect))
    print(string)
    #print('   Transform to nonorthogonal basis')
    #print('   Energy {0:<+20.16f} Overlap {1:<+20.16f}'.format(top,bottom))

ndr=ndet
BHr=Ham
Kr=Kover
Kri = numpy.linalg.inv(Kr)

dvec=numpy.zeros((ndr))
dvec[0]=1.0

ovr = numpy.einsum('i,ij,j',dvec,Kr,dvec)
print('ovr',ovr)
# Check energy
en = numpy.einsum('i,ij,j',dvec,BHr,dvec)
ovi = numpy.einsum('i,ij,j',dvec,Kr,dvec)
print('en', en, 'ovi', ovi)

vec = Eivec[:,0]
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
fig=plt.figure(figsize=(6,7))
ax=fig.add_axes([0,0,2,1])
ax.plot(numpy.linspace(db,beta,num=nb),eb, linewidth=2, color=colors(0))
ax.plot(numpy.linspace(db,beta,num=nb),numpy.ones((nb))*Eival[2],linewidth=2, color=colors(1))
ax.set_xlim(0,200)
#ax.set_ylim(-14.8615,-14.8575)
ax.set_ylabel('Energy',labelpad=10)
ax.set_xlabel(r'$\mathregular{\beta}$',labelpad=10)
plt.savefig('biased_65_prop.png',dpi=300, transparent=False,bbox_inches='tight')

exit()