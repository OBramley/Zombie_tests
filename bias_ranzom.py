# Generating random zombie states and the Hamiltonian between them
import math, numpy, scipy
import zombie, timeit
from pyscf import gto, scf, ao2mo
from pyscf import tools
from pyscf import symm
norb = 10
# Hnr, H1ei, H2ei = zombie.readin('Integrals_Li2_6',norb)
# Li2 = zombie.system(norb,Hnr,H1ei,H2ei)

mol = gto.M(
    unit = 'Bohr',
    atom = 'Li 0 0 0; Li 0 0 2', #[['H', 0, 0, i] for i in range(6)],
    basis = '6-31g**',
    verbose = 4,
    symmetry = True,
    # spin=1,
    # charge=-1
    #symmetry_subgroup = 0, #0 is code for A1 point group
)
print(mol.nelec)
print(mol._atom)
myhf = scf.RHF(mol)  #myhcf
print(myhf.kernel())
Hnr, H1ei, H2ei = zombie.pyscfint(mol,myhf,norb)
Li2 = zombie.system(norb,Hnr,H1ei,H2ei)



ndet = 2**norb

numpy.random.seed(1)
zstore = []

ndet=64
skipper=0
av=0
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
        elif((i>5)and(i<7)):
            mu=0
            sig=0.351
            num[i]=numpy.random.normal(mu,sig)
        elif((i>6)and(i<8)):
            mu=0
            sig=0.351
            num[i]=numpy.random.normal(mu,sig)
        else:
            mu=0
            sig=0.12
            num[i]=numpy.random.normal(mu,sig)
            # num[i]=0
    # if(idet==0):
    #     zstore.append(zombie.zom(norb,typ='aufbau',nel=5))    
    # else:
    iteration=zombie.zom(norb,typ='theta',thetas=num)
    zstore.append(iteration)
    nel=zombie.numf(iteration.zs,iteration.zs)
    print(nel)
    av=av+nel
        # if(randomizer<0.9):
        #     #alt=[]
        #     # alt=numpy.array([num[1],num[0],num[3],num[2],num[5],num[4],num[7],num[6],num[9],num[8]])
        #     alt=numpy.array([num[0],num[1],num[2],num[3],num[5],num[4],num[7],num[6],num[9],num[8]])
        #     zstore.append(zombie.zom(norb,typ='theta',thetas=alt))
        #     skipper=1
print("average nel")
print(av/ndet)
Kover = numpy.zeros((ndet,ndet))
Ham = numpy.zeros((ndet,ndet))
# Overlap matrix
ranham = open('biassed_64BF_14_pyscf_noHF.dat','w')
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

exit()
