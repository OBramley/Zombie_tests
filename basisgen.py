# Generating random zombie states and the Hamiltonian between them
from itertools import cycle
import math, numpy, scipy
import zombie, timeit
from pyscf import gto, scf, ao2mo
from pyscf import tools
from pyscf import symm

filename='Nitrogen_test.dat'
#Orbital numbering starts at 0
norb = 18
#Set this to the fin al orbital number that will be alive or fully alive. So if the first 4 electrons are all alive allalive=3 
Alivestop=13
allalive=3
mol = gto.M(
    unit = 'Bohr',
    atom = 'N 0 0 0; N 0 0 6', #[['H', 0, 0, i] for i in range(6)],
    basis = '6-31g**',
    verbose = 4,
    symmetry = True,
    #symmetry_subgroup = 0, #0 is code for A1 point group
)

print(mol._atom)
myhf = scf.RHF(mol)  #myhcf
print(myhf.kernel())
Hnr, H1ei, H2ei = zombie.pyscfint(mol,myhf,norb)
zomsy = zombie.system(norb,Hnr,H1ei,H2ei)

#numpy.random.seed(2)
zstore = []

ndet=50
for idet in range(ndet):
    num=numpy.zeros(norb)
    for i in range(norb):
        if(i<allalive+1):
            num[i]=0.25
            continue
        if(i<Alivestop+1):
            mu=0.25
        else:
            mu=0
        if(i==4):
            sig=1
        if(i==5):
            sig=1
        if(i==6):
            sig=1
        if(i==7):
            sig=1
        if(i==8):
            sig=1
        if(i==9):
            sig=1
        if(i==10):
            sig=1
        if(i==11):
            sig=1
        if(i==12):
            sig=1
        if(i==13):
            sig=1
        if(i==14):
            sig=1
        if(i==15):
            sig=1
        if(i==16):
            sig=1
        if(i==17):
            sig=1
        
        num[i]=numpy.random.normal(mu,sig)
    zstore.append(zombie.zom(norb,typ='theta',thetas=num))

Kover = numpy.zeros((ndet,ndet))
Ham = numpy.zeros((ndet,ndet))
# Overlap matrix
ranham = open(filename,'w')
for idet in range(ndet):
    for jdet in range(idet,ndet):
        Kover[idet,jdet] = zombie.overlap_f(zstore[idet].zs, zstore[jdet].zs)
        Kover[jdet,idet] = Kover[idet,jdet]
        Ham[idet,jdet] = zomsy.HTot(zstore[idet].zs, zstore[jdet].zs)
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

nei = norb
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
