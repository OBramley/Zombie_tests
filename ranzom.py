# Generating random zombie states and the Hamiltonian between them
import math, numpy, scipy
import zombie, timeit

from pyscf import gto, scf, ao2mo
from pyscf import tools
from pyscf import symm


norb = 10
mol = gto.M(
    unit = 'Bohr',
    atom = 'Li 0 0 0; Li 0 0 6', #[['H', 0, 0, i] for i in range(6)],
    basis = '6-31g**',
    verbose = 4,
    symmetry = True,
    #spin=1,
    #charge=-1
    #symmetry_subgroup = 0, #0 is code for A1 point group
)

print(mol._atom)
myhf = scf.RHF(mol)  #myhcf
print(myhf.kernel())


# Hnr, H1ei, H2ei = zombie.readin('Integrals_LiH_f_5',norb)
Hnr, H1ei, H2ei = zombie.pyscfint(mol,myhf,norb)
Li2 = zombie.system(norb,Hnr,H1ei,H2ei)

ndet = 2**norb

# numpy.random.seed(1)
# zstore = []

# for idet in range(ndet):
#     zstore.append(zombie.zom(norb,typ='ran'))

# Kover = numpy.zeros((ndet,ndet))
# Ham = numpy.zeros((ndet,ndet))
# # Overlap matrix
# fover = open('ran_hamf_anion.dat','w')
# ranham = open('ran_ham_anion.dat','w')
# for idet in range(ndet):
#     for jdet in range(idet,ndet):
#         Kover[idet,jdet] = zombie.overlap_f(zstore[idet].zs, zstore[jdet].zs)
#         Kover[jdet,idet] = Kover[idet,jdet]
#         fover.write('{:5d}, {:5d}, {:<20.15f}\n'.format(idet,jdet,Kover[idet,jdet]))
#         Ham[idet,jdet] = Li2.HTot(zstore[idet].zs, zstore[jdet].zs)
#         Ham[jdet,idet] = Ham[idet,jdet]
#         ranham.write('{:5d}, {:5d}, {:<20.15}, {:<20.15}\n'.\
#                      format(idet,jdet,Kover[idet,jdet],Ham[idet,jdet]))
#     print(idet,'completed!',flush=True)                    
# fover.close()
# ranham.close()
# 
# exit()

bini = numpy.zeros((norb),dtype=int)
Detlist = []

ff = open('Ham_2.out','w')

for i in range(ndet):
    it = i
    for j in range(norb):
        bini[j] = it%2
        it -= bini[j]
        it = it/2
    znew = zombie.new(norb,bini)
    Detlist.append(znew)

for i1 in range(ndet):
    zom1 = Detlist[i1]
    nz1 = zombie.numf(zom1,zom1)
    for i2 in range(i1, 2**norb):
        zom2 = Detlist[i2]
        nz2 = zombie.numf(zom2,zom2)
        if nz1 == nz2:
            Hamel = Li2.HTot(zom1,zom2)
        else:
            Hamel = 0.0 # Zero interaction if different number of electrons
        ff.write('{} {} {} {} {} \n'.format(i1,i2,nz1,nz2,Hamel))
    print('i1', i1, 'finished!',flush=True)
exit()