import numpy as np
from pyscf import gto, scf, ao2mo, fci

mol = gto.M()
mol.nelectron = 12
mol.incore_anyway = True
n = 12
mf = scf.RHF(mol)
h1 = np.zeros((n,n))
for i in range(n-1):
    h1[i,i+1] = h1[i+1,i] = -1.0

h1[n-1,0] = h1[0,n-1] = 1.0
eri = np.zeros((n,n,n,n))
np.random.seed(123)
for i in range(n):
    eri[i,i,i,i] = 2.0 + np.random.uniform(0,1)
mf.get_hcore = lambda *args: h1
mf.get_ovlp = lambda *args: np.eye(n)

mf._eri = ao2mo.restore(8, eri, n)
mf.scf()

fcisolver = fci.FCI(mf)
E, fcivec = fcisolver.kernel()
print(E)
