'''
    QC-DMET: a python implementation of density matrix embedding theory for ab initio quantum chemistry
    Copyright (C) 2015 Sebastian Wouters
    
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
'''

import sys
sys.path.append('../src')
import localintegrals, dmet, qcdmet_paths
import numpy as np
from pyscf import gto, scf

# build a molecule of carbon atoms
mol = gto.Mole()
mol.build(
   atom = [['C', (0.0,                          0.0,  0.0)],
           ['C', (0.0,              1.46060096158367, 0.0)],
           ['C', (1.26491753752344, 2.19090144237551, 0.0)],
           ['C', (1.26491753752344, 3.65150240395918, 0.0)]],
   verbose = 4
)
# parameters for dmet
natm = mol.natm
nelec = mol.tot_electrons()
norb = mol.nao_nr()
factor = 1.0
mtype = np.float64
cmtype = 'RHF'

mf = scf.RHF(mol)
mf.scf()

myInts = localintegrals.localintegrals(mf, range(mol.nao_nr()), 'meta_lowdin')
iatm = 1    # number of atoms in each impurity
iorb = myInts.Norbs*iatm/natm

impurityClusters = []
for cluster in range(natm/iatm):
   impurities = np.zeros( [myInts.Norbs], dtype=int )
   for orb in range(iorb):
      impurities[cluster*iorb + orb] = 1
   impurityClusters.append(impurities)

method = 'FCI'
SCmethod = 'BFGS'
isTranslationInvariant = False
myInts.TI_OK = True

theDMET = dmet.dmet(myInts, impurityClusters, isTranslationInvariant, method, SCmethod)
theDMET.doselfconsistent()

