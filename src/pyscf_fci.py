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

import numpy as np
import ctypes
import os  # for dev/null
import sys # for sys.stdout
import qcdmet_paths
from pyscf import ao2mo, gto, scf
from pyscf import fci

def solve( CONST, OEI, FOCK, TEI, Norb, Nel, Nimp, DMguessRHF, chempot_imp):
    
    # Augment the FOCK operator with the chemical potential
    FOCKcopy = FOCK.copy()
    
    if (chempot_imp != 0.0):
        for orb in range(Nimp):
            FOCKcopy[ orb, orb ] -= chempot_imp
    
    # Get the RHF solution
    mol = gto.Mole()
    mol.build( verbose=0 )
    mol.atom.append(('C', (0, 0, 0)))
    mol.nelectron = Nel
    mol.incore_anyway = True
    mf = scf.RHF( mol )
    mf.get_hcore = lambda *args: FOCKcopy
    mf.get_ovlp = lambda *args: np.eye( Norb )
    mf._eri = ao2mo.restore(8, TEI, Norb)
    mf.scf( DMguessRHF )
    DMloc = np.dot(np.dot( mf.mo_coeff, np.diag( mf.mo_occ )), mf.mo_coeff.T )
    
    if ( mf.converged == False ):
        mf = mf.newton()
        DMloc = np.dot(np.dot( mf.mo_coeff, np.diag( mf.mo_occ )), mf.mo_coeff.T )
        
    # Check the RHF solution
    assert( Nel % 2 == 0 )
    numPairs = Nel / 2
    FOCKloc = FOCKcopy + np.einsum('ijkl,ij->kl', TEI, DMloc) - 0.5 * np.einsum('ijkl,ik->jl', TEI, DMloc)
    eigvals, eigvecs = np.linalg.eigh( FOCKloc )
    idx = eigvals.argsort()
    eigvals = eigvals[ idx ]
    eigvecs = eigvecs[ :, idx ]
    # print "psi4cc::solve : RHF homo-lumo gap =", eigvals[numPairs] - eigvals[numPairs-1]
    DMloc2  = 2 * np.dot( eigvecs[ :, :numPairs ], eigvecs[ :, :numPairs ].T )
    # print "Two-norm difference of 1-RDM(RHF) and 1-RDM(FOCK(RHF)) =", np.linalg.norm(DMloc - DMloc2)
    
    # Get the fci solution from pyscf
    fcisolver = fci.FCI( mf )
    fcisolver.verbose = 1
    E, fcivec = fcisolver.kernel()
    
    # Compute the impurity energy
    pyscfRDM1 = fcisolver.make_rdm1(fcivec, Norb, Nel) # MO space
    pyscfRDM2 = fcisolver.make_rdm2(fcivec, Norb, Nel) # MO space
    
    pyscfRDM1 = 0.5 * ( pyscfRDM1 + pyscfRDM1.T ) # Symmetrize
            
    # Change the pyscfRDM1/2 from MO space to localized space
    pyscfRDM1 = np.dot(mf.mo_coeff, np.dot(pyscfRDM1, mf.mo_coeff.T ))
    pyscfRDM2 = np.einsum('ai,ijkl->ajkl', mf.mo_coeff, pyscfRDM2)
    pyscfRDM2 = np.einsum('bj,ajkl->abkl', mf.mo_coeff, pyscfRDM2)
    pyscfRDM2 = np.einsum('ck,abkl->abcl', mf.mo_coeff, pyscfRDM2)
    pyscfRDM2 = np.einsum('dl,abcl->abcd', mf.mo_coeff, pyscfRDM2)
        
    # To calculate the impurity energy, rescale the JK matrix with a factor 0.5 to avoid double counting: 0.5 * ( OEI + FOCK ) = OEI + 0.5 * JK
    ImpurityEnergy = CONST \
                     + 0.25  * np.einsum('ij,ij->',     pyscfRDM1[:Nimp,:],     FOCK[:Nimp,:] + OEI[:Nimp,:]) \
                     + 0.25  * np.einsum('ij,ij->',     pyscfRDM1[:,:Nimp],     FOCK[:,:Nimp] + OEI[:,:Nimp]) \
                     + 0.125 * np.einsum('ijkl,ijkl->', pyscfRDM2[:Nimp,:,:,:], TEI[:Nimp,:,:,:]) \
                     + 0.125 * np.einsum('ijkl,ijkl->', pyscfRDM2[:,:Nimp,:,:], TEI[:,:Nimp,:,:]) \
                     + 0.125 * np.einsum('ijkl,ijkl->', pyscfRDM2[:,:,:Nimp,:], TEI[:,:,:Nimp,:]) \
                     + 0.125 * np.einsum('ijkl,ijkl->', pyscfRDM2[:,:,:,:Nimp], TEI[:,:,:,:Nimp])
    
    return ( ImpurityEnergy, pyscfRDM1 )

