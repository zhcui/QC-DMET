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
import localintegrals_hubbard, dmet, qcdmet_paths
import numpy as np

imp_size   = 4

fillings = []
energies = []

file = open("energy.dat", "w+")

Norbs = 120
Nelectrons = Norbs - 4
HubbardU = np.zeros(Norbs)

np.random.seed(123)
for orb in range(0,Norbs):
   HubbardU[ orb ] = 2.0 + np.random.uniform(0,1)

hopping  = np.zeros( [Norbs, Norbs], dtype=float )
for orb in range(Norbs-1):
   hopping[ orb, orb+1 ] = -1.0
   hopping[ orb+1, orb ] = -1.0
hopping[ 0, Norbs-1 ] = 1.0 # anti-PBC
hopping[ Norbs-1, 0 ] = 1.0 # anti-PBC

myInts = localintegrals_hubbard.localintegrals_hubbard( hopping, HubbardU, Nelectrons )
   
impurityClusters = []
for cluster in range( Norbs / imp_size ):
   impurities = np.zeros( [ myInts.Norbs ], dtype=int )
   for orb in range( cluster*imp_size, (cluster+1)*imp_size ):
      impurities[ orb ] = 1
   impurityClusters.append( impurities )

totalcount = np.zeros( [ myInts.Norbs ], dtype=int )
for item in impurityClusters:
   totalcount += item
assert ( np.linalg.norm( totalcount - np.ones( [ myInts.Norbs ], dtype=float ) ) < 1e-12 )

isTranslationInvariant = False
method = 'CC'
SCmethod = 'BFGS' # 'LSTSQ'
theDMET = dmet.dmet( myInts, impurityClusters, isTranslationInvariant, method, SCmethod )
theEnergy = theDMET.doselfconsistent()

fillings.append( (1.0 * Nelectrons) / Norbs )
energies.append( theEnergy / Norbs )

file.write(str(theEnergy/Nelectrons) + '\n')
file.close()

np.set_printoptions(precision=8, linewidth=160)
print "For U =", HubbardU,"and Norbs =", Norbs
print "Fillings ="
print np.array( fillings )
print "E / site ="
print np.array( energies )


