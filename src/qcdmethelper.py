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

import localintegrals
import rhf
import numpy as np
import ctypes
lib_qcdmet = ctypes.CDLL('../lib/libqcdmet.so')

class qcdmethelper:
    
    def __init__( self, theLocalIntegrals, list_H1, altcf, minFunc ):
        
        self.locints = theLocalIntegrals
        assert( self.locints.Nelec % 2 == 0 )
        self.numPairs = self.locints.Nelec / 2
        self.altcf = altcf
        self.minFunc = None
        self.doT = True
        if self.altcf:
            assert (minFunc == 'OEI' or minFunc == 'FOCK_INIT')
            self.minFunc = minFunc
        
        # Variables for c gradient calculation
        self.list_H1 = list_H1
        H1start, H1row, H1col = self.convertH1sparse()
        self.H1start = H1start
        self.H1row = H1row
        self.H1col = H1col
        self.Nterms = len( self.H1start ) - 1
        
        # variables for finite temperature density matrix
        self.mu = 0.0
        self.Tempr = 60
        self.doT = True
        
    def convertH1sparse( self ):
        
        H1start = []
        H1row   = []
        H1col   = []
        H1start.append( 0 )
        totalsize = 0
        for count in range( len( self.list_H1 ) ):
            rowco, colco = np.where( self.list_H1[count] == 1 )
            totalsize += len( rowco )
            H1start.append( totalsize )
            for count2 in range( len( rowco ) ):
                H1row.append( rowco[ count2 ] )
                H1col.append( colco[ count2 ] )
        H1start = np.array( H1start, dtype=ctypes.c_int )
        H1row   = np.array( H1row,   dtype=ctypes.c_int )
        H1col   = np.array( H1col,   dtype=ctypes.c_int )
        return ( H1start, H1row, H1col )
    
    def construct1RDM_loc( self, doSCF, umat_loc ):
        
        # Everything in this functions works in the original local AO / lattice basis!
        if self.altcf and self.minFunc == 'OEI' :
            OEI = self.locints.loc_oei() + umat_loc
        elif self.altcf and self.minFunc == 'FOCK_INIT' :
            OEI = self.locints.loc_rhf_fock() + umat_loc
        else:
            OEI   = self.locints.loc_rhf_fock() + umat_loc
        DMloc = self.construct1RDM_base( OEI, self.numPairs )
        
        if ( doSCF == True ):
            if ( self.locints.ERIinMEM == True ):
                DMloc = rhf.solve_ERI( self.locints.loc_oei() + umat_loc, self.locints.loc_tei(), DMloc, self.numPairs )
            else:
                DMloc = rhf.solve_JK( self.locints.loc_oei() + umat_loc, self.locints.mol, self.locints.ao2loc, DMloc, self.numPairs )
        return DMloc
    
    def construct1RDM_loc_T( self, doSCF, umat_loc ):
        
        # Everything in this functions works in the original local AO / lattice basis!
        if self.altcf and self.minFunc == 'OEI' :
            OEI = self.locints.loc_oei() + umat_loc
        elif self.altcf and self.minFunc == 'FOCK_INIT' :
            OEI = self.locints.loc_rhf_fock() + umat_loc
        else:
            OEI   = self.locints.loc_rhf_fock() + umat_loc
        #DMloc = self.construct1RDM_base( OEI, self.numPairs )
        DMloc = self.construct1RDM_base_T( OEI, self.numPairs )
        if ( doSCF == True ):
            if ( self.locints.ERIinMEM == True ):
                DMloc = rhf.solve_ERI( self.locints.loc_oei() + umat_loc, self.locints.loc_tei(), DMloc, self.numPairs )
            else:
                DMloc = rhf.solve_JK( self.locints.loc_oei() + umat_loc, self.locints.mol, self.locints.ao2loc, DMloc, self.numPairs )
        return DMloc
    
    def construct1RDM_response( self, doSCF, umat_loc, NOrotation ):
        
        # This part works in the original local AO / lattice basis!
        OEI = self.locints.loc_rhf_fock() + umat_loc
        
        if ( doSCF == True ):
            DMloc = self.construct1RDM_base( OEI, self.numPairs )
            if ( self.locints.ERIinMEM == True ):
                DMloc = rhf.solve_ERI( self.locints.loc_oei() + umat_loc, self.locints.loc_tei(), DMloc, self.numPairs )
            else:
                DMloc = rhf.solve_JK( self.locints.loc_oei() + umat_loc, self.locints.mol, self.locints.ao2loc, DMloc, self.numPairs )
            OEI = self.locints.loc_rhf_fock_bis( DMloc ) + umat_loc
        
        # This part works in the rotated NO basis if NOrotation is specified
        rdm_deriv_rot = np.ones( [ self.locints.Norbs * self.locints.Norbs * self.Nterms ], dtype=ctypes.c_double )
        if ( NOrotation != None ):
            OEI = np.dot( np.dot( NOrotation.T, OEI ), NOrotation )
        OEI = np.array( OEI.reshape( (self.locints.Norbs * self.locints.Norbs) ), dtype=ctypes.c_double )
        if(self.doT):
            lib_qcdmet.rhf_response_T( ctypes.c_int( self.locints.Norbs ),
                                       ctypes.c_int( self.Nterms ),
                                       self.H1start.ctypes.data_as( ctypes.c_void_p ),
                                       self.H1row.ctypes.data_as( ctypes.c_void_p ),
                                       self.H1col.ctypes.data_as( ctypes.c_void_p ),
                                       OEI.ctypes.data_as( ctypes.c_void_p ),
                                       ctypes.c_double( self.Tempr ),
                                       ctypes.c_double( self.mu ),
                                       rdm_deriv_rot.ctypes.data_as( ctypes.c_void_p ) )
        else:
            lib_qcdmet.rhf_response( ctypes.c_int( self.locints.Norbs ),
                                     ctypes.c_int( self.Nterms ),
                                     ctypes.c_int( self.numPairs ),
                                     self.H1start.ctypes.data_as( ctypes.c_void_p ),
                                     self.H1row.ctypes.data_as( ctypes.c_void_p ),
                                     self.H1col.ctypes.data_as( ctypes.c_void_p ),
                                     OEI.ctypes.data_as( ctypes.c_void_p ),
                                     rdm_deriv_rot.ctypes.data_as( ctypes.c_void_p ) )
        
        rdm_deriv_rot = rdm_deriv_rot.reshape( (self.Nterms, self.locints.Norbs, self.locints.Norbs), order='C' )
        return rdm_deriv_rot
        
    def construct1RDM_base( self, OEI, myNumPairs ):
        
        eigenvals, eigenvecs = np.linalg.eigh( OEI ) # Does not guarantee sorted eigenvectors!
        idx = eigenvals.argsort()
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:,idx]
        
        OneDM = 2 * np.dot( eigenvecs[:,:myNumPairs] , eigenvecs[:,:myNumPairs].T )
        #print "SP gap =", eigenvals[myNumPairs] - eigenvals[myNumPairs-1]
        return OneDM

    
    def construct1RDM_base_T( self, OEI, myNumPairs):
        
        eigenvals, eigenvecs = np.linalg.eigh( OEI ) # Does not guarantee sorted eigenvectors!
        idx = eigenvals.argsort()
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:,idx]
        
        # find the proper chemical potential
        mu_eps = 1e-8
        iteration = 0
        mu = eigenvals[myNumPairs]
        invT = self.Tempr
        mu_diff = 100
        mu1 = eigenvals[0]
        mu2 = eigenvals[len(eigenvals)-1]
        
        N1 = 0.0; N2 = 0.0
        for i in range(0,len(eigenvals)):
            trm1 = np.exp(invT*(eigenvals[i]-mu1))
            trm2 = np.exp(invT*(eigenvals[i]-mu2))
            N1 += 1.0/(1.0+trm1)
            N2 += 1.0/(1.0+trm2)
            
        while(mu_diff > mu_eps and iteration < 100):
            iteration += 1
            # mu_old = mu
            
            # N = 0
            # gradN = 0
            # for i in range(0, len(eigenvals)):
            #     trm0 = np.exp(invT*(eigenvals[i]-mu))
            #     N += 1/(1+trm0)
            #     gradN += (1+trm0)**(-2)*trm0*invT
            
            # mu = mu - (N-myNumPairs)/gradN
            # mu_diff = np.abs(mu_old-mu)
            
            N0 = 0.0
            mu0 = (mu1+mu2)/2.0
            for i in range(0, len(eigenvals)):
                trm0 = np.exp(invT*(eigenvals[i]-mu0))
                N0 += 1.0/(1.0+trm0)
            if(np.abs(N1-myNumPairs) > np.abs(N2-myNumPairs)):
                N1 = N0; mu1 = mu0
            else:
                N2 = N0; mu2 = mu0
            mu_diff = np.abs(mu1-mu2)
            
            
        # print("Chemical potential", mu, eigenvals)
        if(iteration > 100):
            print("Did not find the proper chemical potential")
            exit()
            
        vecs = np.zeros((len(eigenvals), len(eigenvals)))
        for i in range(0, len(eigenvals)):
            vecs[i,:] = eigenvecs[:,i]/(1+np.exp(invT*(eigenvals[i]-mu)))
            
        
        OneDM = 2 * np.dot( eigenvecs[:,:] , vecs[:,:] )
        #print "SP gap =", eigenvals[myNumPairs] - eigenvals[myNumPairs-1]
        
        self.mu = mu
        
        return OneDM
    
    
    def constructbath( self, OneDM, impurityOrbs, numBathOrbs, threshold=1e-13 ):
    
        embeddingOrbs = 1 - impurityOrbs
        embeddingOrbs = np.matrix( embeddingOrbs )
        if (embeddingOrbs.shape[0] > 1):
            embeddingOrbs = embeddingOrbs.T # Now certainly row-like matrix (shape = 1 x len(vector))
        isEmbedding = np.dot( embeddingOrbs.T , embeddingOrbs ) == 1
        numEmbedOrbs = np.sum( embeddingOrbs )
        embedding1RDM = np.reshape( OneDM[ isEmbedding ], ( numEmbedOrbs , numEmbedOrbs ) )

        numImpOrbs   = np.sum( impurityOrbs )
        numTotalOrbs = len( impurityOrbs )
        
        eigenvals, eigenvecs = np.linalg.eigh( embedding1RDM )
        idx = np.maximum( -eigenvals, eigenvals - 2.0 ).argsort() # Occupation numbers closest to 1 come first
        tokeep = np.sum( -np.maximum( -eigenvals, eigenvals - 2.0 )[idx] > threshold )
        if ( tokeep < numBathOrbs ):
            print "DMET::constructbath : Throwing out", numBathOrbs - tokeep, "orbitals which are within", threshold, "of 0 or 2."
        numBathOrbs = min(np.sum( tokeep ), numBathOrbs)
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:,idx]
        pureEnvironEigVals = -eigenvals[numBathOrbs:]
        pureEnvironEigVecs = eigenvecs[:,numBathOrbs:]
        idx = pureEnvironEigVals.argsort()
        eigenvecs[:,numBathOrbs:] = pureEnvironEigVecs[:,idx]
        pureEnvironEigVals = -pureEnvironEigVals[idx]
        coreOccupations = np.hstack(( np.zeros([ numImpOrbs + numBathOrbs ]), pureEnvironEigVals ))
    
        for counter in range(0, numImpOrbs):
            eigenvecs = np.insert(eigenvecs, counter, 0.0, axis=1) #Stack columns with zeros in the beginning
        counter = 0
        for counter2 in range(0, numTotalOrbs):
            if ( impurityOrbs[counter2] ):
                eigenvecs = np.insert(eigenvecs, counter2, 0.0, axis=0) #Stack rows with zeros on locations of the impurity orbitals
                eigenvecs[counter2, counter] = 1.0
                counter += 1
        assert( counter == numImpOrbs )
    
        # Orthonormality is guaranteed due to (1) stacking with zeros and (2) orthonormality eigenvecs for symmetric matrix
        assert( np.linalg.norm( np.dot(eigenvecs.T, eigenvecs) - np.identity(numTotalOrbs) ) < 1e-12 )

        # eigenvecs[ : , 0:numImpOrbs ]                      = impurity orbitals
        # eigenvecs[ : , numImpOrbs:numImpOrbs+numBathOrbs ] = bath orbitals
        # eigenvecs[ : , numImpOrbs+numBathOrbs: ]           = pure environment orbitals in decreasing order of occupation number
        return ( numBathOrbs, eigenvecs, coreOccupations )
        
