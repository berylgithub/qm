import numpy as np
import numba
from copy import deepcopy

root2,ipi=2**0.5,np.pi*1j
half_rootpi=(np.pi**0.5)/2
c1,c2,c3=4.08858*(10**12),(np.pi**0.5)/2,(np.pi**0.5)*np.exp(-0.25)*1j/4
c4=-1j*(np.pi**0.5)*np.exp(-1/8)/(4*root2)
a2b = 1.88973


@numba.jit(nopython=True)
def erfunc(z):
    t = 1.0 / (1.0 + 0.5 * np.abs(z))
    ans = 1 - t * np.exp( -z*z -  1.26551223 +
                        t * ( 1.00002368 +
                        t * ( 0.37409196 + 
                        t * ( 0.09678418 + 
                        t * (-0.18628806 + 
                        t * ( 0.27886807 + 
                        t * (-1.13520398 + 
                        t * ( 1.48851587 + 
                        t * (-0.82215223 + 
                        t * ( 0.17087277))))))))))
    return ans


@numba.jit(nopython=True)
def hermite_polynomial(x, degree, a=1):
    if degree == 0:
        return 1
    elif degree == 1:
        return -2*a*x
    elif degree == 2:
        x1 = (a*x)**2
        return 4*x1 - 2*a
    elif degree == 3:
        x1 = (a*x)**3
        return -8*x1 - 12*a*x
    elif degree == 4:
        x1 = (a*x)**4
        x2 = (a*x)**2
        return 16*x1 - 48*x2 + 12*a**2

@numba.jit(nopython=True)
def fcut(Rij, rcut): #checked
    return 0.5*(np.cos((np.pi*Rij)/rcut)+1)

@numba.jit(nopython=True)
def generate_data(size,z,atom,charges,coods,cutoff_r=12):
    """
    returns 2 and 3-body internal coordinates
    """
    
    twob=np.zeros((size,3))
    threeb=np.zeros((size,size,6))
    z1=z**0.8

    for j in range(size):
        rij=atom-coods[j]
        rij_norm=np.linalg.norm(rij)

        if rij_norm!=0 and rij_norm<cutoff_r:
            z2=charges[j]**0.8
            fcutij = fcut(rij_norm, cutoff_r)
            #fcutij=1.0
            twob[j]=rij_norm,np.sqrt(z1*z2),fcutij

            for k in range(size):
                if j!=k:
                    rik=atom-coods[k]
                    rik_norm=np.linalg.norm(rik)

                    if rik_norm!=0 and rik_norm<cutoff_r:
                        z3=charges[k]**0.8
                        
                        rkj=coods[k]-coods[j]
                        
                        rkj_norm=np.linalg.norm(rkj)

                        fcutik, fcutjk =  fcut(rik_norm, cutoff_r), fcut(rkj_norm, cutoff_r)      
                        fcut_tot = fcutij*fcutik*fcutjk
                        #fcut_tot=1.0

                        threeb[j][k][0] = np.minimum(1.0,np.maximum(np.dot(rij,rik)/(rij_norm*rik_norm),-1.0))
                        threeb[j][k][1] = np.minimum(1.0,np.maximum(np.dot(rij,rkj)/(rij_norm*rkj_norm),-1.0))
                        threeb[j][k][2] = np.minimum(1.0,np.maximum(np.dot(-rkj,rik)/(rkj_norm*rik_norm),-1.0))
                        
                        atm = (rij_norm*rik_norm*rkj_norm)**2
                        
                        charge = np.cbrt(z1*z2*z3)
                        
                        threeb[j][k][3:] =  atm, charge, fcut_tot

    return twob, threeb                        

@numba.jit(nopython=True)
def angular_integrals(size,threeb,nAs, grid1,astep = 0.02, order=2,alength=160,a=2.0):
    """
    evaluates the 3-body functionals using the trapezoidal rule
    """

    desc_size = nAs*(order+1)
    arr=np.zeros((alength,desc_size))

    theta=0
    
    for i in range(alength):
        f1 = np.zeros(desc_size)
        ga = grid1[i]
        costheta = ga[1]
        
        for j in range(size):

            for k in range(size):

                if threeb[j][k][-2]!=0:
                    
                    angle1,angle2,angle3,atm,charge,fcut_tot=threeb[j][k]

                    #x=costheta - angle1
                    x = theta - np.arccos(angle1)

                    exponent=np.exp(-a*x**2)

                    index = 0

                    for l in range(order+1):
                        
                        h = hermite_polynomial(x, l, a)
                        pref = charge*exponent*h*fcut_tot

                        for m in range(nAs):
                            if m==1:
                                gna = ga[m]*angle2*angle3
                            else:
                                gna = ga[m]
                            f1[index] += (pref*gna)/atm
                            index+=1

        arr[i]=f1
        theta+=0.02

    trapz=[np.trapz(arr[:,i],dx=astep) for i in range(arr.shape[1])]

    return trapz


@numba.jit(nopython=True)
def radial_integrals(size,rlength,twob,step_r,nRs1,nRs2, nRgrid, order=2,a=1,normalized=False):
    """
    evaluates the 2-body functionals using the trapezoidal rule
    """
    desc_size = (nRs1+nRs2)*(order+1)

    arr=np.zeros((rlength,desc_size))

    r=0
    
    for i in range(rlength):

        f1 = np.zeros(desc_size)
        gr = nRgrid[i]
        #print(gr.shape)

        for j in range(size):

            if twob[j][-2]!=0:
                dist,charge,fcutij=twob[j]
                x=r-dist

                if normalized==True:
                    norm=(erfunc(dist)+1)*half_rootpi
                    exponent=np.exp(-a*(x)**2)/norm
                
                else:
                    exponent=np.exp(-a*(x)**2)
                
                index = 0
                
                for k in range(order+1):
                    
                    h = hermite_polynomial(x,k,a)
                    pref = charge*exponent*h*fcutij

                    #for gnr in gr:
                    #    f1[index] += pref/gnr
                    #    index+=1

                    for gnr in gr[:nRs1]:
                        f1[index] += pref*gnr
                        index+=1

                    for gnr in gr[nRs1:]:
                        f1[index] += pref/gnr
                        index+=1

        r+=step_r
        arr[i]=f1
    
    trapz=[np.trapz(arr[:,i],dx=step_r) for i in range(arr.shape[1])]

    return trapz


@numba.jit(nopython=True)
def mbdf_local(charges,coods,grid1,grid2,rlength,alength,nRs1,nRs2,nAs,astep,order,a,a2,pad=29,step_r=0.1,cutoff_r=12):
    """
    returns the local MBDF representation for a molecule
    """
    size = len(charges)
    nr, na = (nRs1+nRs2)*(order+1), nAs*(order+1)
    desc_size = nr+na
    mat=np.zeros((pad,desc_size))
    
    assert size > 1, "No implementation for monoatomics"

    for i in range(size):
        twob,threeb = generate_data(size,charges[i],coods[i],charges,coods,cutoff_r)
        mat[i][:nr] = radial_integrals(size,rlength,twob,step_r,nRs1,nRs2,grid1,order,a)     
        mat[i][nr:] = angular_integrals(size,threeb,nAs,grid2,astep,order,alength,a2)

    return mat


def mbdf_global(charges,coods,asize,rep_size,keys,grid1,grid2,rlength,alength,step_r=0.1,cutoff_r=12,angular_scaling=2.4):
    """
    returns the flattened, bagged MBDF feature vector for a molecule
    """
    elements = {k:[[],k] for k in keys}

    size = len(charges)

    for i in range(size):
        elements[charges[i]][0].append(coods[i])

    mat, ind = np.zeros((rep_size,6)), 0

    assert size > 1, "No implementation for monoatomics"

    if size>2:

        for key in keys:
            
            num = len(elements[key][0])
            
            if num!=0:
                bags = np.zeros((num,6))
                
                for j in range(num):
                    twob,threeb = generate_data(size,key,elements[key][0][j],charges,coods,cutoff_r)

                    bags[j][:4] = radial_integrals(size,rlength,twob,step_r)     

                    bags[j][4:] = angular_integrals(size,threeb,alength,grid1=grid1,grid2=grid2,angular_scaling=angular_scaling)

                mat[ind:ind+num] = -np.sort(-bags,axis=0)
                
            ind += asize[key]
    
    elif size == 2:

        for key in keys:
            
            num = len(elements[key][0])
            
            if num!=0:
                bags = np.zeros((num,6))
                
                for j in range(num):
                    z1, z2, rij = charges[0]**0.8, charges[1]**0.8, coods[0]-coods[1]
        
                    pref, dist = z1*z2, np.linalg.norm(rij)

                    twob = np.array([[pref, dist], [pref, dist]])
                    
                    bags[j][:4] = radial_integrals(size,rlength,twob,step_r)     

                mat[ind:ind+num] = -np.sort(-bags,axis=0)
                
            ind += asize[key]

    return mat
                        

#@numba.jit(nopython=True)
def int_grid(alpha, nRs, nAs, cutoff_r, step_r,astep):
    
    rgrid = np.arange(0.0, cutoff_r, step_r)

    grid1 = []

    #for n in range(nRs):
    #    grid1.append(np.exp(-(alpha+((4*n))*rgrid)))
    #for n in range(nRs):
    #    grid1.append(np.exp(-(alpha+(2*n))*rgrid))
    #grid1.append(np.exp(-0.8*rgrid))
    grid1.append(np.exp(-1.5*rgrid))
    grid1.append(np.exp(-5.0*rgrid))
    #grid1.append(np.exp(-6.4*rgrid))

    #for n in range(nRs):
    #    grid1.append(2.2508*((rgrid+1)**(2*n + 3)))
    rgrid = np.arange(1.0, cutoff_r+1.0, step_r)

    #for n in range(nRs):
    #    grid1.append(2.2508*((rgrid)**((2*n)+1)))
    grid1.append(2.2508*((rgrid)**3.0))
    grid1.append(2.2508*((rgrid)**5.0))
    #grid1.append(2.2508*((rgrid)**alpha))

    angles = np.arange(0,np.pi,astep)
    
    grid2 = []

    for n in range(nAs):
        grid2.append(np.cos(n*angles))
    
    return np.array(grid1).T, np.array(grid2).T


@numba.jit(nopython=True)
def normalize(A,normal='mean'):
    """
    normalizes the functionals based on the given method
    """
    
    A_temp = np.zeros(A.shape)
    
    if normal=='mean':
        for i in range(A.shape[2]):
            
            avg = np.mean(A[:,:,i])

            if avg!=0.0:
                A_temp[:,:,i] = A[:,:,i]/avg
            
            else:
                pass
   
    elif normal=='min-max':
        for i in range(A.shape[2]):
            
            diff = np.abs(np.max(A[:,:,i])-np.min(A[:,:,i]))
            
            if diff!=0.0:
                A_temp[:,:,i] = A[:,:,i]/diff
            
            else:
                pass
    
    return A_temp


from joblib import Parallel, delayed

def generate_mbdf(nuclear_charges,coords,local=True,n_jobs=-1,pad=None,step_r=0.04,cutoff_r=8.0,step_a=0.02,nRs1=2,nRs2=2, nAs=4, order=4, alpha=1.5,normalized='min-max',progress_bar=False,a2=0.5):
    """
    Generates the local MBDF representation arrays for a set of given molecules

    :param nuclear_charges: array of arrays of nuclear_charges for all molecules in the dataset
    :type nuclear_charges: numpy array NxM, where N is the number of molecules and M is the number of atoms (can be different for each molecule)
    :param coords : array of arrays of input coordinates of the atoms
    :type coords: numpy array NxMx3, where N is the number of molecules and M is the number of atoms (can be different for each molecule)
    ordering of the molecules in the nuclear_charges and coords arrays should be consistent
    :param n_jobs: number of cores to parallelise the representation generation over. Default value is -1 which uses all available cores in the system
    :type n_jobs: integer
    :param pad: Number of atoms in the largest molecule in the dataset. Can be left to None and the function will calculate it using the nuclear_charges array
    :type pad: integer
    :param step_r: radial step length in Angstrom
    :type step_r: float
    :param cutoff_r: local radial cutoff distance for each atom
    :type cutoff_r: float
    :param step_a: angular step length in Radians
    :type step_a: float
    :param angular_scaling: scaling of the inverse distance weighting used in the angular functionals
    :type : float
    :param normalized: type of normalization to be applied to the functionals. Available options are 'min-max' and 'mean'. Can be turned off by passing False
    :type : string
    :param progress: displays a progress bar for representation generation process. Requires the tqdm library
    :type progress: Bool

    :return: NxPadx6 array containing Padx6 dimensional MBDF matrices for the N molecules
    """
    assert nuclear_charges.shape[0] == coords.shape[0], "charges and coordinates array length mis-match"
    
    lengths, charges = [], []

    for i in range(len(nuclear_charges)):
        
        q, r = nuclear_charges[i], coords[i]
        
        assert q.shape[0] == r.shape[0], "charges and coordinates array length mis-match for molecule at index" + str(i)

        lengths.append(len(q))

        charges.append(q.astype(np.float64))

    if pad==None:
        pad = max(lengths)

    #charges = np.array(charges)

    rlength = int(cutoff_r/step_r) 
    alength = int(np.pi/step_a) + 1

    grid1,grid2 = int_grid(alpha, nRs1, nAs, cutoff_r, step_r,astep=step_a)
    
    coords, cutoff_r = a2b*coords, a2b*cutoff_r

    if local:
        if progress_bar==True:

            from tqdm import tqdm    
            mbdf = Parallel(n_jobs=n_jobs)(delayed(mbdf_local)(charge,cood,grid1,grid2,rlength,alength,nRs1,nRs2,nAs,step_a,order,0.5,a2,pad,step_r,cutoff_r) for charge,cood in tqdm(list(zip(charges,coords))))

        else:
            mbdf = Parallel(n_jobs=n_jobs)(delayed(mbdf_local)(charge,cood,grid1,grid2,rlength,alength,nRs1,nRs2,nAs,step_a,order,0.5,a2,pad,step_r,cutoff_r) for charge,cood in zip(charges,coords))

        mbdf=np.array(mbdf)

        if normalized==False:

            return mbdf

        else:

            return normalize(mbdf,normal=normalized)
        
