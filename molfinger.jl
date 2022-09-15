using PyCall, ASE, ACSF, LinearAlgebra, JLD

function test_ASE()
    at = bulk("Si")
    #at1 = Atoms("H2", [1. 0.5; 0. √3])

    display(at)
    out = acsf(at)
    display(out)
end

function test_pycall()
    math = pyimport("math")
    println(math.sin(math.pi / 4)) # returns ≈ 1/√2 = 0.70710678...

    py"""
    import numpy as np
    
    def sinpi(x):
        return np.sin(np.pi * x)
    """
    py"sinpi"(1)
end


function test_julip()
    # problem parameters 
    r0 = 1.0  # atomic spacing 
    N = 11    # number of atoms in each direction 

    # generate positions  (for real crystals this is automated)
    #=
    t = LinRange(0.0, (N-1)*r0, N)
    o = ones(N)
    x, y = t * o', o * t'
    X = [ [1.0 0.5; 0.0 √3/2] * [x[:]'; y[:]']; zeros(N^2)' ]
    display(X)
    X = X |> vecs
    display(X)
    =#
    # generate the atoms object  (H is just a place-holder)
    
    at = Atoms([[1., 0., 0.], [0., 0., 0.], [√2, √2, √2]], 
                [0., 0., 0.], 
                [1., 1.], 
                [1., 8., 1.], 
                diagm([2.0, 2.0, 1.0]), 
                (false, false, false))
    at1 = Atoms([[1., 0., 0.], [0., 0., 0.], [√2, √2, √2]], 
                [0., 0., 0.], 
                [1., 1.], 
                [8., 8., 8.], 
                diagm([10.0, 2.0, 1.0]), 
                (false, false, false))
    display(at)
    #set_cell!(at, diagm([2.0*N, 2.0*N, 1.0]))
    #set_pbc!(at, (true, true, true))
    #set_constraint!(at, FixedCell(at))
    desc = acsf(at)
    desc1 = acsf(at1)
    display(desc)
    display(desc1)
    display(size(desc[1]))
    display(desc ≈ desc1)
end


function extract_descriptor()
    num_subset = 1000
    data = load("data/qm9_dataset_$num_subset.jld")["data"] # load subdataset
    for d ∈ data
        break
    end
    
    atom = Atoms() # compute descriptor as it is (for testing) length of vector: 29*51 atleast
    # 
end

extract_descriptor()