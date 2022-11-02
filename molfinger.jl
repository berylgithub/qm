using PyCall, ASE, ACSF, LinearAlgebra, JLD, Statistics, Distributions, DelimitedFiles, Printf

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
                diagm([10.0, 10.0, 10.0]), 
                (false, false, false))
    #coor = rand(Uniform(0., 20.), (3, 5))
    coor = transpose([1. 0. 0.; 0. 0. 0.; √2 √2 √2])
    display(coor)
    #coor = [coor[:, i] for i ∈ 1:size(coor)[2]]
    cellbounds = diagm([30., 30., 30.])
    display(cellbounds)
    at1 = Atoms(coor, 
                [0., 0., 0.], 
                [1., 1.], 
                [8., 8., 8.], 
                cellbounds, 
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
    #display(desc ≈ desc1)
end

"""
THE MAIN DEFAULT mode function to compute the descriptor given a structured Dict dataset/base.
Default mode means the vector output is stacked until the length is 1479 (29*51)
params:
    - filenames
"""
function extract_ACSF()
    max_coor = 30. # is actually 12. from the dataset, just to add more safer boundary
    cellbounds = diagm([max_coor, max_coor, max_coor])
    n_finger = 1479 # 29*51
    tdata = @elapsed begin
        dataset = load("data/qm9_dataset_1000.jld")["data"]
    end
    n_data = length(dataset)
    A = zeros(n_data, n_finger) 
    counter = 1
    #limiter = 100 # for prototyping
    tcomp = @elapsed begin
        open("data/ACSF_1000.txt","w") do io
            for d ∈ dataset
                # LIMITER for prototyping:
                #= if counter == limiter
                    break
                end =#
                # extract data from datasetbinary:
                coord = transpose(d["coordinates"])
                n_atom = d["n_atom"]
                # compute descriptor:
                at = Atoms(coord,
                        [0., 0., 0.], 
                        [1., 1.], 
                        [8., 8., 8.], 
                        cellbounds, 
                        (false, false, false)
                        )
                desc = acsf(at)
                # fill data matrix:
                for j ∈ 1:n_atom
                    A[counter, (j-1)*51 + 1:j*51] = desc[j]
                end
                # print file to txt here:
                ct = 1
                str = ""
                for c ∈ 1:n_finger
                    s = lstrip(@sprintf "%16.8e" A[counter, c])
                    if ct != n_finger
                        str *= s*"  " 
                    else
                        str *= s
                    end
                    ct += 1
                end
                print(io, str*"\n")
                println("datacounter = ", counter)
                counter += 1
            end
        end
    end
    #save("data/qm9_desc_acsf_1000.jld", "data", list_data)
    #display(load("data/qm9_desc_acsf_1000.jld")["data"])
    save("data/qm9_matrix_1000.jld", "data", A)
    A = load("data/qm9_matrix_1000.jld")["data"]
    display(A)
    println("time to load data: ",tdata)
    println("computing time: ",tcomp)
end


"""
descriptor symmetrizer (or fingerprint computer), by taking for example: the sum, sum(square), mult(i ≢ j) of descriptors; the analogy of a^2 + b^2 + ab.
mult not yet available
params:
    - desc, vector of vectors or Matrix, Float64
    - n_atom
    - len_desc, length of the vector of a descriptor
returns:
    - finger, Vector{Float64}
"""
function desc_symm!(finger, desc, n_atom, len_desc)
    # if desc is vector of vectors:
    for i ∈ 1:n_atom
        finger[1:len_desc] .+= desc[i] # sum 
        finger[len_desc+1:2*len_desc] .+= (desc[i].^2) # sum(square)
        # mult
    end 
end


"""
extracts the ACSF descriptor, however instead of being stacked, the vectors are: summed, summed(squared), and multiplied(i ≢ j);
params:
    - filenames
"""
function extract_ACSF_sum(infile, outfile)
    max_coor = 30. # is actually 12. from the dataset, just to add more safer boundary
    len_desc = 51 
    n_finger = 102 # 51 for sum, 51 for sum(squared)
    cellbounds = diagm([max_coor, max_coor, max_coor])
    tdata = @elapsed begin
        dataset = load(infile)["data"]
    end
    n_data = length(dataset)
    A = zeros(n_data, n_finger)
    finger = zeros(n_finger) # preallocated vector
    tcomp = @elapsed begin # start timer
        for i ∈ 1:n_data # loop dataset
            # generate atom datastructure:
            coord = transpose(dataset[i]["coordinates"])
            n_atom = dataset[i]["n_atom"]
            # compute descriptor:
            at = Atoms(coord,
                    [0., 0., 0.], 
                    [1., 1.], 
                    [8., 8., 8.], 
                    cellbounds, 
                    (false, false, false)
                    )
            desc = acsf(at)
            # compute fingerprint:
            desc_symm!(finger, desc, n_atom, len_desc)
            A[i, :] = finger
            finger .= 0. # reset vector
            # save to file?:
        end
    end
    println("data time = ",tdata)
    println("comp time = ",tcomp)
    save(outfile, "data", A)
    load(outfile)["data"]
end

"""
this returns vector of matrices for the ACSF of length N, each matrix ∈ Float64 (n_atom, n_f) however with different n_atom
"""
function extract_ACSF_array(infile, outfile)
    max_coor = 30. # is actually 12. from the dataset, just to add more safer boundary
    len_desc = 51 
    cellbounds = diagm([max_coor, max_coor, max_coor])
    tdata = @elapsed begin
        dataset = load(infile)["data"]
    end
    n_data = length(dataset)
    ACSF = Vector{Matrix{Float64}}(undef, n_data) # initialize output
    tcomp = @elapsed begin # start timer
        for i ∈ 1:n_data # loop dataset
            # generate atom datastructure:
            coord = transpose(dataset[i]["coordinates"])
            n_atom = dataset[i]["n_atom"]
            # compute descriptor:
            at = Atoms(coord,
                    [0., 0., 0.], 
                    [1., 1.], 
                    [8., 8., 8.], 
                    cellbounds, 
                    (false, false, false)
                    )
            desc = acsf(at)
            # compute fingerprint:
            A = zeros(n_atom, len_desc) # unavoidable reallocation, due to dynamic n_atom
            for i ∈ 1:n_atom
                A[i,:] .= desc[i]
            end
            ACSF[i] = A
        end
    end
    println("data-loading time = ",tdata)
    println("comp time = ",tcomp)
    save(outfile, "data", ACSF)
    load(outfile)["data"]
end


"""
transforms the descriptors to fingerprint matrix, usually not needed, as the output of the descriptor extractor is already in matrix
"""
function transform_desc_to_matrix()
    num_subset = 1000
    desc = load("data/qm9_desc_acsf_$num_subset.jld")["data"]
    n_data = length(desc)
    finger_length = 1479 # 29*51
    A = zeros(n_data, finger_length) # data matrix; n_data = num_subset
    t = @elapsed begin
        for i ∈ 1:n_data
            n_atom = length(desc[i])
            for j ∈ 1:n_atom
                A[i, (j-1)*51 + 1:j*51] = desc[i][j]
            end
        end
    end
    println("elapsed = ",t)
    save("data/qm9_matrix_$num_subset.jld", "data", A)
    display(load("data/qm9_matrix_$num_subset.jld")["data"])
end


function transform_to_ascii()
    A = load("data/qm9_matrix.jld")["data"]
    m = (s->(@sprintf "%16.8e" s)).(A)
    m = lstrip.(m)
    display(m)
    rows, cols = size(m)
    #writedlm("data/matsub.txt", m, "\t", quotes=false) # this works i think
    open("data/ACSF.txt","w") do io
        for r ∈ 1:rows
            str = ""
            count = 1
            for c ∈ 1:cols
                if count != cols
                    str *= m[r,c]*"  " 
                else
                    str *= m[r,c]
                end
                count += 1
            end
            print(io, str*"\n")
        end
    end
end


