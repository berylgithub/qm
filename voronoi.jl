using Plots, Statistics, LaTeXStrings, LinearAlgebra, Distributions, JLD, StatsBase, Random
include("linastic.jl")
"""
dummy distance between two coordinates, should use "Mahalanobis distance" later
"""
function f_distance(x1, x2)
    return norm(x1 - x2) 
end

"""
overloader for Mahalanobis distance
params:
    - B, matrix computed from the diagonal*Qᵀ
    - w, variable vector of fingerprint, 
    - wk, the fingerprint which the distance wants to be computed to, similar(w)
"""
function f_distance(B, w, wk)
    return norm(B*(w-wk), 2)^2
end

"""
specialized distances
"""
function fcenterdist(F, T)
    D = zeros(size(F, 1), length(T))
    for j ∈ axes(D, 2)
        for i ∈ axes(D, 1)
            D[i, j] = norm((@view F[i,:]) - (@view F[T[j], :]), 2)^2
        end
    end
    return D
end

"""
compute D_k(w_l) and store it in file, O(n^2) algo
params:
    - W, the fingerprint matrix, ∈ Float64(n_data, n_finger)
    - B, linear transformer matrix from covariance, ∈ Float64(n_finger, n_finger)
    - filename, String
"""
function compute_distance_all(W, B)
    _, n_data = size(W)
    D = zeros(n_data, n_data) # symmetric matrix
    for j ∈ 1:n_data
        for i ∈ 1:n_data
            if i > j
                #D[i,j] = f_distance(W[:,i],W[:,j]) ## for testing, use grid distance calculator first, change later !!
                D[i,j] = f_distance(B,W[:,i],W[:,j])
            elseif i < j
                D[i,j] = D[j,i]
            end
        end
    end
    return D
end


"""
eldar's clustering algo by farthest minimal distance
***currently only ("default", "default") works***
params:
    - coords: matrix of data (or coords), Matrix{Float64}(n_dim, n_data)
    - M: total number of cluster centers, Int.
    - distance: this is specific for molecular computation, by default it shouldnt be changed.
        - "default" = uses 2norm distance
        - "mahalanobis" = uses Mahalanobis distance with mean <- wbar, and additional B linear transformer matrix
    - mode: defines the algorithm used to generate the cluster: 
        - "default" is farthest minimal distance
        - "fmd" farthest minimal distance
        - "fsd" farthest sum of distances
        - "fssd" farthest sum of squared distances
        String 
    - break_ties: defines how the algorithm behave in the face of multiple points fulfilling the "mode" param:
        - "default" break ties with the earliest point index
        - "fsd"
        - "fmd"
        String
outputs:
    - mean_point: the coordinate of the mean point, Vector{Float64}(n_dim)
    - center_ids: containing the sorted (by order of selection) IDs of the centers, Vector{Float64}(M)
"""
function eldar_cluster(coords, M; wbar = nothing, B = nothing, distance="default", mode="default", break_ties="default", get_distances=false)
    data_size = size(coords)[2] # compute once
    # Eldar's [*cite*] sampling algo, default ver: break ties by earliest index:
    # later move all of the matrices and vectors alloc outside:
    centers = zeros(Int64, data_size) # 1 if it is a center
    center_ids = Vector{Int64}() # sorted center id
    #centers[[1,3,4]] .= 1 # dum
    mean_distances = Vector{Float64}(undef, data_size) # distances from mean point
    distances = Matrix{Float64}(undef, data_size, M) # distances from k_x, init matrix oncew
    ## Start from mean of all points:
    mean_point = nothing
    if distance == "default"
        mean_point = vec(mean(coords, dims=2)) # mean over the data for each fingerprint
    elseif distance == "mahalanobis" # specialized for molecule
        mean_point = wbar
    end
    ref_point = mean_point # init with mean, then k_x next iter
    ## compute the distances of all points to the reference point
    if distance == "default"
        for i ∈ 1:data_size
            mean_distances[i] = f_distance(ref_point, coords[:, i])
        end
    elseif distance == "mahalanobis"
        for i ∈ 1:data_size
            mean_distances[i] = f_distance(B, ref_point, coords[:, i])
        end
    end
    
    ### get point with max distance from mean ("default", "default"), no differences for initial center:
    _, selected_id = findmax(mean_distances)
    centers[selected_id] = 1

    ### set the next k as the reference:
    ref_point = coords[:, selected_id]
    push!(center_ids, selected_id)

    # SELECT MODE
    if distance == "default" # DEFAULT:
        # farthest minimum distance mode:
        if mode == "fmd"
            # init useful vector:
            min_dist = Vector{Float64}(undef, data_size) # init vector containing min distances
            for m ∈ 1:M-1  ## To find k_x s.t. x > 1, for m ∈ M:
                ## Find largest distance:
                ### compute list of distances from mean:
                for i ∈ 1:data_size
                    distances[i, m] = f_distance(ref_point, coords[:, i]) #compute distances
                end
                ### min of column:        
                for i ∈ 1:data_size
                    min_dist[i] = minimum(distances[i, 1:m])
                end
                ### sort distances descending:
                sorted_idx = sortperm(min_dist, rev=true)
                selected_id = 0
                for id ∈ sorted_idx
                    if centers[id] == 0
                        centers[id] = 1
                        selected_id = id
                        break 
                    end
                end
                ### reassign ref point by the new center:
                ref_point = coords[:, selected_id]
                push!(center_ids, selected_id)
                #= println(m, " ", ref_point)
                println() =#
            end
        # farthest sums of distance mode:
        elseif mode == "fsd"
            sums_dist = zeros(data_size) # contains the sums of any funcs of distances
            for m ∈ 1:M-1
                ### compute list of distances from mean:
                for i ∈ 1:data_size
                    distances[i, m] = f_distance(ref_point, coords[:, i]) #compute distances
                    sums_dist[i] += distances[i, m] # compute and store the sum of distances
                end
                ### sort sd descending (same way as md):
                sorted_idx = sortperm(sums_dist, rev=true)
                selected_id = 0
                for id ∈ sorted_idx
                    if centers[id] == 0
                        centers[id] = 1
                        selected_id = id
                        break 
                    end
                end
                ### reassign ref point by the new center:
                ref_point = coords[:, selected_id]
                push!(center_ids, selected_id)
                #= println(m, " ", ref_point)
                println() =#
            end
        # farthest squared sums of distance mode:
        elseif mode == "fssd"
            sums_dist = zeros(data_size) # contains the sums of any funcs of distances
            for m ∈ 1:M-1
                ### compute list of distances from mean:
                for i ∈ 1:data_size
                    distances[i, m] = f_distance(ref_point, coords[:, i]) #compute distances
                    sums_dist[i] += distances[i, m]^2 # compute and store the SQUARED sum of distances
                end
                ### sort sd descending (same way as md):
                sorted_idx = sortperm(sums_dist, rev=true)
                selected_id = 0
                for id ∈ sorted_idx
                    if centers[id] == 0
                        centers[id] = 1
                        selected_id = id
                        break 
                    end
                end
                ### reassign ref point by the new center:
                ref_point = coords[:, selected_id]
                push!(center_ids, selected_id)
                #= println(m, " ", ref_point)
                println() =#
            end
        # minimal sums of inverse distances mode:
        elseif mode == "msid"
            sums_dist = zeros(data_size) # contains the sums of any funcs of distances
            for m ∈ 1:M-1
                ### compute list of distances from mean:
                for i ∈ 1:data_size
                    distances[i, m] = f_distance(ref_point, coords[:, i]) #compute distances
                    sums_dist[i] += 1/distances[i, m] # compute and store the sum of inverse distances
                end
                ### sort sd vector ascending, since we're taking the minimal:
                sorted_idx = sortperm(sums_dist)
                selected_id = 0
                for id ∈ sorted_idx
                    if centers[id] == 0
                        centers[id] = 1
                        selected_id = id
                        break 
                    end
                end
                ### reassign ref point by the new center:
                ref_point = coords[:, selected_id]
                push!(center_ids, selected_id)
                #= println(m, " ", ref_point)
                println() =#
            end
        # minimal sums of inverse SQUARED distance:
        elseif mode == "msisd"
            sums_dist = zeros(data_size) # contains the sums of any funcs of distances
            for m ∈ 1:M-1
                ### compute list of distances from mean:
                for i ∈ 1:data_size
                    distances[i, m] = f_distance(ref_point, coords[:, i]) #compute distances
                    sums_dist[i] += 1/(distances[i, m])^2 # compute and store the sum of som functions
                end
                ### sort sd vector ascending, since we're taking the minimal:
                sorted_idx = sortperm(sums_dist)
                selected_id = 0
                for id ∈ sorted_idx
                    if centers[id] == 0
                        centers[id] = 1
                        selected_id = id
                        break 
                    end
                end
                ### reassign ref point by the new center:
                ref_point = coords[:, selected_id]
                push!(center_ids, selected_id)
                #= println(m, " ", ref_point)
                println() =#
            end
        end
    elseif distance == "mahalanobis" # MOLECULE:
        # farthest minimum distance mode:
        if mode == "fmd"
            # init useful vector:
            min_dist = Vector{Float64}(undef, data_size) # init vector containing min distances
            for m ∈ 1:M-1  ## To find k_x s.t. x > 1, for m ∈ M:
                ## Find largest distance:
                ### compute list of distances from mean:
                @simd for i ∈ 1:data_size
                    distances[i, m] = f_distance(B, ref_point, (@view coords[:, i])) #compute distances
                end
                ### min of column:        
                for i ∈ 1:data_size
                    min_dist[i] = minimum(distances[i, 1:m])
                end
                ### sort distances descending:
                sorted_idx = sortperm(min_dist, rev=true)
                selected_id = 0
                for id ∈ sorted_idx
                    if centers[id] == 0
                        centers[id] = 1
                        selected_id = id
                        break 
                    end
                end
                ### reassign ref point by the new center:
                ref_point = coords[:, selected_id]
                push!(center_ids, selected_id)
                #= println(m, " ", ref_point)
                println() =#
            end
        # minimal sums of inverse distances mode:
        elseif mode == "msid"
            sums_dist = zeros(data_size) # contains the sums of any funcs of distances
            for m ∈ 1:M-1
                ### compute list of distances from mean:
                for i ∈ 1:data_size
                    distances[i, m] = f_distance(B, ref_point, coords[:, i]) #compute distances
                    sums_dist[i] += 1/distances[i, m] # compute and store the sum of inverse distances
                end
                ### sort sd vector ascending, since we're taking the minimal:
                sorted_idx = sortperm(sums_dist)
                selected_id = 0
                for id ∈ sorted_idx
                    if centers[id] == 0
                        centers[id] = 1
                        selected_id = id
                        break 
                    end
                end
                ### reassign ref point by the new center:
                ref_point = coords[:, selected_id]
                push!(center_ids, selected_id)
                #= println(m, " ", ref_point)
                println() =#
            end
        # minimal sums of inverse SQUARED distance:
        elseif mode == "msisd"
            sums_dist = zeros(data_size) # contains the sums of any funcs of distances
            for m ∈ 1:M-1
                ### compute list of distances from mean:
                for i ∈ 1:data_size
                    distances[i, m] = f_distance(B, ref_point, coords[:, i]) #compute distances
                    sums_dist[i] += 1/(distances[i, m])^2 # compute and store the sum of som functions
                end
                ### sort sd vector ascending, since we're taking the minimal:
                sorted_idx = sortperm(sums_dist)
                selected_id = 0
                for id ∈ sorted_idx
                    if centers[id] == 0
                        centers[id] = 1
                        selected_id = id
                        break 
                    end
                end
                ### reassign ref point by the new center:
                ref_point = coords[:, selected_id]
                push!(center_ids, selected_id)
                #= println(m, " ", ref_point)
                println() =#
            end
        end
    end
    if get_distances
        # since the distance to last ref point by default is not needed, but needed here:
        @simd for i ∈ 1:data_size
            distances[i, M] = f_distance(B, ref_point, (@view coords[:, i]))
        end
        return center_ids, mean_point, distances
    else
        return center_ids, mean_point
    end
end

"""
unused!!
"""
function usequence_unused(N, d; prt=0)
    #d, N = size(z)
    M=max(1000,N);
    z=rand(d,M);            # random reservoir of M vectors
    zerM=zeros(Int, M);        # for later vectorization
    x=zeros(d,N);           # storage for the sequence to be constructed

    u, s = [zeros(M) for _ ∈ 1:2]
    for k ∈ 1:N
        # pick a vector from the reservoir
        if k == 1 
            j = 1
        else 
            # find the reservoir vector with largest minimum distance
            # from the vectors already chosen
            umax, j = findmax(u)
        end
        x[:, k] = z[:,j]

        # update the reservoir
        zj = rand(d,1)  
        z[:, j] = zj

        # update minimum squared distance vector 
        onk = ones(Int, k)
        u[j] = minimum(sum((zj[:, onk] - x[:, 1:k]) .^ 2, dims=1))
        s .= vec(sum((z - x[:, k .+ zerM]) .^ 2, dims=1)); 
        if k == 1 
            u .= s
        else
            u .= min.(u, s) # elemwisemin
        end
    end
    return x
end

"""
overloader with z pre-generated
z is reservoir NOT the whole input data, must hold |z| < N_data, to preserve randomness, init the reservoir with random points. 
params:
    - z, reservoir matrix
    - init_labels, absolute labels of the z matrix
    - glob_labels, all labels ∖ init_labels
    - N number of chosen labels (num of iter)
"""
function usequence(z::Matrix{Float64}, F::Matrix{Float64}, init_labels::Vector{Int}, glob_labels::Vector{Int}, N::Int; rep=true)
    d, M = size(z)
    zerM=zeros(Int, M);                     # for later vectorization
    x=zeros(d,N);                           # storage for the sequence to be constructed, (dim, number of selected points)
    chosen_labels = Vector{Int}(undef, 0)          # final labels stored
    abs_labels = Dict()     # absolute indices chosen to the actual data points ordering

    u, s = [zeros(M) for _ ∈ 1:2]           # initialize 2 empty vectors
    for k ∈ 1:N
        # pick a vector from the reservoir
        idc = nothing
        if k == 1 
            j = 1
        else 
            # find the reservoir vector with largest minimum distance
            # from the vectors already chosen
            # check if j is already selected!:
            #umax, j = findmax(u)
            idc = findall(u .== maximum(u))
            # random sample until element not in chosen_label is found
            j = sample(idc, 1)[1]
            while j ∈ chosen_labels
                j = sample(idc, 1)[1]
            end
            umax = u[j] # how to find out which absolute label this is??
        end
        x[:, k] = z[:,j] # update points
        push!(chosen_labels, j) # and add to label
        # update the reservoir, instead of rand, pick randomly from available set of integers:
        new_j = sample(glob_labels, 1)[1] # random sample
        zj = F[:, new_j]
        if rep
            z[:, j] = zj
            abs_labels[j] = new_j
        end
        #deleteat!(init_labels, findfirst(el -> el == new_j, init_labels)) # remove new_j from ids
        # update minimum squared distance vector 
        onk = ones(Int, k)
        if rep
            u[j] = minimum(sum((zj[:, onk] - x[:, 1:k]) .^ 2, dims=1))
        end
        s .= vec(sum((z - x[:, k .+ zerM]) .^ 2, dims=1)); 
        if k == 1 
            u .= s
        else
            u .= min.(u, s) # elemwisemin
        end
    end
    # transform chosen local labels into global labels:
    display(chosen_labels)
    display(abs_labels)
    return x, chosen_labels # add labels return too
end

function test_grid()
    # inputs:
    indices_M = convert(Vector{Int64}, range(10,50,5))
    # ∀ requested M, do the algorithm:
    #M = 10 # number of centers
    # fixed coords, ∈ (fingerprint length, data length):
    len_finger = 2
    n_data = 70
    coords = Matrix{Float64}(undef, len_finger, n_data) # a list of 2d coord arrays for testing
    # fill fixed coords:
    counter = 1
    for i ∈ 1.:7. 
        for j ∈ 1.:10.
            coords[1, counter] = i # dim1 
            coords[2, counter] = j # dim2
            counter += 1
        end
    end

    #= len_finger = 3
    n_data = 70
    coords = rand(len_finger, n_data) # test 3-dimensional data =#

    # perturb points:
    perturb_val = .15
    perturb = rand(Uniform(-perturb_val, perturb_val), size(coords))
    coords .+= perturb
    for M ∈ [30]
        # compute B:
        wbar, C = mean_cov(coords, 35, n_data, len_finger)
        B = compute_B(C)
        display(wbar)
        display(B)

        # init params:
        ws = [nothing, wbar]
        Bs = [nothing, B]
        ds = ["default", "mahalanobis"]

        # test with Mahalanobis distance:
        for i ∈ eachindex(ds)
            for md ∈ ["fmd"] 
                center_ids, mean_point = eldar_cluster(coords, M, 
                                            wbar=ws[i], B=Bs[i], distance=ds[i], mode=md) # generate cluster centers
                #display(center_ids)
                # plot the points:
                s = scatter(coords[1, :], coords[2, :], legend=false, markersize = 3)
                scatter!(coords[1, center_ids[1:6]], coords[2, center_ids[1:6]], legend = false, markersize=7.5, color="blue") # datapoints
                scatter!(coords[1, center_ids[7:12]], coords[2, center_ids[7:12]], legend = false, markersize=6., color="blue")
                scatter!(coords[1, center_ids[13:18]], coords[2, center_ids[13:18]], legend = false, markersize=5, color="blue")
                scatter!(coords[1, center_ids[19:24]], coords[2, center_ids[19:24]], legend = false, markersize=4., color="blue")
                scatter!(coords[1, center_ids[25:30]], coords[2, center_ids[25:30]], legend = false, markersize=3, color="blue")
                # mean point:
                scatter!([mean_point[1]], [mean_point[2]], color="red")
                annotate!([mean_point[1]] .+ .1, [mean_point[2]] .+ .4, L"$\bar w$")
                # centers:
                #scatter!([coords[1, center_ids]], [coords[2, center_ids]], color="red", shape = :x, markersize = 10)
                for i ∈ eachindex(center_ids)
                    annotate!([coords[1, center_ids[i]]] .- .15, [coords[2, center_ids[i]]] .+ 0.4, L"$%$i$")
                end
                display(s)
                savefig(s, "clusterplot/$md"*"_$M.png")
            end
        end
    end
end

function test_usequence()
    # call usequence here:
    #N, d = (10, 2)
    #M = max(10, N)
    #z = rand(d, M)

    d = 2; N = 10_000
    F = Matrix{Float64}(undef, d, N) # a list of 2d coord arrays for testing
    # fill fixed coords:
    counter = 1
    for i ∈ 1.:100. 
        for j ∈ 1.:100.
            F[1, counter] = i # dim1 
            F[2, counter] = j # dim2
            counter += 1
        end
    end

    #perturb:
    #Random.seed!(123)
    #z .+= rand(Uniform(-.15, .15), size(z))
    #F_init = copy(F) # actual data, since the data will be changed by usequence op

    K = 10
    # t_cl1 = @elapsed begin
    #     center_ids, mean_point = eldar_cluster(F, K, distance="default", mode="fmd") # generate cluster centers
    # end
    #pl = scatter([z_init[1, center_ids]], [z_init[2, center_ids]], makershape = :star, legend=false)
    display(F)
    #display(pl)
    glob_labels = Vector{Int}(1:size(F, 2))
    init_labels = rand(1:size(F, 2), 500)
    glob_labels = setdiff(glob_labels, init_labels) # exclude the selected initial points
    z = F[:, init_labels]
    display(z)
    t_cl2 = @elapsed begin
        x, labels = usequence(z, glob_labels, K)
    end
    display(F_init)
    display([center_ids, labels])
    pl = scatter(x[1,:], x[2,:], markershape = :circle, legend=false)
    display(pl)
    pl = scatter(F[1,labels], F[2,labels], markershape = :cross, legend=false)
    display(pl)
    #display([t_cl1, t_cl2])
end

function test_u2()
    nf = 2; nd = 1000; k = 100
    A = rand(nf, nd) # looks like this is uniformly distributed
    #A .+= rand(1)
    #A = rand(Uniform(1., 100.), nf, nd)
    #A = (((A .- minimum(A)) ./ (maximum(A) .- minimum(A))) .* (10.75 - 1.5)) .+ 1.5  # change scaling
    display(A)
    Ks = []
    for i ∈ 1:3
        cpA = copy(A)
        push!(Ks, usequence(cpA, k)[2])
    end
    display([Ks[1] Ks[2] Ks[3]])
    println(Ks[1] == Ks[2], Ks[1] == Ks[3], Ks[2] == Ks[3])
    pl = scatter(A[1,Ks[1]], A[2,Ks[1]], markershape = :circle, legend=false)
    display(pl)
    pl = scatter(A[1,Ks[2]], A[2,Ks[2]], markershape = :circle, legend=false)
    display(pl)
    # summary: uniformly distributed data especially [0,1], gives the same selected points, even on lower dimensions
end

function test_distances()
    # inputs:
    indices_M = convert(Vector{Int64}, range(10,50,5))
    # ∀ requested M, do the algorithm:
    #M = 10 # number of centers
    # fixed coords, ∈ (fingerprint length, data length):
    len_finger = 2
    n_data = 70
    coords = Matrix{Float64}(undef, len_finger, n_data) # a list of 2d coord arrays for testing
    # fill fixed coords:
    counter = 1
    for i ∈ 1.:7. 
        for j ∈ 1.:10.
            coords[1, counter] = i # dim1 
            coords[2, counter] = j # dim2
            counter += 1
        end
    end

    # perturb points:
    perturb_val = .3
    perturb = rand(Uniform(-perturb_val, perturb_val), size(coords))
    coords .+= perturb

    M = 10
    wbar, C = mean_cov(coords, 17, n_data, len_finger)
    B = Matrix{Float64}(I, len_finger, len_finger)
    display(wbar)
    center_ids, mean_point, distances = eldar_cluster(coords, M, wbar=wbar, B=B, distance="mahalanobis", mode="fmd", get_distances=true) # generate cluster centers
    display(center_ids)
    display(distances)
    D = compute_distance_all(coords, B)
    display(D[:, center_ids])
end
