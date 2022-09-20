using Plots, Statistics, LaTeXStrings, LinearAlgebra, Distributions

"""
dummy distance between two coordinates, should use "Mahalanobis distance" later
"""
function f_distance(x1, x2)
    return norm(x1 - x2) 
end

"""
eldar's clustering algo by farthest minimal distance
***currently only ("default", "default") works***
params:
    - coords: matrix of data (or coords), Matrix{Float64}(n_dim, n_data)
    - M: total number of cluster centers, Int.
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
function eldar_cluster(coords, M; mode="default", break_ties="default")
    data_size = size(coords)[2] # compute once
    # Eldar's [*cite*] sampling algo, default ver: break ties by earliest index:
    # later move all of the matrices and vectors alloc outside:
    centers = zeros(Int64, data_size) # 1 if it is a center
    center_ids = Vector{Int64}() # sorted center id
    #centers[[1,3,4]] .= 1 # dum
    mean_distances = Vector{Float64}(undef, data_size) # distances from mean point
    distances = Matrix{Float64}(undef, data_size, M) # distances from k_x, init matrix oncew
    ## Start from mean of all points:
    mean_point = vec(mean(coords, dims=2)) # mean over the data for each fingerprint
    ref_point = mean_point # init with mean, then k_x next iter
    for i ∈ 1:data_size
        mean_distances[i] = f_distance(ref_point, coords[:, i])
    end
    ### get point with max distance from mean ("default", "default"), no differences for initial center:
    _, selected_id = findmax(mean_distances)
    centers[selected_id] = 1

    ### set the next k as the reference:
    ref_point = coords[:, selected_id]
    push!(center_ids, selected_id)

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
            println(m, " ", ref_point)
            println()
        end
    # farthest sums of distance mode:
    elseif mode == "fsd"
        sums_dist = zeros(data_size) # contains the sums or squaresums of distances
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
            println(m, " ", ref_point)
            println()
        end
    # farthest squared sums of distance mode:
    elseif mode == "fssd"
        sums_dist = zeros(data_size) # contains the sums or squaresums of distances
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
            println(m, " ", ref_point)
            println()
        end
    # minimal sums of inverse distances mode:
    elseif mode == "msid"
        sums_dist = zeros(data_size) # contains the sums or squaresums of distances
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
            println(m, " ", ref_point)
            println()
        end
    end

    return center_ids, mean_point
end
"""
tempoorary main container
"""
function main()
    # inputs:
    indices_M = convert(Vector{Int64}, range(10,50,5))
    # ∀ requested M, do the algorithm:
    for M ∈ [30]
        #M = 10 # number of centers
        # fixed coords, ∈ (fingerprint length, data length):
        coords = Matrix{Float64}(undef, 2, 70) # a list of 2d coord arrays for testing
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
        perturb_val = .2
        perturb = rand(Uniform(-perturb_val, perturb_val), size(coords))
        coords .+= perturb

        for md ∈ ["fmd", "fsd", "fssd", "msid"] 
            center_ids, mean_point = eldar_cluster(coords, M, mode=md) # generate cluster centers
            # plot the points:
            s = scatter(coords[1, :], coords[2, :], legend = false) # datapoints
            # mean point:
            scatter!([mean_point[1]], [mean_point[2]], color="red")
            annotate!([mean_point[1]].+0.15, [mean_point[2]].+0.25, L"$\bar w$")
            # centers:
            #scatter!([coords[1, center_ids]], [coords[2, center_ids]], color="red", shape = :x, markersize = 10)
            for i ∈ eachindex(center_ids)
                annotate!([coords[1, center_ids[i]]], [coords[2, center_ids[i]]].+0.35, L"$%$i$")
            end
            display(s)
            savefig(s, "clusterplot/$md"*"_$M.png")
        end
    end
end

main()