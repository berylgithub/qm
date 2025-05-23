using DelimitedFiles, Random, StatsBase, LinearAlgebra, Combinatorics


include("utils.jl")
include("expdriver.jl")

"""
=========================
Tabu search w/ penalty
=========================
"""

"""
compute penalty given new f,u,opt
"""
function f_penalty(f, u, opt)
    (f + opt) / (u + 1)
end


"""
function to initialize fobj contrib and contrib count
    - x is the data index
    - f is the ∑fobj so far
    - u is the count so far
"""
function init_penalty_x(x, f, u, opt, tb_maes, tb_centers)
    for i ∈ eachindex(tb_maes) # for each row
        if x ∈ tb_centers[i,:]
            f += tb_maes[i] # update fobj
            u += 1 # update count
        end
    end
    return f_penalty(f, u, opt), f, u
end

"""
vectorized and binarized version, should be much faster
    - ids_int ∈ vector{Int} length = ndata
    - tb_centers ∈ Matrix{Int} rows converted to binaries, to avoid index search. Done each row to avoid memory bloat
"""
function init_penalties_x!(ps, fs, us, ids_int, opt, tb_maes, tb_centers::Matrix{Bool})
    for x ∈ ids_int 
        for i ∈ eachindex(tb_maes)
            if tb_centers[i,x] # tb_centers is binaries
                fs[x] += tb_maes[i] # sum the fobj involved
                us[x] += 1  # increment count where x is involved
            end
        end
    end
    ps[ids_int] .= f_penalty.(fs[ids_int], us[ids_int], opt) # compute penalty
end

"""
update the table values (p,f,u) given new score fobj and training set S
    - fobj ∈ R, opt ∈ R
    - S ∈ vector[0,1] length = ndata (binary)
    - ps, fs, us, ∈ vectors{R} length = ndata
"""
function update_penalties_x!(ps, fs, us, fobj, opt, S::Vector{Bool})
    fs[S] .+= fobj # increase fobj value
    us[S] .+= 1 # increment counter
    ps[S] .= f_penalty.(fs[S], us[S], opt)
end

"""
updates the set S: replaces some x∈S (up to m numbers) that has high penalty with some x∉S with low penalty
    - m ∈ Int > 0
!!! BUG POSSIBILITY, it is possible that the one that is added is not disjoint with S, check 3rd line of the function

"""
function update_set(S_int::Vector{Int}, ps::Vector{Float64}, n, m)
    S = int_to_bin(S_int, n)
    id_remove_S = sortperm(ps[S])[end-(m-1):end] # select last m (m-largest) penalties of S
    # get the id which contains the lowest penalties, make sure that no intersection between S and new_S
    sorted_ps = sortperm(ps)
    id_add = []
    c = 0
    for i ∈ sorted_ps
        if c == m
            break
        end
        if !S[i]
            push!(id_add, i)
            c += 1
        end
    end 
    # replace the ids:
    S[S_int[id_remove_S]] .= 0
    S[id_add] .= 1
    return bin_to_int(S)
end

"""
!! fill out and test later
the Tabu search algorithm for combinatorial problem
parameters:
    - fun_obj = objective function
    - f_data = Vector{Real} of initially known objective functions
    - P0 = Vector{AbstractArray{Int}} of initial points, e.g. P0 = [[0,1,0],[1,1,0],[1,1,1]]
    - ps0 = Vector{Real} of initial penalties
    - fs0 = Vector{Real} of sum of scores
    - us0 = Vector{Int} of num of simulations in which x ∈ eachindex(us0) participated in
    - 
optionals:
    - n_update = Int, number of elements swapped in each iteration
    - n_P = Int, number of elements included in the memory set
"""
function alg_tabu_search(fun_obj, opt, x_data, P0, ps0, fs0, us0; 
                        n_update = 1, n_P = 5, n_tol = 100, n_reset = 5, 
                        fun_x_transform = nothing, fun_params = [], fun_x_trans_params = [])
    # initialization:
    ps = copy(ps0); fs = copy(fs0); us = copy(us0); P = copy(P0)

    opt_upd = [] # memory for where the minimum is updated
    itol = ireset = 0 # for now: reset when a lower fobj is found
    iter = 1 # iteration tracker
    exit_signal = false # when the algorithm stops
    hps = []

    while true # irest < nrest
        itol = 0 # tolerance counter
        while itol < ntol
            # opt steps, put in a loop:
            println("==== iteration = ",iter)
            println("memory set = ", P)
            id_P = StatsBase.sample(1:n_P, 1, replace=false)[1] # StatsBase.sample the integer to slice the set rather than sampling the set itself
            S = P[id_P] # pick one from P (useful for parallel later)
            println("picked S to be updated = ", S,", id =",id_P)
            S = update_set(S, ps, n, n_update) # update the decision variable
            P[id_P] = S  # update the "memory" set
            x = int_to_bin(S,n)
            # file tracking to cut computation time hopefully:
            new_fobj = 0. 
            if tracking # check if the iterates is already in file:
                new_fobj = track_cache(path_tracker, fx_dummy, S, 1, [2, 2+length(S)-1];
                             fun_params = [A, b], fun_x_transform = int_to_bin, fun_x_transform_params = [n])
            else
                new_fobj = fun_obj(x, A, b)
            end
            println("new fobj = ", new_fobj, " by ", S)
            # found better point check:
            if new_fobj < opt
                println("lower fobj found!", new_fobj, " < ", opt)
                opt = new_fobj
                push!(opt_upd, iter)
                itol = 0
            else
                itol += 1 # increment 
            end
            # termination check: 
            if opt ≤ global_min
                println("global minimum found in ", iter, " iterations!!")
                exit_signal = true
                break
            end
            #println("penalties pre-update:", [ps fs us])
            update_penalties_x!(ps, fs, us, new_fobj, opt, x) # update penalty
            #println("penalties post-update:", [ps fs us])
            iter += 1
        end
        # check if exit signal is found
        if exit_signal
            break
        end
        # reset mechanism:
        println([P, P0])
        # change hyperparameters after n_reset: (randomly?)
        if ireset ≥ n_reset
            println("Hyperparameters change!!")
            n_update = StatsBase.sample(1:ns, 1, replace=false)[1]; n_P = StatsBase.sample(1:nsim, 1)[1]
            push!(hps, [nP, n_update])
            ireset = 0
        end
        ps = copy(ps0); fs = copy(fs0); us = copy(us0); P = copy(P0)
        ireset += 1
        println("restarted!!")
        println([P, P0], [nP, n_update])
    end
    println("number of restarts = ", ireset)
    println("updated opt at ", opt_upd)
    #println("hyperparameters = ", hps)
    #println("initial penalties:", [ps0 us0])
    #println("final penalties:", [ps us])
    println("opt0 = ",opt0)
    println("final optimal value = ", opt, " at iter ",iter, " w/ tolerance =", itol)
end

"""
initialize opt,u(x),f(x) -> p(x) (u(x),f(x),p(x) table ∀x) given some simulation data tables
should be computed only once per batch
"""
function main_init_opttable()
    # load basic info:
    E = readdlm("data/energies.txt")
    n_data = length(E)
    ids_data = 1:n_data 
    # simulation data tables:
    tb_centers = readdlm("data/custom_CMBDF_centers_181023.txt", Int)[:, 1:100]
    tb_maes = vec(readdlm("result/deltaML/MAE_10k_custom_CMBDF_centers_081123.txt"))
    # transform centers to binaries:
    tb_centers_bin = zeros(Bool, size(tb_centers, 1), n_data)
    for i ∈ axes(tb_centers, 1)
        tb_centers_bin[i, :] = int_to_bin(tb_centers[i,:], n_data)
    end
    # get minima:
    id_opt = argmin(tb_maes)
    opt = tb_maes[id_opt]
    # compute penalty of x:
    p = zeros(n_data); f = zeros(n_data); u = zeros(Int, n_data)
    t = @elapsed begin
        init_penalties_x!(p, f, u, ids_data, opt, tb_maes, tb_centers_bin)
    end
    writedlm("data/tsopt/table_penalties.txt", [p f u])
    writedlm("data/tsopt/opt.txt", opt)
    display([p f u])
    display(t)
end


"""
serial training-set-optimization(tsopt), using the algorithm on the test_main_master function, 
just to make sure something runs while waiting for the parallelization.

params:
    n_P # size of P
    n_update  # size of swapped elements per iteration
    n_tol; n_reset; # number of iteratoin tolerance (resets to initial points) and num of restart tolerance (changes hyperparameters) 
"""
function main_serial_tsopt(; n_P = 5, n_update = 10, n_tol = 10_000, n_reset = 5)
    # set necessary data:
    Random.seed!(777)
    DFs = [load("data/atomref_features.jld", "data"), [], [], []]
    dataset = load("data/qm9_dataset.jld", "data")
    E = vec(readdlm("data/energies.txt", Float64))
    f = load("data/CMBDF.jld", "data")
    # previously recorded simulations:
    centerss = Int.(readdlm("data/custom_CMBDF_centers_181023.txt")[:,1:100]) # all previously computed training sets
    fobjs = vec(readdlm("result/deltaML/MAE_custom_CMBDF_centers_181023.txt"))
    global_min = 1.0 # 1 kcal/mol target
    # scores initialization:
    opt = opt0 = readdlm("data/tsopt/opt.txt")[1] # the optimal value
    tb_pen = readdlm("data/tsopt/table_penalties.txt") # load penalties
    ps0 = tb_pen[:, 1]; fs0 = tb_pen[:, 2]; us0 = tb_pen[:, 3];
    ps = copy(ps0); fs = copy(fs0); us = copy(us0) # copy the initial scores
    
    # opt params:
    n_data = length(E); n_var = size(centerss, 2); # number of data and size of variable
    n_test = 10_000 # number of test data
    n_sim = size(centerss, 1) # number of training points for the algorithm
    

    # opt init:
    ireset = 0; iter = 1
    exit_signal = false
    opt_upd = []; hps = [] # trackers
    # trackers:
    path_tracker = "data/tsopt/tracker.txt" # initialize file cache (see if immediate write to disk is fast --> it is very fast, faster than reevaluation)
    path_ftrack = "data/tsopt/f_tracker.txt"
    path_opttrack = "data/tsopt/opt_tracker.txt"
    # take the n_P best training sets as the initial point:
    id_sort = sortperm(fobjs)
    P0 = [centerss[i,:] for i ∈ id_sort[1:n_P]]; P = copy(P0);
    # main loop:
    while true # loop indefinitely
        itol = 0
        while itol < n_tol
            println("==== iteration = ",iter)
            id_P = StatsBase.sample(1:n_P, 1, replace=false)[1] # StatsBase.sample the integer to slice the set rather than sampling the set itself
            S = P[id_P] # pick one from P
            println("picked id of S to be updated = ",id_P)
            S = update_set(S, ps, n_data, n_update) # update the decision variable
            P[id_P] = S  # update the "memory" set
            idtests = StatsBase.sample(setdiff(1:n_data, S), n_test, replace=false) # compute the test set
            # compute next objective function value by tracking the cache too:
            params = (E, dataset, DFs, f); arg_params = Dict(:idtests_in => idtests);
            new_fobj = track_cache(path_tracker, min_main_obj, S, 1, [2, 1+length(S)];
                            fun_params = params, fun_arg_params = arg_params)
            println("new fobj = ", new_fobj)
            writestringline(string.([new_fobj]), path_ftrack; mode="a") # write to f_tracker
            # found better point check:
            if new_fobj < opt
                println("lower fobj found!", new_fobj, " < ", opt)
                opt = new_fobj
                push!(opt_upd, iter)
                writedlm(path_opttrack, vcat(opt, S)') # write the new opt and its iterates to tracker
                itol = 0
            else
                itol += 1 # increment iteration tolerance
            end
            # termination check: 
            if opt ≤ global_min
                println("global minimum found in ", iter, " iterations!!")
                exit_signal = true
                break
            end
            update_penalties_x!(ps, fs, us, new_fobj, opt, int_to_bin(S,n_data))
            iter += 1
        end
        if exit_signal
            break
        end
        # reset mechanism:
        # change hyperparameters after n_reset: (randomly)
        if ireset ≥ n_reset
            println("Hyperparameters change!!")
            n_update = StatsBase.sample(1:n_var, 1)[1]; n_P = StatsBase.sample(1:n_sim, 1)[1]
            P0 = [centerss[i,:] for i ∈ id_sort[1:n_P]]; # reset the initial points to include n_P sets
            push!(hps, [n_P, n_update])
            ireset = 0
        end
        ps = copy(ps0); fs = copy(fs0); us = copy(us0); P = copy(P0) # reset to initial point
        ireset += 1
        println("restarted!!")
        println([n_P, n_update])
    end
    #println("number of restarts = ", ireset)
    println("updated opt at ", opt_upd)
    println("hyperparameters = ", hps)
    #println("initial penalties:", [ps0 us0])
    #println("final penalties:", [ps us])
    println("opt0 = ",opt0)
    println("final optimal value = ", opt, " at iter ",iter)
end

function test_pen()
    id_data = 1:6
    tb_centers = [1 2 3 4; 1 3 5 4; 1 2 5 3]
    tb_maes = [15.; 10.; 20]
    p = zeros(length(id_data)); f = zeros(length(id_data)); u = zeros(Int, length(id_data))
    for i ∈ id_data
        p[i], f[i], u[i] = init_penalty_x(i, f[i], u[i], 10., tb_maes, tb_centers)
    end
    display([p, f, u])
end

function test_update()
    E = readdlm("data/energies.txt")
    n_data = length(E)
    ps = fs = zeros(n_data); us = zeros(Bool, n_data)
    # load penalty infos:
    tbp = readdlm("data/tsopt/table_penalties.txt"); ps = tbp[:, 1]; fs = tbp[:, 2]; us = tbp[:, 3]
    display(ps)
    opt = readdlm("data/tsopt/opt.txt")[1]
    # dummy simulator returns (fobj, S): # turns out simulator doesnt need to return S, since its already recorwded in masterr
    fobj = 10.
    opt = fobj < opt ? fobj : opt # sorting new opt value
    display(opt)
    S = int_to_bin([1,2,10,130800], n_data)
    display([ps[S], fs[S], us[S]])
    i = [4,5,6] # StatsBase.sample non perturbed
    update_penalties_x!(ps, fs, us, fobj, opt, S)
    display([ps[S], fs[S], us[S]])
end

function test_update_set()
    Random.seed!(777)
    E = readdlm("data/energies.txt")
    n_data = length(E)
    ps = fs = zeros(n_data); us = zeros(Bool, n_data)
    # load penalty infos:
    tbp = readdlm("data/tsopt/table_penalties.txt"); ps = tbp[:, 1]; fs = tbp[:, 2]; us = tbp[:, 3]
    display(ps)
    opt = readdlm("data/tsopt/opt.txt")[1]
    # dummy S:
    S_int = StatsBase.sample(1:n_data, 100, replace=false)
    println(S_int)
    S = int_to_bin(S_int, n_data)
    update_set!(S, ps, 10)
    println(findall(S .== 1))
    println(setdiff(findall(S .== 1), S_int))
end

function fx_dummy(x, A, b)
    θ = A[x,:]\b[x] # train on index x
    nx = similar(x)
    for i ∈ eachindex(x)
        if x[i] == 0
            nx[i] = 1
        elseif x[i] == 1
            nx[i] = 0
        end
    end 
    return norm(A[nx,:]*θ - b[nx])
end

"""
test the optimization with dummy simulations
"""
function test_main_master()
    # generate dummy simulations:
    Random.seed!(777)
    n = 30; ns = 3;
    A = rand(n, 4)
    b = rand(n)
    xs = collect(combinations(1:n, ns))
    fobjs = [fx_dummy(int_to_bin(x, n),A,b) for x ∈ xs] # all of the possible fobjs
    id_min = argmin(fobjs) # the global minimum
    global_min = fobjs[id_min]
    println("global minimum to be found : ")
    display([global_min, xs[id_min], id_min])
    # initialize "training opt set":
    nsim = 10 # number of "previous simulations"
    id_select = StatsBase.sample(1:binomial(n, ns), nsim, replace=false)
    xs = xs[id_select]
    println(id_select)
    fobjs = fobjs[id_select]
    opt = minimum(fobjs)
    ps = zeros(n); fs = zeros(n); us = zeros(Int, n)
    xsbin = mapreduce(permutedims, vcat, int_to_bin.(xs, n))
    init_penalties_x!(ps, fs, us, 1:n, opt, fobjs, xsbin)
    
    # optimization init:
    tracking = true
    path_tracker = "data/tsopt/tracker_dummy.txt" # initialize file cache (see if immediate write to disk is fast)
    id_min = argmin(fobjs) 
    opt0 = opt = fobjs[id_min] # set the known minimum
    println("current 'known' minimum from data = ", opt, " by ", xs[id_min])
    id_sort = sortperm(fobjs)
    nP = min(length(id_select), 10) # (HYPERPARAMETER), number of included search set, length(id_select) = actual "previous data" after excluding the global minimum
    P0 = [xs[id] for id ∈ id_sort[1:nP]] # global set containing best known nset of training set S
    ps0 = copy(ps); fs0 = copy(fs); us0 = copy(us); P = copy(P0) # copy init state
    println(ps0)
    n_update = 2 # (HYPERPARAMETER) number of variables to be updated each iterations
    ntol = 10; nreset = 5 # number of tolerance and num of restart (in real scenario, no number of restart, it will restart indefinitely)
    opt_upd = []
    itol = ireset = 0 # for now: reset when a lower fobj is found
    iter = 1
    exit_signal = false
    hps = []
    while true # irest < nrest
        itol = 0 # tolerance counter
        while itol < ntol
            # opt steps, put in a loop:
            println("==== iteration = ",iter)
            println("memory set = ", P)
            id_P = StatsBase.sample(1:nP, 1, replace=false)[1] # StatsBase.sample the integer to slice the set rather than sampling the set itself
            S = P[id_P] # pick one from P (useful for parallel later)
            println("picked S to be updated = ", S,", id =",id_P)
            S = update_set(S, ps, n, n_update) # update the decision variable
            P[id_P] = S  # update the "memory" set
            x = int_to_bin(S,n)
            # file tracking to cut computation time hopefully:
            new_fobj = 0. 
            if tracking # check if the iterates is already in file:
                new_fobj = track_cache(path_tracker, fx_dummy, S, 1, [2, 2+length(S)-1];
                             fun_params = [A, b], fun_x_transform = int_to_bin, fun_x_transform_params = [n])
            else
                new_fobj = fx_dummy(x, A, b)
            end
            println("new fobj = ", new_fobj, " by ", S)
            # found better point check:
            if new_fobj < opt
                println("lower fobj found!", new_fobj, " < ", opt)
                opt = new_fobj
                push!(opt_upd, iter)
                itol = 0
            else
                itol += 1 # increment 
            end
            # termination check: 
            if opt ≤ global_min
                println("global minimum found in ", iter, " iterations!!")
                exit_signal = true
                break
            end
            #println("penalties pre-update:", [ps fs us])
            update_penalties_x!(ps, fs, us, new_fobj, opt, x) # update penalty
            #println("penalties post-update:", [ps fs us])
            iter += 1
        end
        # check if exit signal is found
        if exit_signal
            break
        end
        # reset mechanism:
        println([P, P0])
        # change hyperparameters after n_reset: (randomly)
        if ireset ≥ nreset
            println("Hyperparameters change!!")
            n_update = StatsBase.sample(1:ns, 1)[1]; nP = StatsBase.sample(1:nsim, 1)[1]
            P0 = [xs[id] for id ∈ id_sort[1:nP]] # reset the original P0 to include nP sets
            push!(hps, [nP, n_update])
            ireset = 0
        end
        ps = copy(ps0); fs = copy(fs0); us = copy(us0); P = copy(P0)
        ireset += 1
        println("restarted!!")
        println([P, P0], [nP, n_update])
    end
    println("number of restarts = ", ireset)
    println("updated opt at ", opt_upd)
    println("hyperparameters = ", hps)
    #println("initial penalties:", [ps0 us0])
    #println("final penalties:", [ps us])
    println("opt0 = ",opt0)
    println("final optimal value = ", opt, " at iter ",iter, " w/ tolerance =", itol)
end