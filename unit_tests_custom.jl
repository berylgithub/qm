include("RoSemi.jl")
include("utils.jl")

using Random, DelimitedFiles, Combinatorics

#= using JuMP, Juniper, Ipopt
using NOMAD
using GLPK
using NLopt =#
using Metaheuristics
using PyCall # test run CMBDF

"""
inner product kernel entry (reproducing kernel)
"""
function comp_repker_entry_t(u::AbstractArray, v::AbstractArray)::Float64
    return u'v
end

"""
atomic level repker: K_ll' = ∑_{ij} δ_{il,jl'} K(ϕ_il, ϕ_jl')
similar to gaussian kernel entry
"""
function comp_atomic_repker_entry_t(f1::AbstractArray, f2::AbstractArray, l1::Vector{String}, l2::Vector{String})::Float64
    entry = 0.
    @simd for i ∈ eachindex(l1)
        @simd for j ∈ eachindex(l2)
            @inbounds begin
                if l1[i] == l2[j] # manually set Kronecker delta using if 
                    @views d = comp_repker_entry_t(f1[i, :], f2[j, :]) # (vector, vector, scalar)
                    entry += d
                end 
            end
        end
    end
    return entry
end

function get_repker_atom_t(F1::AbstractArray, F2::AbstractArray, L1::Vector{Vector{String}}, L2::Vector{Vector{String}}; threading=true)::Matrix{Float64}
    if threading
        rowids = eachindex(L1); colids = eachindex(L2)
        iterids = Iterators.product(rowids, colids)
        A = ThreadsX.map((tuple_id) -> comp_atomic_repker_entry_t(F1[tuple_id[1]], F2[tuple_id[2]], L1[tuple_id[1]], L2[tuple_id[2]]), iterids)
    else
        nm1 = length(L1); nm2 = length(L2)
        A = zeros(nm1, nm2)
        @simd for j ∈ eachindex(L2) # col
            @simd for i ∈ eachindex(L1) # row
                @inbounds A[i, j] = comp_atomic_repker_entry(F1[i], F2[j], L1[i], L2[j])
            end
        end
    end
    return A
end

function test_dt()
    strrow = strcol = [["C", "C", "C"],["C", "C", "C"]]
    Random.seed!(603)
    f = [rand(3,3)' for i ∈ 1:2]
    K = get_repker_atom(f[1:2], f[1:2], strrow, strcol)
end


function test_warm_up()
    strrow = strcol = [["C", "C", "C"],["C", "C", "C"]]
    Random.seed!(603)
    f = [rand(3,3) for i ∈ 1:2]
    get_repker_atom(f[1:2], f[1:2], strrow, strcol)
    c = 2048.
    get_gaussian_kernel(f[1:2], f[1:2], strrow, strcol, c)
    println("warm up done!")
end

function test_actual()
    Random.seed!(603)
    dataset = load("data/qm9_dataset.jld", "data")
    f = load("data/ACSF_51.jld", "data")
    idrows = 1:1000
    idcols = 1:100
    frow = f[idrows]; fcol = f[idcols]; atomsrows = [d["atoms"] for d ∈ dataset[idrows]]; atomscols = [d["atoms"] for d ∈ dataset[idcols]];
    println("DPK:")
    @time K1 = get_repker_atom(frow, fcol, atomsrows, atomscols)

    println("GK:")
    c = 2048.
    @time K1 = get_gaussian_kernel(frow, fcol, atomsrows, atomscols, c)
end

function test_kernels()
    test_warm_up()
    test_actual()
end

function test_corr()
    n_ids = [100, 150, 500]
    println("correctness test")
    Random.seed!(603)
    E = readdlm("data/energies.txt")
    dataset = load("data/qm9_dataset.jld", "data")
    numrow = length(E)
    main_kernels_warmup()
    idall = 1:numrow
    idtest = StatsBase.sample(idall, n_ids[end], replace=false)
    idrem = setdiff(idall, idtest) # remainder ids
    max_n = maximum(n_ids[1:end-1]) # largest ntrain
    max_idtrains = StatsBase.sample(idrem, max_n, replace=false)
    idtrainss = map(n_id -> max_idtrains[1:n_id], n_ids[1:end-1]) # vector of vectors
    f = load("data/FCHL19.jld", "data")
    Fds = [load("data/atomref_features.jld", "data")]
    atomsrow = [d["atoms"] for d ∈ dataset]
    ET = E
    idtrain = idtrainss[1]
    # try using the pre-generated ids:
    idss = load("data/test_corr_ids.jld", "data")
    idtrain = idss[1]; idtest = idss[3]; max_idtrains = idss[2]
    atomscol = [d["atoms"] for d ∈ dataset[max_idtrains]]
    Kr = get_repker_atom(f, f[max_idtrains], atomsrow, atomscol)
    println(idtrain)
    θ = Fds[1][idtrain, :]\ET[idtrain];
    Ea = Fds[1]*θ
    MAEtrain = mean(abs.(ET[idtrain] - Ea[idtrain]))*627.503
    MAEtest = mean(abs.(ET[idtest] - Ea[idtest]))*627.503
    println([MAEtrain, MAEtest])
    ET -= Ea
    trids = indexin(idtrain, max_idtrains)
    println("new kernel code:")
    K = Kr
    Ktr = K[idtrain, trids]
    Ktr[diagind(Ktr)] .+= 1e-8
    θ = Ktr\ET[idtrain]
    Epred = Ktr*θ
    MAEtrain = mean(abs.(ET[idtrain] - Epred))*627.503
    # pred:
    Kts = K[idtest, trids]
    Epred = Kts*θ
    MAEtest = mean(abs.(ET[idtest] - Epred))*627.503
    println([MAEtrain, MAEtest])
end


"""
==============
opt tests
==============
"""

"""
dummy obj function given kernel K(H) with fixed hyperparameters H
"""
function fobj_dummy_lsq(x, n; A=zeros(), b=zeros())
    #= train_ids = grange[x]
    test_ids = setdiff(grange, train_ids)
    θ = A[train_ids,:]\b[train_ids] # train on index x
    bpred = A[test_ids,:]*θ
    return norm(bpred-b[test_ids]) =# # error of the test set

    θ = A[x,:]\b[x] # train on index x
    nx = similar(x)
    for i ∈ eachindex(x)
        if x[i] == 0
            nx[i] = 1
        elseif x[i] == 1
            nx[i] = 0
        end
    end 
    return norm(A[nx,:]*θ - b[nx]), [0.0], [sum(x) - n]
end

function ssquared(x)
    return sum(x .^ 2)
end


#= """
simple test for JuMP NLP with constraints
"""
function test_JUMP()
    # ================
    # case 1: simple LP but solved by NLP (Ipopt):
    # dummy fobj:
    #= function fobj(x,y;h=0.)
        return 12*x + 20*y + h
    end
    model = Model()
    register(model, :fobj, 2, fobj; autodiff=true) # register custom fobj, number of optimized variables = 2
    @variable(model, x >= 0)
    @variable(model, y >= 0)
    @constraint(model, c1, 6x + 8y >= 100)
    @constraint(model, c2, 7x + 12y >= 120)
    @objective(model, Min, fobj(x,y;h=1000))
    set_optimizer(model, Ipopt.Optimizer)
    optimize!(model)
    display(objective_value(model))
    display([value(x), value(y)]) =#

    # ===============
    # case 2: dummy underdetermined lsq, binary selection of array
    Random.seed!(777)
    A = rand(100, 20)
    b = rand(100)
    grange = 1:length(b)
    function fobj(x::T...; A=zeros(), b=zeros(), grange=[]) where {T<:Real}
        #= trainids = grange[x]
        testids = setdiff(grange, trainids)
        θ = A[trainids,:]\b[trainids] # train on index x
        bpred = A[testids,:]*θ
        return norm(bpred-b) =# # error of the test 

        Dx = diagm(x)
        θ = (Dx*A)\(Dx*b)
        nx = similar(x)
        for i ∈ eachindex(x)
            if x[i] == 0
                nx[i] = 1
            elseif x[i] == 1
                nx[i] = 0
            end
        end
        Dnx = diagm(nx)
        return norm(Dnx*A*θ - Dnx*b)
    end
    model = Model()
    register(model, :fobj, 1, fobj; autodiff=true) # register custom fobj, number of optimized variables = 2
    @variable(model, x[1:length(b)], Bin)
    @constraint(model, sum(x) == 10)
    @objective(model, Min, fobj(x;A=A, b=b, grange=grange))
    set_optimizer(model, Ipopt.Optimizer)
    optimize!(model)
    display(objective_value(model))
    display([value(x)])
end =#

function test_NOMAD() # !! nomad froze for this problem
    # problem:
    Random.seed!(777)
    n_prob=20
    A_lsq = rand(n_prob, 5)
    b_lsq = rand(n_prob)
    grange = 1:n_prob
    # objective:
    function fx(x; A=[], b=[], grange=[])
        x = Bool.(round.(x))
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

    # blackbox
    function bb(x; A=[], b=[], grange=[])
        #f = (x[1]- 1)^2 * (x[2] - x[3])^2 + (x[4] - x[5])^2 +c
        f = fx(x, A=A, b=b, grange=grange)
        bb_outputs = [f]
        success = true
        count_eval = true
        return (success, count_eval, bb_outputs)
    end
    # linear equality constraints
    #= A = [1.0 1.0 1.0 1.0 1.0;
        0.0 0.0 1.0 -2.0 -2.0]
    b = [5.0; -3.0]
    =#
    A = permutedims(ones(n_prob))
    b = [10.]

    # Define blackbox
    p = NomadProblem(n_prob, 1, ["OBJ"], # the equality constraints are not counted in the outputs of the blackbox
                    x->bb(x;A=A_lsq, b=b_lsq, grange=grange);
                    input_types = repeat(["R"], n_prob),
                    granularity = Float64.(vcat(0,ones(n_prob-1))),
                    #lower_bound = -10.0 * ones(5),
                    #upper_bound = 10.0 * ones(5),
                    A = A, b = b)

    # Fix some options
    p.options.max_bb_eval = 10_000

    # Define starting points. It must satisfy A x = b.
    #= x0 = [0.57186958424864897665429452899843;
        4.9971472653643420613889247761108;
        -1.3793445664086618762667058035731;
        1.0403394252630473459930726676248;
        -0.2300117084673765077695861691609] =#
    x0 = Float64.(zeros(Bool, n_prob))
    x0[[1,3,5,7,9,11,13,15,17,19]] .= 1. # to satisfy the linear constraints, assign random indices as 1

    # Solution
    result = solve(p, x0)
    println("Solution: ", result.x_best_feas)
    println("Satisfy Ax = b: ", A * result.x_best_feas ≈ b)
    println("And inside bound constraints: ", all(-10.0 .<= result.x_best_feas .<= 10.0))
end

function test_Alpine() # has some promise for MINLP with convergence properties BUT requires Gurobi
    #= m = JuMP.Model(Ipopt.Optimizer)
    # ----- Variables ----- #
    @variable(m, -10 <= x <= 10)
    @variable(m, -10 <= y <= 10)

    @NLobjective(m, Min, 100 * (y - x^2)^2 + (1 - x)^2)

    optimize!(m)
    display(objective_value(m))
    display([value(x), value(y)]) =#

    Random.seed!(777)

    nlp_solver = get_ipopt() # local continuous solver
    mip_solver = get_gurobi() # convex mip solver
    minlp_solver = get_juniper(mip_solver, nlp_solver) # local mixed-intger solver
    alpine = JuMP.optimizer_with_attributes(
        Alpine.Optimizer,
        "minlp_solver" => minlp_solver,
        "nlp_solver" => nlp_solver,
        "mip_solver" => mip_solver,
    ) 
    m = JuMP.Model(alpine)
    A = rand(100, 20)
    b = rand(100)
    grange = 1:length(b)
    function fobj(x; A=zeros(), b=zeros(), grange=[])
        trainids = grange[x]
        testids = setdiff(grange, trainids)
        θ = A[trainids,:]\b[trainids] # train on index x
        bpred = A[testids,:]*θ
        return norm(bpred-b) # error of the test
        #return sum(x.^2)
    end
    register(m, :fobj, 1, fobj; autodiff=true) # register custom fobj, number of optimized variables = 2
    @variable(m, x[1:length(b)], Bin)
    @constraint(m, sum(x) == 10)
    @objective(m, Min, fobj(x;A=A, b=b, grange=grange))
    optimize!(m)
    display(objective_value(m))
    display([value(x)])
end

function test_Juniper()
    #= ipopt = optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>0)
    optimizer = optimizer_with_attributes(Juniper.Optimizer, "nl_solver"=>ipopt)
    model = Model(optimizer)
    v = [10, 20, 12, 23, 42]
    w = [12, 45, 12, 22, 21]
    @variable(model, x[1:5], Bin)
    @objective(model, Max, v' * x)
    @constraint(model, sum(w[i]*x[i]^2 for i in 1:5) == 45)
    optimize!(model)
    println(termination_status(model))
    println(objective_value(model))
    println(value.(x)) =#

    # ==== test dummy lsq ====
    ipopt = optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>0)
    optimizer = optimizer_with_attributes(Juniper.Optimizer, "nl_solver"=>ipopt)
    model = Model(optimizer)
    register(model, :fobj_dummy_lsq, 1, fobj_dummy_lsq; autodiff=true)
    Random.seed!(777)
    A = rand(20, 5)
    b = rand(20)
    grange = 1:length(b)
    @variable(model, x[1:length(b)], Bin)
    #@objective(model, Min, x->fobj_dummy_lsq(x; A=A, b=b, grange=grange))
    @objective(model, Min, mean(abs.((A[x,:] - b[x]))))
    @constraint(model, sum(x) == 10)
    optimize!(model)
    println(termination_status(model))
    println(objective_value(model))
    println(value.(x))
end

function test_GLPK() # from chatGPT -> doesnt work
    m = Model(GLPK.Optimizer)

    @variable(m, x[1:3], Bin)
    @constraint(m, 2x[1] + x[2] <= 5)
    @constraint(m, x[2] + 3x[3] <= 6)

    function black_box_objective(x::Vector{Float64})
        # Calculate the objective using x
        # Example: Replace this with your actual objective calculation
        obj_value = sum(x)
        return obj_value
    end

    function evaluate_callback(cb_data)
        x_values = JuMP.value.(x)  # Get the current variable values
    
        obj_value = black_box_objective(x_values)  # Evaluate the black-box objective
    
        JuMP.set_objective_value(m, obj_value)  # Set the objective value in the model
    end

    register(m, :evaluate_callback, evaluate_callback)
    set_optimizer_attribute(m, "MOI.Raw", "callback", evaluate_callback)

    optimize!(m)
    display(objective_value(m))
    display([value(x)])
end

function test_NLOpt()
    # problem:
    Random.seed!(777)
    n_prob=20
    A = rand(n_prob, 5)
    b = rand(n_prob)
    # objective:
    function fx(x; c=0., A=[], b=[])
        #= x = Bool.(round.(x))
        θ = A[x,:]\b[x] # train on index x
        nx = similar(x)
        for i ∈ eachindex(x)
            if x[i] == 0
                nx[i] = 1
            elseif x[i] == 1
                nx[i] = 0
            end
        end 
        return norm(A[nx,:]*θ - b[nx]) =#
        return sum(x .^ 2) + c
    end

    function myconstraint(x)
        return sum(x) - 10
    end

    opt = Opt(:GN_ISRES, 2)
    opt.lower_bounds = 0
    opt.upper_bounds = 1
    #opt.xtol_rel = 1e-4

    opt.min_objective = x -> fx(x,c=10)
    #inequality_constraint!(opt, (x,g) -> myconstraint(x,g,2,0), 1e-8)
    #inequality_constraint!(opt, (x,g) -> myconstraint(x,g,-1,1), 1e-8)
    inequality_constraint!(opt, (x) -> myconstraint(x), 0.)

    (minf,minx,ret) = optimize(opt, vcat(ones(10), zeros(10)))
    numevals = opt.numevals # the number of function evaluations
    println("got $minf at $minx after $numevals iterations (returned $ret)")
end


"""
wanted features:
    - supply init points --> done
    - callback --> done by passing logger
    - stopping criteria (restart when stuck, etc) --> add reset by setting the best found points as initial points
"""
function test_MH()
    # test with the usual dummy problem:
    Random.seed!(777)
    n = 40; ns = 4;
    A = rand(n, 4)
    b = rand(n)
    xs = collect(combinations(1:n, ns))
    fobjs = [fobj_dummy_lsq(int_to_bin(x, n), ns; A=A, b=b) for x ∈ xs]
    display(length(fobjs))
    id_min = argmin(fobjs) # the global minimum
    global_min = fobjs[id_min]
    println("global minimum to be found : ")
    display([global_min, xs[id_min], id_min])
    # optimization:
    N = 500; p_cross = 0.5; p_mutate = 1e-5 # GA params
    init_xs = mapreduce(permutedims, vcat, [int_to_bin(x,n) for x ∈ xs[1:N]]) # take N initial points:
    display(init_xs)
    options = Options(f_calls_limit = Inf, time_limit = Inf, iterations = 10_000, verbose=true)
    information = Information(f_optimum = global_min[1])
    
    algo = GA(;
        N = N,
        p_mutation  = p_mutate,
        p_crossover = p_cross,
        initializer = RandomInBounds(;N=N),
        selection   = TournamentSelection(;N=N),
        crossover   = UniformCrossover(;p=p_cross),
        mutation    = BitFlipMutation(;p=p_mutate),
        environmental_selection = ElitistReplacement(),
        information = information,
        options = options
    )
    set_user_solutions!(algo, init_xs, x->fobj_dummy_lsq(x,ns; A=A, b=b)); # set initial solutions
    result = Metaheuristics.optimize(x->fobj_dummy_lsq(x,ns; A=A, b=b), BitArraySpace(n), 
                                    algo, logger = status -> MH_log_result(status, "test.txt"))
    x = minimizer(result)
    display([bin_to_int(x), minimum(result)])
    display([result.f_calls, result.h_calls])
    display(termination_status_message(result))
    display(fobj_dummy_lsq(x,ns; A=A, b=b))
    
    
end

"""
a way to track the best iterates in each iteration
"""
function MH_log_result(status, filepath)
    writestringline(string.(vcat(minimum(status), minimizer(status))), filepath; mode= "a")
end

function test_call_QML()
    
end