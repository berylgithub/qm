using DelimitedFiles


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
function init_penalty_x!(x, f, u, opt, tb_maes, tb_centers)
    for i ∈ eachindex(tb_maes) # for each row
        if x ∈ tb_centers[i,:]
            f += tb_maes[i] # update fobj
            u += 1 # update count
        end
    end
    return f_penalty(f, u, opt), f, u
end

function test_pen()
    id_data = 1:6
    tb_centers = [1 2 3 4; 1 3 5 4; 1 2 5 3]
    tb_maes = [15.; 10.; 20]
    p = zeros(length(id_data)); f = zeros(length(id_data)); u = zeros(Int, length(id_data))
    for i ∈ id_data
        p[i], f[i], u[i] = init_penalty_x!(i, f[i], u[i], 10., tb_maes, tb_centers)
    end
    display([p, f, u])
end

"""
initialize opt,u(x),f(x) -> p(x) (u(x),f(x),p(x) table ∀x) given some simulation data tables
should be computed only once per batch
"""
function main_init_opttable() 
    # simulation data tables:
    tb_centers = readdlm("data/custom_CMBDF_centers_181023.txt", Int)[:, 1:100]
    tb_maes = vec(readdlm("result/deltaML/MAE_custom_CMBDF_centers_181023.txt"))
    # get minima:
    id_opt = argmin(tb_maes)
    opt = tb_maes[id_opt]
    # compute penalty of x:
    E = readdlm("data/energies.txt")
    n_data = length(E)
    id_data = 1:n_data
    p = zeros(n_data); f = zeros(n_data); u = zeros(Int, n_data)
    t = @elapsed begin
        for i ∈ id_data
            p[i], f[i], u[i] = init_penalty_x!(i, f[i], u[i], opt, tb_maes, tb_centers)
        end
    end
    writedlm("data/tsopt/table_penalties.txt", [p f u])
    writedlm("data/tsopt/opt.txt", opt)
    display([p f u])
    display(t)
end