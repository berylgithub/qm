using Plots, DelimitedFiles, LaTeXStrings, Printf, Statistics
include("utils.jl")

function plot_mae()
    molnames = readdir("result")[2:end]
    count = 1
    m = molnames[1]
    N_set = parse.(Int, readdlm("result/"*m*"/err_$m.txt", '\t', String, '\n')[end-16:end, 2])
    println(N_set)
    for m ∈ molnames
        MAEs = parse.(Float64, readdlm("result/"*m*"/err_$m.txt", '\t', String, '\n')[end-16:end,3])
        sidx = sortperm(MAEs) # sort MAE ascending
        println(sidx, " ", MAEs[sidx], " ", N_set[sidx])
        sNset = string.(N_set[sidx]); sMAEs = MAEs[sidx];
        s = scatter(sNset, sMAEs, xticks = sNset,
                    title = m, xlabel = "N", ylabel = "MAE (kcal/mol)", xrotation = -45, xtickfontsize=6, legend=false)
        display(s)
        savefig(s, "plot/MAE_$m.png")
        count += 1
    end
    #mean runtime:
    T = 0.
    for i ∈ eachindex(molnames)
        m = molnames[i]
        mT = parse.(Float64, readdlm("result/"*m*"/err_$m.txt", '\t', String, '\n')[:, end-2:end-1])
        T += sum(mT)/size(mT)[1]
    end
    println(T/length(molnames))

end

function plot_mae_spec()
    molname = "H7C8N1"
    M_set = parse.(Int, readdlm("result/"*molname*"/err_$molname.txt", '\t', String, '\n')[end-9:end, 1])
    MAE = parse.(Float64, readdlm("result/"*molname*"/err_$molname.txt", '\t', String, '\n')[end-9:end,3])
    MAE_old = parse.(Float64, readdlm("result/"*molname*"/err_$molname.txt", '\t', String, '\n')[17+8:17+17,3])
    display(M_set)
    display(MAE)
    display(MAE_old)
    s = scatter(M_set, [MAE_old, MAE], xticks = M_set, markershape = [:cross :xcross], labels = ["MAD" "farthest-dist"], legend_position = :outertopright,
                title = molname, xlabel = L"$M$", ylabel = "MAE (kcal/mol)", xrotation = -45, xtickfontsize=6)
    display(s)
    savefig(s, "plot/MAE_$molname.png")
end

"""
ΔML stuffs:
"""

"""
automatic yticks generator given the number of desired points, and data points
"""
function yticks_generator(data, n)
    n = n+1
    mind = minimum(data); maxd = maximum(data)
    q25 = quantile(data, .1); q75 = quantile(data, .9); # quantiles
    mid = median(data)
    # generate left range and right range of median:
    nhalf = n ÷ 2;
    # split to 4 parts rather than 2?:
    # left:
    mult = max(10^(ndigits(Int(round(abs(mid - q25)))) - 1), 10)
    rl = range(q25, mid, nhalf); rl = rl .- (rl .% mult)
    # right: 
    mult = max(10^(ndigits(Int(round(abs(q75 - mid)))) - 1), 10)
    rr = range(mid, q75, nhalf); rr = rr .- (rr .% mult)
    # combine:
    yticks = [mind; rl; rr; maxd]
    yticks = yticks[yticks .> 0]
    return unique(yticks)
end

"""
plot prototype for delta levels
"""
function plot_MAE_db()
    tb = readdlm("result/deltaML/MAE_enum_ns_dn_PCAdn_020723.txt")
    tbsel = readdlm("result/deltaML/MAE_enum_s2_dn_PCAdn_020723.txt")
    
    # plot prototype for each {feature, model, solver}:
    ind = Dict("acsf" => [13:16, 29:32], "soap" => [41:44, 57:60], "fchl19" => [77:80, 85:88]) # db indexing
    inn = Dict("acsf" => [13:16, 29:32, 29:32, 45:48, 45:48], 
                "soap" => [57:60, 73:76, 73:76, 89:92, 89:92], 
                "fchl19" => [97:100, 113:116, 117:120, 129:132, 133:136]) # indices for dn output

    tb_da = tb[inn["fchl19"][1], :]
    tb_db = tb[inn["fchl19"][2], :]
    tbsel_db = tbsel[inn["fchl19"][3], :]
    tb_dn = tb[inn["fchl19"][4], :]
    tbsel_dn = tbsel[inn["fchl19"][5], :]
    
    MAEs = vcat(tb_da[:, 7], tb_db[:, 7], tb_dn[:, 7], tbsel_db[:, 7], tbsel_dn[:, 7])
    xticks = tb_da[:, 1]
    xtformat = string.(map(x -> @sprintf("%.0f",x), xticks))
    yticks = round.(vcat(maximum(MAEs), minimum(MAEs)), digits=3)
    yticks = vcat(yticks, range(10, 50, 5))
    ytformat = vcat(string.(yticks[1:2]), map(x -> @sprintf("%.0f",x), yticks[3:end]))
    p = plot(tb_da[:, 1], [tb_da[:, 7], tb_db[:, 7], tbsel_db[:, 7], tb_dn[:, 7], tbsel_dn[:, 7]],
        yticks = (yticks, ytformat), xticks = (xticks, xtformat),
        xaxis = :log, yaxis = :log,
        markershape = [:xcross :cross :rect :auto :auto], markersize = [6 6 4 5 5],
        labels = ["MAE(da)" "MAE(db)" "MAE(db,sel)" "MAE(dn)" "MAE(dn,sel)"], xlabel = "Ntrain", ylabel = "MAE (kcal/mol)")
    display(p)
    savefig(p, "plot/deltaML/MAE_PCAdn_FCHL19_best.png")

    # plot prototype for vs{features} in the same environment:
    tb1 = tbsel[29:32, :] #ACSF
    tb2 = tbsel[61:64, :] #SOAP
    tb3 = tbsel[93:96, :] #FCHL19
    MAEs = vcat(tb1[:, 7], tb2[:, 7], tb3[:, 7])
    xticks = tb1[:, 1]
    xtformat = string.(map(x -> @sprintf("%.0f",x), xticks))
    yticks = round.(vcat(maximum(MAEs), minimum(MAEs)), digits=3)
    yticks = vcat(yticks, range(10, 50, 5))
    ytformat = vcat(string.(yticks[1:2]), map(x -> @sprintf("%.0f",x), yticks[3:end]))
    p = plot(tb1[:, 1], [tb1[:, 7], tb2[:, 7], tb3[:, 7]],
        yticks = (yticks, ytformat), xticks = (xticks, xtformat),
        xaxis = :log, yaxis = :log,
        markershape = [:xcross :cross :rect], markersize = [6 6 4],
        labels = ["MAE(ACSF)" "MAE(SOAP)" "MAE(FCHL19)"], xlabel = "Ntrain", ylabel = "MAE (kcal/mol)")
    display(p)
    #savefig(p, "plot/deltaML/MAE_features_vs.png")
end

function plot_MAE_dt()
    tbns = readdlm("result/deltaML/MAE_enum_ns_dt_180823.txt")
    tbs = readdlm("result/deltaML/MAE_enum_s2_dt_180823.txt")
    tb = tbs # select which table
    # see effect of dressed on each feature:
    d_minind = Dict("acsf" => query_min_f(tb, feature_type = "ACSF_51"), 
                "soap" => query_min_f(tb, feature_type = "SOAP"), 
                "fchl19" => query_min_f(tb, feature_type = "FCHL19"))
    d_colq = Dict("acsf" =>tb[d_minind["acsf"],[2,3,4]], 
                "soap" =>tb[d_minind["soap"],[2,3,4]], 
                "fchl19"=>tb[d_minind["fchl19"],[2,3,4]])
    d_ind = Dict("acsf"=>query_indices(tb, [2,3,4], d_colq["acsf"]),
                "soap"=>query_indices(tb, [2,3,4], d_colq["soap"]), 
                "fchl19"=>query_indices(tb, [2,3,4], d_colq["fchl19"]))


    # 1) fix the best hyperparameter from ns and sx data then compare up to dt (see the effect of baselines) (2 plots):
    tables = [tbns, tbs] # noselect and select
    tbnames = ["ns", "s2"]
    for (i,tb) ∈ enumerate(tables)
        minid = query_min_f(tb)
        qcol = tb[minid, [2,3,4]]
        ind = query_indices(tb, [2,3,4], qcol)
        MAEs = tb[ind, end]
        xticks = tb[:, 1][1:4]
        xtformat = string.(map(x -> @sprintf("%.0f",x), xticks))
        y = tb[ind, 7]
        yticks = yticks_generator(y, 5)
        ytformat = vcat(string(round(yticks[1], digits=3)), map(x -> @sprintf("%.0f",x), yticks[2:end-1]), string(round(yticks[end], digits=3)))
        p = plot(xticks, [tb[ind, :][1:4, end], tb[ind, :][5:8, end], tb[ind, :][9:12, end], tb[ind, :][end-3:end, end]],
            yticks = (yticks, ytformat), xticks = (xticks, xtformat),
            xaxis = :log, yaxis = :log,
            markershape = [:circle :rect :diamond :utriangle], markersize = (ones(5)*6)',
            labels = ["MAE(da)" "MAE(db)" "MAE(dn)" "MAE(dt)"], xlabel = "Ntrain", ylabel = "MAE (kcal/mol)")
        display(p)
        println(join(tb[ind, [2,3,4]][1,:], "-"))
        savefig(p, "plot/deltaML/MAE_bh-"*join(tb[ind, [2,3,4]][1,:], "-")*"_"*tbnames[i]*"_upto-dt.png")
        writedlm("plot/deltaML/MAE_bh-"*join(tb[ind, [2,3,4]][1,:], "-")*"_"*tbnames[i]*"_upto-dt.txt", [tb[ind, :][1:4, :]; tb[ind, :][5:8, :]; tb[ind, :][9:12, :]; tb[ind, :][end-3:end, :]]) # store the plot information
    end

    # 2) - get the best from ns and s then fix hyperparam then plot for each ns and s,
    #   - get each best of ns and s
    # (1 plot, 4 curves total, see the efefect of data selection) 
    # fix the hyperparameters in which the best from both ns and s:
    jointb = vcat(tbns, tbs)
    minid = query_min_f(jointb)
    qcol = jointb[minid, [2,3,4,5]]
    id_ns = query_indices(tbns, [2,3,4,5], qcol)
    id_s = query_indices(tbs, [2,3,4,5], qcol)
    # best of ns mode:
    minid = query_min_f(tbns)
    id_bns = query_indices(tbns, [2,3,4,5], tbns[minid, [2,3,4,5]])
    jointb = vcat(tbns[id_ns, :], tbns[id_bns, :], tbs[id_s, :])

    xticks = jointb[:, 1][1:4]; xtformat = string.(map(x -> @sprintf("%.0f",x), xticks))
    MAEs = jointb[:, end]; minMAE = minimum(MAEs); maxMAE = maximum(MAEs);
    yticks = range(minMAE, maxMAE, 7)
    y = sort(vcat(tbns[id_ns, 7], tbns[id_bns, 7], tbs[id_s, 7]))
    yticks = yticks_generator(y, 5)
    ytformat = vcat(string(round(yticks[1], digits=3)), map(x -> @sprintf("%.0f",x), yticks[2:end-1]), string(round(yticks[end], digits=3)))
    p = plot(xticks, [tbns[id_ns, 7], tbns[id_bns, 7], tbs[id_s, 7]],
        yticks = (yticks, ytformat), xticks = (xticks, xtformat),
        xaxis = :log, yaxis = :log,
        markershape = [:circle :rect :diamond :utriangle], markersize = (ones(5)*6)',
        labels = ["MAE(ns)" "MAE(bons)" "MAE(bos)"], xlabel = "Ntrain", ylabel = "MAE (kcal/mol)")
    display(p)
    savefig(p, "plot/deltaML/MAE_sel_effect.png")
    writedlm("plot/deltaML/MAE_sel_effect.txt", [tbns[id_ns, :]; tbns[id_bns, :]; tbs[id_s, :]])

    # 3) fix the best MAE on each feature (1 plot w/ 3 curves, see the effect of feature)
    ftypes = ["ACSF_51", "SOAP", "FCHL19"]
    jtb = vcat(tbns, tbs)
    nrow = size(jtb, 1); halfrow = (nrow ÷ 2);
    tbslices = [] # reference to slices of tables
    # find min location (in which location of table):
    minids = []
    for (i, ftype) ∈ enumerate(ftypes)
        minid = query_min_f(jtb; feature_type = ftype)
        push!(minids, minid)
        println([ftype, minid, halfrow])
        if minid > halfrow # get from s2 table:
            minid = minid - halfrow
            qid = query_indices(tbs, [2,3,4,5], tbs[minid, [2,3,4,5]])
            push!(tbslices, tbs[qid, :]) 
        else
            qid = query_indices(tbns, [2,3,4,5], tbns[minid, [2,3,4,5]])
            push!(tbslices, tbns[qid, :])
        end
    end
    xticks = tbs[1:4, 1]; xtformat = string.(map(x -> @sprintf("%.0f",x), xticks))
    jtb = reduce(vcat, tbslices)
    MAEs = jtb[:, end]
    yticks = yticks_generator(MAEs, 5); ytformat = vcat(string(round(yticks[1], digits=3)), map(x -> @sprintf("%.0f",x), yticks[2:end-1]), string(round(yticks[end], digits=3)))
    p = plot(xticks, [tbslices[1][:, end], tbslices[2][:, end], tbslices[3][:, end]],
            yticks = (yticks, ytformat), xticks = (xticks, xtformat),
            xaxis = :log, yaxis = :log,
            markershape = [:circle :rect :diamond :utriangle], markersize = (ones(5)*6)',
            labels = ["MAE(acsf)" "MAE(soap)" "MAE(fchl19)"], xlabel = "Ntrain", ylabel = "MAE (kcal/mol)"
        )
    display(p)
    savefig(p, "plot/deltaML/MAE_best_each-feature.png")
    writedlm("plot/deltaML/MAE_best_each-feature.txt", jtb)
end

function plot_subsec_33()
    #4) for PCA subsection, 1 plot 4 curves, each curve is the best of each elvl on s2 data (see how far PCA can improve things)
    tb1 = readdlm("result/deltaML/MAE_enum_s2_dt_180823.txt") # wait for the computation
    tb2 = readdlm("result/deltaML/MAE_enum_s2_dt_PCAjl-dn5-dt5-280823.txt")
    # best without PCA:
    minid = query_min(tb1, [], [])
    id_minnopca = query_indices(tb1, [2,3,4,5], tb1[minid, [2,3,4,5]])
    display(tb1[id_minnopca, :])
    # best of DN without PCA:
    minid = query_min(tb1, [5], ["dressed_angle"])
    id_mindn = query_indices(tb1, [2,3,4,5], tb1[minid, [2,3,4,5]])
    display(tb1[id_mindn, :])
    # best of DT without PCA:
    minid = query_min(tb1, [5], ["dressed_torsion"])
    id_mindt = query_indices(tb1, [2,3,4,5], tb1[minid, [2,3,4,5]])
    display(tb1[id_mindt, :])
    # best of DN with PCA:
    minid = query_min(tb2, [5], ["dressed_angle"])
    id_mindn_pca = query_indices(tb2, [2,3,4,5], tb2[minid, [2,3,4,5]])
    display(tb2[id_mindn_pca, :])
    # best of DT with PCA:
    minid = query_min(tb2, [5], ["dressed_torsion"])
    id_mindt_pca = query_indices(tb2, [2,3,4,5], tb2[minid, [2,3,4,5]])
    display(tb2[id_mindt_pca, :])
    # get all selected MAEs ....:
    MAEs = vcat(tb1[id_minnopca, end], tb1[id_mindn, end], tb1[id_mindt, end], tb2[id_mindn_pca, end], tb2[id_mindt_pca, end])
    # generate the ticks:
    xticks = tb1[1:4, 1]; xtformat = string.(map(x -> @sprintf("%.0f",x), xticks))
    yticks = yticks_generator(MAEs, 7); ytformat = vcat(string(round(yticks[1], digits=3)), map(x -> @sprintf("%.0f",x), yticks[2:end-1]), string(round(yticks[end], digits=3)))
    # plot:
    p = plot(xticks, [tb1[id_minnopca, end], tb1[id_mindn, end], tb1[id_mindt, end], tb2[id_mindn_pca, end], tb2[id_mindt_pca, end]],
            yticks = (yticks, ytformat), xticks = (xticks, xtformat),
            xaxis = :log, yaxis = :log,
            markershape = [:circle :rect :diamond :utriangle :pentagon], markersize = (ones(5)*6)',
            labels = ["MAE(bo,db)" "MAE(bo,dn)" "MAE(bo,dt)" "MAE(bo,dn,PCA1)" "MAE(bo,dt,PCA1)"], xlabel = "Ntrain", ylabel = "MAE (kcal/mol)"
        )
    display(p)
    savefig(p, "plot/deltaML/MAE_PCA1_comparison.png")
    writedlm("plot/deltaML/MAE_PCA1_comparison.txt", [tb1[id_minnopca, :]; tb1[id_mindn, :]; tb1[id_mindt, :]; tb2[id_mindn_pca, :]; tb2[id_mindt_pca, :]])

    # 5) (6 curves) BODB (from above), PCA1BODN (from above), PCA1BODT (from above), PCA2BODB, PCA2BODN, PCA2BODT:
    tb3 = readdlm("result/deltaML/MAE_enum_s2_dt_PCAjl-db5-dn5-dt5-300823.txt")
    # PCA2 BODB:
    minid = query_min(tb3, [5], ["dressed_bond"])
    id_mindb2 = query_indices(tb3, [2,3,4,5], tb3[minid, [2,3,4,5]])
    display(tb3[id_mindb2, :])
    # PCA2 BODN:
    minid = query_min(tb3, [5], ["dressed_angle"])
    id_mindn2 = query_indices(tb3, [2,3,4,5], tb3[minid, [2,3,4,5]])
    display(tb3[id_mindn2, :])
    # PCA2 BODT:
    minid = query_min(tb3, [5], ["dressed_torsion"])
    id_mindt2 = query_indices(tb3, [2,3,4,5], tb3[minid, [2,3,4,5]])
    display(tb3[id_mindt2, :])
    MAEs = vcat(tb1[id_minnopca, end], tb2[id_mindn_pca, end], tb2[id_mindt_pca, end], tb3[id_mindb2, end], tb3[id_mindn2, end], tb3[id_mindt2, end])
    yticks = yticks_generator(MAEs, 7); ytformat = vcat(string(round(yticks[1], digits=3)), map(x -> @sprintf("%.0f",x), yticks[2:end-1]), string(round(yticks[end], digits=3)))
    # plot:
    p = plot(xticks, [tb1[id_minnopca, end], tb2[id_mindn_pca, end], tb2[id_mindt_pca, end], tb3[id_mindb2, end], tb3[id_mindn2, end], tb3[id_mindt2, end]],
            yticks = (yticks, ytformat), xticks = (xticks, xtformat),
            xaxis = :log, yaxis = :log,
            markershape = [:circle :rect :diamond :utriangle :pentagon :star], markersize = (ones(5)*6)',
            labels = ["MAE(bo,db)" "MAE(bo,dn,PCA1)" "MAE(bo,dt,PCA1)" "MAE(bo,db,PCA2)" "MAE(bo,dn,PCA2)" "MAE(bo,dt,PCA2)"], 
            linestyles = [:solid :solid :solid :dash :dash :dash],
            xlabel = "Ntrain", ylabel = "MAE (kcal/mol)"
        )
    display(p)
    savefig(p, "plot/deltaML/MAE_PCA_1_2_comparison.png")
    writedlm("plot/deltaML/MAE_PCA_1_2_comparison.txt",[tb1[id_minnopca, :], tb2[id_mindn_pca, :], tb2[id_mindt_pca, :], tb3[id_mindb2, :], tb3[id_mindn2, :], tb3[id_mindt2, :]])

    # 4+5) Table of best of non PCA, best of PCA1 for each level, best of PCA2 for each level:
    # best of DN and DT non PCA:
    minid = query_min(tb1, [5], ["dressed_angle"])
    jtb = [reshape(tb1[id_minnopca[end], :], 1, :); reshape(tb1[id_mindn[end], :], 1, :); reshape(tb1[id_mindt[end], :], 1, :);
            reshape(tb2[id_mindn_pca[end], :], 1, :); reshape(tb2[id_mindt_pca[end], :], 1, :); 
            reshape(tb3[id_mindb2[end], :], 1, :); reshape(tb3[id_mindn2[end], :], 1, :); reshape(tb3[id_mindt2[end], :], 1, :)] 
    jtb = hcat(jtb, ["none", "none", "none", "PCA1", "PCA1", "PCA2", "PCA2", "PCA2"])
    # process (4+5) table:
    # move last column to first:
    jtb[:, 1] = jtb[:, end]
    jtb = jtb[:, 1:end-1]
    jtb[:, 5] = ["DB", "DN", "DT", "DN", "DT", "DB", "DN", "DT"]
    for i ∈ axes(jtb, 1)
        # change featurenames:
        if jtb[i, 2] == "ACSF_51"
            jtb[i, 2] = "ACSF"
        end
        # change model names:
        if jtb[i, 3] == "REAPER"
            jtb[i, 3] = "DPK"
        elseif jtb[i, 3] == "GAK"
            jtb[i, 3] = "GK"
        end
    end
    jtb = jtb[:, vcat(1:end-2, end)]
    display(jtb)
    jtb[:, end] = clean_float(jtb[:, end])
    writelatextable(jtb, "plot/deltaML/tb_PCA_comparison.tex")
    
end