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

function plot_subsec_PCA()
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


function main_write_full_tbs()
    tb1 = readdlm("result/deltaML/MAE_enum_ns_dt_180823.txt")
    tb2 = readdlm("result/deltaML/MAE_enum_s2_dt_180823.txt")
    # query DN and DT from each:
    id_tb1_dn = query_indices(tb1, [5], ["dressed_angle"])
    id_tb1_dt = query_indices(tb1, [5], ["dressed_torsion"])
    id_tb2_dn = query_indices(tb2, [5], ["dressed_angle"])
    id_tb2_dt = query_indices(tb2, [5], ["dressed_torsion"])
    jtb1 = [tb1[id_tb1_dn, :]; tb1[id_tb1_dt, :]] 
    jtb2 = [tb2[id_tb2_dn, :]; tb2[id_tb2_dt, :]]
    jtbs = [jtb1, jtb2]
    for jtb ∈ jtbs # should just make a mandatory function for these guys
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
            # change elevel names
            if jtb[i, 5] == "dressed_angle"
                jtb[i, 5] = "DN"
            elseif jtb[i, 5] == "dressed_torsion"
                jtb[i, 5] = "DT"
            end
        end
        jtb[:, [end-1, end]] = clean_float(jtb[:, [end-1, end]])
    end
    writelatextable(jtbs[1], "result/deltaML/tb_ns_DNDT.tex")
    writelatextable(jtbs[2], "result/deltaML/tb_s2_DNDT.tex")
end

function main_tb_hda()
    tbbase1 = readdlm("result/deltaML/MAE_base_s2_dt_180823.txt")
    tb1 = readdlm("result/deltaML/MAE_enum_s2_dt_180823.txt")
    tbbase2 = readdlm("result/deltaML/MAE_base_s2_hda_dt_120923.txt")
    tb2 = readdlm("result/deltaML/MAE_enum_s2_hda_dt_120923.txt")
    
    # base MAE table
    tbb1x = replace(tbbase1, "dressed_atom" => "DA", "dressed_bond" => "DB", 
                    "dressed_angle" => "DN", "dressed_torsion" => "DT")
    tbb2x = replace(tbbase2, "dressed_atom" => "DA", "dressed_bond" => "DB", 
                    "dressed_angle" => "DN", "dressed_torsion" => "DT")
    jtb = hcat(tbb1x, tbb2x[:, end-1:end])
    jtb[2:end, 2:end] = clean_float(jtb[2:end, 2:end])
    writelatextable(jtb, "result/deltaML/tb_base_vs_hda.tex", hline=false)
    # enum table comparison (DA vs HDA best on DA level, global minimum DA vs global minimum HDA):
    id1da = query_min(tb1, [5], ["dressed_atom"]) # best of DA
    id2da = query_min(tb2, [5], ["dressed_atom"])
    id1min = query_min(tb1, [], [])
    id2min = query_min(tb2, [], [])
    jtb = vcat(reshape(tb1[id1da, :], 1, :), reshape(tb2[id2da, :], 1, :), reshape(tb1[id1min, :], 1, :), reshape(tb2[id2min, :], 1, :))
    jtb = hcat(jtb, ["standard", "hybridized", "standard", "hybridized"])
    writedlm("result/deltaML/tb_enum_da-vs-hda.txt", jtb)
    jtb_tex = Matrix{Any}(undef, 3, 3)
    jtb_tex[1, :] = ["query", "SDA MAE test", "HDA MAE test"]
    jtb_tex[2, :] = vcat("best on DA", jtb[1:2, end-1])
    jtb_tex[3, :] = vcat("best from all", jtb[3:4, end-1])
    jtb_tex[2:end, 2:end] = clean_float(jtb_tex[2:end, 2:end])
    writelatextable(jtb_tex, "result/deltaML/tb_enum_da-vs-hda.tex"; hline=false)
end

function MAE_enum_v2_plot()
    tb = readdlm("result/deltaML/MAE_enum_v2_30k_100k_H_280923.txt") # assume the default includes Hydrogens
    display(tb)
    minid = query_min(tb, [], [], 9) # 9 is the index of the test MAE
    # query b_MAEs_* from one table:
    b_MAEs = [] # will be a vector of vectors
    bs = ["A", "AB", "ABN", "ABNT"]
    for (i,b) ∈ enumerate(bs)
        qid = query_indices(tb, [3, 4, 5], [b, "GK", "ACSF_51"])
        push!(b_MAEs, tb[qid, 7])
    end
    xticks = tb[1:8, 1]; xtformat = string.(map(x -> @sprintf("%.0f",x), xticks))
    yticks = yticks_generator(reduce(vcat, b_MAEs), 7) ∪ [12, 16];
    ytformat = map(x -> @sprintf("%.0f",x), yticks)
    #ytformat = vcat(string(round(yticks[1], digits=3)), map(x -> @sprintf("%.0f",x), yticks[2:end-1]), string(round(yticks[end], digits=3)))
    p = plot(xticks, b_MAEs,
            yticks = (yticks, ytformat), xticks = (xticks, xtformat),
            xaxis = :log, yaxis = :log,
            markershape = [:circle :rect :diamond :utriangle], markersize = (ones(5)*6)',
            labels = ["A" "AB" "ABN" "ABNT"], xlabel = "Ntrain", ylabel = "MAE (kcal/mol)"
        )
    display(p)
    # sample w/ for (elv, solver) = ([A, AB, ABN, ABNT], GK):
    MAEs = []
    for (i,b) ∈ enumerate(bs)
        qid = query_indices(tb, [3, 4, 5], [b, "GK", "FCHL19"])
        push!(MAEs, tb[qid, 9])
    end
    qids_A = query_indices(tb, [3, 4,5], ["A", "GK", "FCHL19"])
    qids_AB = query_indices(tb, [3, 4,5], ["AB", "GK", "FCHL19"])
    # e.g., on FCHL19 space:
    ys = [tb[qids_A, 9], tb[qids_AB, 9]]
    yticks = [2.0^i for i ∈ 0:4]
    ytformat = map(x -> @sprintf("%.0f",x), yticks)
    p = plot(xticks, MAEs,
            yticks = (yticks, ytformat), xticks = (xticks, xtformat),
            xaxis = :log, yaxis = :log,
            markershape = [:circle :rect :diamond :utriangle], markersize = (ones(5)*6)',
            labels = ["A" "AB" "ABN" "ABNT"], xlabel = "Ntrain", ylabel = "MAE (kcal/mol)"
        )
    hline!([1], labels = nothing)
    display(p)
    # look at MBDF acc to determine whether it's feasible for dataselection:
    tb2 = readdlm("result/deltaML/MAE_enum_v2_MBDF_30k_100k_H_280923-2.txt")
    minid2 = query_min(tb2, [], [], 9)
    display(tb2[minid2,9])
    qids_A = query_indices(tb, [3,4,5], ["A", "GK", "ACSF_51"])
    qids_AB = query_indices(tb, [3,4,5], ["AB", "GK", "ACSF_51"])
    qids1_A = query_indices(tb, [3,4,5], ["A", "GK", "FCHL19"])
    qids1_AB = query_indices(tb, [3,4,5], ["AB", "GK", "FCHL19"])
    qids2_A = query_indices(tb2, [3,4,5], ["A", "GK", "MBDF"])
    qids2_AB = query_indices(tb2, [3,4,5], ["AB", "GK", "MBDF"])
    ys = [tb[qids_A, 9], tb[qids_AB, 9], tb[qids1_A, 9], tb[qids1_AB, 9], tb2[qids2_A, 9], tb2[qids2_AB, 9]]
    p = plot(xticks, ys,
            yticks = (yticks, ytformat), xticks = (xticks, xtformat),
            xaxis = :log, yaxis = :log,
            markershape = [:circle :rect :diamond :utriangle], markersize = (ones(5)*6)',
            labels = ["(ACSF, A)" "(ACSF, AB)" "(FCHL19, A)" "(FCHL19, AB)" "(MBDF, A)" "(MBDF, AB)"], xlabel = "Ntrain", ylabel = "MAE (kcal/mol)"
        )
    display(p)
    qid100 = [] 
    display(tb2[qids2_A,:])
end

function test_bond_energy_plot()
    # ZiZkr^-c:
    function pairpot(Z1, Z2, r, c)
        return Z1*Z2/r^c
    end
    Zs = Float64.([1,6,7,8,9]) # [H, C, N, O, F]
    rs = range(0., 5., 1000)
    cs = Float64.([1, 6])
    
    x = rs
    y = pairpot.(Zs[2], Zs[4], rs, cs[2])
    display(y)
    plot(x, y, ylims = [0, 100])

    # morse:
    function morse_pot(r, D, a, r0, s; mode=1)
        if mode==1
            return D*(exp(-2*a*(r-r0)) - 2*exp(-a*(r-r0))) + s # additional shift constant s for "pure" fitting
        elseif mode == 2
            return D*(1-exp(-a*(r-r0)))^2
        end
    end
    D = .5
    a = 2.
    r0 = 1.
    s = 0.1
    x = rs
    y = morse_pot.(rs, D, a, r0, s,mode=1)
    display(y)
    plot(x, y, ylims = [-1., 1.])
end


"""
combine rand useq table by adding new column (H and HDA too) 
"""
function main_combine_tbs()
    headers = ["ntrain", "ntest", "elv", "model", "feature", 
                "b_MAEtrain", "b_MAEtest", "MAEtrain", "MAEtest", 
                "t_mcom", "t_etrain", "t_epred", "t_mtrain", "t_mtest"]
    headers_add = ["select", "hydrogen","hybrid"] # additional columns to be combined
    headers = vcat(headers_add, headers)
    display(headers)
    # combine tables:
    tb1 = readdlm("result/deltaML/MAE_enum_v2_30k_100k_srand_H_101123.txt")
    tb2 = readdlm("result/deltaML/MAE_enum_v2_30k_100k_sid57_H_101123.txt")
    tb3 = readdlm("result/deltaML/MAE_enum_v2_30k_100k_sid57_H_HDA_101123.txt") # later when the computation is finished
    tbj = Matrix{Any}(undef, sum(size.([tb1, tb2, tb3], 1))+1, size(tb1, 2)+length(headers_add))
    tbj[1,:] = headers
    id_tbs = [(2,1+size(tb1,1)), (2+size(tb1,1), size(tb1,1)+size(tb2,1)+1), (size(tb1,1)+size(tb2,1)+2, size(tb1,1)+size(tb2,1)+size(tb3,1)+1)]
    display(id_tbs)
    tbj[id_tbs[1][1]:id_tbs[1][2],1:3] .= ["rand" "true" "false"]; tbj[id_tbs[1][1]:id_tbs[1][2],4:end] = tb1
    tbj[id_tbs[2][1]:id_tbs[2][2],1:3] .= ["sid57" "true" "false"]; tbj[id_tbs[2][1]:id_tbs[2][2],4:end] = tb2
    tbj[id_tbs[3][1]:id_tbs[3][2],1:3] .= ["sid57" "true" "true"]; tbj[id_tbs[3][1]:id_tbs[3][2],4:end] = tb3
    writedlm("result/deltaML/MAE_enum_v2_combined_101123.txt", tbj)
end


"""
several plotting scenarios:
- plot of the minimum of each ntrain separated by each feature -> 6 curves
- plot of rand vs usequence, sample ACSF, FCHL19, and CMBDF (6 curves, 2 each representation)
- plot ΔML, sample ACSF, FCHL19, and CMBDF -> (3 plots * 4 curves, reduce if too crowded)
"""
function main_plot_v2()
    tb = readdlm("result/deltaML/MAE_enum_v2_combined_101123.txt")
    headers = tb[1,:] # headers
    println(collect(enumerate(headers)))
    tb = tb[2:end, :] # extract values
    id_gmin = query_min(tb, [], [], 12) # global minima 
    
    # plot of minimum on each ntrain on each feature:
    ntrains = tb[1:9,4]
    fts = unique(tb[:,8])
    ids_mins = []
    for ft ∈ fts
        temp = []
        for ntrain ∈ ntrains
            id_min = query_min(tb, [8, 4], [ft, ntrain], 12)
            push!(temp, id_min)
        end
        push!(ids_mins, temp)
    end
    xticks = ntrains; xtformat = string.(map(x -> @sprintf("%.0f",x), xticks))
    ys = [tb[ids_mins[i],12] for i ∈ eachindex(fts)]
    yticks = [2.0^i for i ∈ 0:3]; ytformat = map(x -> @sprintf("%.0f",x), yticks)
    p = plot(xticks, ys,
            yticks = (yticks, ytformat),
            xticks = (xticks, xtformat), 
            xaxis = :log, yaxis = :log,
            markershape = [:circle :rect :utriangle :diamond :hexagon], markersize = (ones(length(fts))*6)',
            labels = "min(" .* permutedims(fts) .*")", xlabel = "Ntrain", ylabel = "MAE (kcal/mol)",
            title = "minimum MAE for each representation",
            dpi = 1000
        )
    hline!([1], labels = nothing, lc = :red)
    display(p)
    savefig(p, "plot/deltaML/fig_min_MAE_representation.png")
    tbplot = vcat(permutedims(headers),tb[reduce(vcat, ids_mins),:])
    display(tbplot)
    writedlm("plot/deltaML/tb_min_MAE_representation.txt", tbplot)

    # plot min(rand) vs min(usequence), sample (ACSF, FCHL19, CMBDF)
    fts = ["ACSF_51", "FCHL19", "SOAP", "CMBDF"]
    selections = unique(tb[:,1])
    iters = collect(Iterators.product(ntrains, fts, selections)) # iterators, so that the loop is single
    id_mins = []
    for it ∈ iters
        push!(id_mins, query_min(tb, [4, 8, 1], it, 12))
    end
    niters = reduce(*, length.([fts,selections]))
    nslice = Int(length(id_mins)/niters)
    ncurves = Int(length(id_mins)/nslice) # number of curves
    id_slices = [((i-1)*nslice + 1, i*nslice) for i ∈ 1:ncurves] # which slices to pick from the id_mins
    ys = [tb[id_mins[id[1]:id[2]], 12] for id ∈ id_slices] # slice the MAEs (y axis)
    yticks = [2.0^i for i ∈ 0:5]; ytformat = map(x -> @sprintf("%.0f",x), yticks)
    labels = permutedims([tb[id_mins[id[1]],1] .* ", " .* tb[id_mins[id[1]],8] for id ∈ id_slices])
    p = plot(xticks, ys,
            yticks = (yticks, ytformat),
            xticks = (xticks, xtformat), 
            xaxis = :log, yaxis = :log,
            linestyle = [:dash :dash :dash :dash :solid :solid :solid :solid],
            markershape = [:circle :rect :utriangle :diamond :circle :rect :utriangle :diamond], 
            markercolor = [:black :green :blue :purple :black :green :blue :purple],
            linecolor = [:black :green :blue :purple :black :green :blue :purple],
            markersize = (ones(ncurves)*6)',
            labels = labels, xlabel = "Ntrain", ylabel = "MAE (kcal/mol)",
            title = "Random vs Usequence",
            dpi = 1000
        )
    hline!([1], labels = nothing, lc = :red)
    display(p)
    savefig(p, "plot/deltaML/fig_RandVFPS.png")
    tbplot = vcat(permutedims(headers),tb[reduce(vcat, id_mins),:])
    writedlm("plot/deltaML/tb_RandVFPS.txt", tbplot)

    # plot of the baseline MAE only (4 elvs * 2 selections + 4 sid_57 hybrid = 12 curves)
    # ! looks like this pattern of plotting is repeating, should probably try to write this in 
    elvs = ["A", "AB", "ABN", "ABNT"]
    selections = unique(tb[2:end, 1])
    hybrids = [false, true]
    iters = collect(Iterators.product(ntrains, elvs, selections, hybrids))
    ids_plot = []
    for iter ∈ iters
        ids = query_indices(tb, [4,6,1,3], iter) # only need one data point
        if !isempty(ids) # some of the combination may not exist
            push!(ids_plot, ids[1])
        end
    end
    display(tb[ids_plot,[4,6,1,3]])
    nslice = length(ntrains) # number of x data points each curve
    ncurves = Int(length(ids_plot)/length(ntrains))
    id_slices = [((i-1)*nslice + 1, i*nslice) for i ∈ 1:ncurves]
    ys = [tb[ids_plot[id[1]:id[2]], 10] for id ∈ id_slices] # slice the MAEs (y axis)
    yticks = vcat([2.0^i for i ∈ 0:5], [10., 12., 20.]); ytformat = map(x -> @sprintf("%.0f",x), yticks)
    labels = permutedims([tb[ids_plot[id[1]],6] .* ", " .* tb[ids_plot[id[1]],1] .* ", " .* string(tb[ids_plot[id[1]],3]) for id ∈ id_slices])
    p = plot(xticks, ys,
            yticks = (yticks, ytformat),
            xticks = (xticks, xtformat), 
            xaxis = :log, yaxis = :log,
            linestyle = permutedims(vcat(repeat([:dash],4), repeat([:solid],8))),
            markershape = hcat(permutedims(hcat(repeat([:circle, :rect, :utriangle, :diamond], 2))), [:+ :x :hexagon :heptagon] ), 
            markercolor = permutedims(hcat(repeat([:black, :green, :blue, :purple],3))),
            linecolor = permutedims(hcat(repeat([:black, :green, :blue, :purple],3))),
            markersize = (ones(ncurves)*4)',
            labels = labels, xlabel = "Ntrain", ylabel = "MAE (kcal/mol)",
            title = "Baseline MAEs",
            legendfontsize = 7, legend = :outertopright,
            dpi=1000
        )
    display(p)
    savefig(p, "plot/deltaML/fig_base.png")
    tbplot = vcat(permutedims(headers),tb[ids_plot,:])
    writedlm("plot/deltaML/tb_base.txt", tbplot)
    
end

function main_get_stats_deltaML()
    # get the hit counts of: [elvs, sid57, hybrid]
    tb = readdlm("result/deltaML/MAE_enum_v2_combined_101123.txt")
    headers = tb[1,:]
    tb = tb[2:end,:]
    println([(k,v) for (k,v) ∈ enumerate(headers)])
    hit_counts = Dict()
    hit_counts 
    #...
end

function main_display_selected_mol()
    trainset = vec(Int.(readdlm("data/tsopt/opt_tracker_freeze.txt")[2:end]))
    tf = map(d->d["formula"],dataset[trainset])
    # generate the string matrix:
    ## subscript all numbers in the formula:
    for (i,data) in enumerate(tf)
        temp = ""
        for chr in data
            if isdigit(chr)
                temp*="\$_"*chr*"\$"
            else
                temp*=chr
            end
        end 
        tf[i] = temp
    end
    display(tf)
    ## write the training set into 10x10 matrix 
    A = Matrix{Any}(undef, 25, 4)
    for i ∈ eachindex(A)
        A[i] = tf[i]
    end
    writelatextable(A, "result/table_selmol.tex"; hline=false)
end