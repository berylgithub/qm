using Plots, DelimitedFiles, LaTeXStrings


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

function plot_MAE_db()
    tb = readdlm("result/deltaML/MAE_enum.txt")
    tbsel = readdlm("result/deltaML/MAE_enum_set-2.txt")
    tb_da = tb[13:16, :]
    tb_db = tb[29:32, :]
    tbsel_db = tbsel[29:32, :]
    yticks = round.(vcat(tb_da[1, 7], tbsel_db[end, 7]), digits=3)
    yticks = vcat(yticks, range(10, 50, 5))
    ytformat = string.(yticks)
    display(ytformat)
    plot(tb_da[:, 1], [tb_da[:, 7], tb_db[:, 7], tbsel_db[:, 7]],
        yticks = (yticks, , xticks = tb_da[:, 1],
        markershape = [:xcross :cross :rect], 
        labels = ["MAE(da)" "MAE(db)" "MAE(db,sel)"], xlabel = "Ntrain", ylabel = "MAE (kcal/mol)")
end