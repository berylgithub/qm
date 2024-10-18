using JLD, Plots, LaTeXStrings

d = load("data/smallmol/hxoy_data.jld", "data")
dx = d[10]["R"]
dy = d[10]["V"]
p = Plots.plot(dx, dy, xlim=[0,6], ylim=[-.5,3], legend=false, linewidth=3, linecolor=:green, 
            title="\$H_2\$", xlabel="\$r\$", ylabel = "\$V\$", dpi=100)
scatter!(dx, dy, markersize=3)
savefig(p, "slides/assets/plots/plot_H2.svg")