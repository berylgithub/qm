using Flux, Plots

function test_NN()
    # generate data:
    A = hcat(collect(Float32, -3:0.1:3)...)
    b = @. 3A + 2;
    A = A .* reshape(rand(Float32, size(A, 2)), size(A));

    # model:
    model = Dense(1 => 1)
    pars = Flux.params(model)
    opt = Flux.Adam(0.01)

    plot(vec(A), vec(b), lw = 3, seriestype = :scatter, label = "", title = "Generated data", xlabel = "x", ylabel= "y");
    
    # transform data to flux container:
    loader = Flux.DataLoader((A, b))

    # opt:
    losses = []
    @showprogress for epoch in 1:1_000
        for (x, y) in loader
            loss, grad = Flux.withgradient(pars) do
                # Evaluate model and loss inside gradient context:
                y_hat = model(x)
                Flux.mse(y_hat, y)
            end
            Flux.update!(opt, pars, grad)
            push!(losses, loss)  # logging, outside gradient context
        end
    end
    b_hat = model(A)
    display(b_hat)
    display(plot!(vec(A), vec(b_hat), lw = 3, seriestype = :scatter, markercolor = "red"))
    display(losses)
end