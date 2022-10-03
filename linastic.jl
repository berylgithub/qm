using LinearAlgebra

"""
placeholder for linear algebra and statistics operations, if RoSemi is overcrowded, or when the need arises
"""

"""
the Kronecker delta function
"""
δ(x,y) = ==(x,y)

"""
computes residual from data matrix A, coefficient vector θ, and target vector b
"""
function residual(A, θ, b)
    return A*θ - b
end

"""
return the Least squares form (norm^2) of the residual r := Aθ - b
"""
function lsq(A, θ, b)
    return norm(residual(A, θ, b))^2
end

"""
compute mean vector and covariance matrix
params:
    - idx, the index of the wanted molecule (w0)
    - N, number of data
    - len_finger, vector length of the fingerprint
output:
    - wbar, mean, size = len_finger
    - C, covariance, size = (len_finger, len_finger)
"""
function mean_cov(w_matrix, idx, N, len_finger)
    # initialize intermediate vars:
    S = zeros(len_finger,len_finger)
    dw = zeros(len_finger)
    dif = zeros(len_finger, N)

    loopset = union(1:idx-1, idx+1:N) # loop index
    # dw := 
    for i ∈ loopset # all except the w0 index itself, since w0 - w0 = 0
        dif[:,i] = w_matrix[:, i] .- w_matrix[:, idx] #wv- w0
        dw .+= dif[:, i]
    end
    dw ./= N
    # S := 
    for i ∈ loopset
        S .+= dif[:,i]*dif[:,i]'
    end

    #= display(w_matrix[:, idx])
    display(dif)
    display(dw)
    display(S)
    display(dw*dw') =#

    return w_matrix[:, idx] .+ dw, (S .- (dw*dw'))./(N-1) # mean, covariance
end


"""
compute the B linear transformer for Mahalanobis distance
params:
    - C, covariance matrix of the fingerprints
outputs:
    - B, the linear transformer for Mahalanobis distance, used for: ||B(w - wk)||₂²
"""
function compute_B(C)
    # eigendecomposition:
    e = eigen(C)
    #display(e)
    # round near zeros to zero, numerical stability:
    v = e.values
    bounds = [-1e-8, 1e-8]
    b = bounds[1] .< v .< bounds[2]
    v[b] .= 0.
    v = v.^(-.5) # take the "inverse sqrt" of the eigenvalue vector
    Q = e.vectors
    # eigenvalue regularizer:
    dmax = 1e4*minimum(v) # take the multiple minimum of the diagonal
    #display(v)
    v = min.(dmax, v)
    #display(v)
    D = diagm(v)
    # compute B:
    return D*Q' # B = D*Qᵀ
end

function test_cov()
    # const:
    N = 3
    len_finger = 2
    # inputs:
    wmat = [1 2 3; 1 2 3]
    idx = 2 # w0 idx
    # func:
    wbar, C = mean_cov(wmat, idx, N, len_finger)
    display(wbar)
    display(C)
    B = compute_B(C)
    display(B)
    display(B'*B)
end