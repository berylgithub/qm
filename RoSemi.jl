using LinearAlgebra

"""
placeholder for the (Ro)bust (S)h(e)pard (mo)del for (i)nterpolation constructor
"""

"""
computes the average of the fingerprints: dw := ∑ᴺᵥ (wᵥ - w₀)/N   
params (matrices are always column major!!):
    - w_matrix, matrix containing the molecular features, ∈ Float64, size = (len_finger, N-1)
    - idx, the index of the wanted molecule (w0)
    - N, number of data
intermediate:
    - dif, wν-w0, Matrix{Float64}(len_finger, N)
output:
    - S, zeros, size = (len_finger, len_finger)
    - dw, zeros size = (len_finger)
"""
function Sdw_finger!(S, dw, dif, w_matrix, idx, N)
    # dw := 
    for i ∈ union(1:idx-1, idx+1:N) # all except the w0 index itself, since w0 - w0 = 0
        dif[:,i] = w_matrix[:, i] .- w_matrix[:, idx]
        dw .+= dif[:, i]
    end
    dw ./= N

    # S := 
    for i ∈ union(1:idx-1, idx+1:N)
        S .+= dif[:,i]*dif[:,i]'
    end
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

    display(w_matrix[:, idx])
    display(dif)
    display(dw)
    display(S)
    display(dw*dw')

    return w_matrix[:, idx] .+ dw, (S .- (dw*dw'))./(N-1) # mean, covariance
end

function test_meancov()
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

end