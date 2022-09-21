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
        display(dif[:,i]*dif[:,i]')
        S .+= dif[:,i]*dif[:,i]'
    end
end

function testsdw()
    # const:
    N = 3
    len_finger = 2
    # outputs:
    S = zeros(len_finger,len_finger)
    dw = zeros(len_finger)
    dif = zeros(len_finger, N)
    # inputs:
    wmat = [1 2 3; 1 2 3]
    idx = 2
    # func:
    Sdw_finger!(S,dw, dif, wmat, idx, N)
    display(dif)
    display(dw)
    display(S)

end