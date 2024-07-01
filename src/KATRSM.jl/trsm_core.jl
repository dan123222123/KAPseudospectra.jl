"""
    forward_solve!(b, M)

Given a square, non-singular lower triangular matrix M, compute `x=M⁻¹b` in-place.
"""
function forward_solve!(b, M)
    m = size(M, 1)
    for i = 1:m
        for j = 1:i-1
            b[i] = b[i] - b[j] * M[i, j]
        end
        b[i] = b[i] / M[i, i]
    end
end
function column_oriented_forward_solve!(b, M)
    m = size(M, 1)
    for j = 1:m
        b[j] /= M[j, j]
        for i = j+1:m # accesses are non-coallesced?
            b[i] -= b[j] * M[i, j]
        end
    end
end

"""
    backward_solve!(b, M)

Given a square, non-singular upper triangular matrix M, compute `x=M⁻¹b` in-place.
"""
function backward_solve!(b, M)
    m = size(M, 1)
    for i = m:-1:1
        for j = i+1:m
            b[i] = b[i] - b[j] * M[i, j]
        end
        b[i] = b[i] / M[i, i]
    end
end
function column_oriented_backward_solve!(b, M)
    m = size(M, 1)
    for j = m:-1:1
        b[j] /= M[j, j]
        for i = 1:j-1
            b[i] -= b[j] * M[i, j]
        end
    end
end