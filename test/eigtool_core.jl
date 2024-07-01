using LinearAlgebra
using GridArrays
using MAT

const global tdir = (@__DIR__) * "/"

# read psa matfile and compute eigtool grid
function readvars(fname)
    vars = matread(fname)
    A = complex(vars["A"])
    gx = vec(vars["gx"])
    gy = vec(vars["gy"])
    srgref = vars["srg"]
    grid = ProductGrid(gx, gy * 1im)
    zg = Matrix{eltype(A)}(sum.(collect(grid))) # matrix
    return srgref, A, zg
end

# test svdpsa against eigtool
function testsvdpsa(fname, tol)
    srgref, A, zg = readvars(fname)
    srg = ℂsvdpsa(zg, A, I, 1, 0)
    println("Normed Error for ℂsvdpsa is $(norm(abs.(srgref .- srg))/norm(srgref))")
    return (norm(abs.(srgref .- srg)) / norm(srgref)) < tol
end

# test ihlpsa against eigtool
function testihlpsa(fname, backend, tol)
    srgref, A, zg = readvars(fname)
    P = MatrixPencil(schur(Matrix{complex(eltype(A))}(A)))
    srg = ihlpsa(backend, zg, P, size(A, 1))
    println("Normed Error for ihlpsa on $(backend) is $(norm(abs.(srgref .- srg))/norm(srgref))")
    return (norm(abs.(srgref .- srg)) / norm(srgref)) < tol
end
