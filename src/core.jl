using LinearAlgebra, KernelAbstractions, Adapt, Optim

abstract type AbstractMatrixPencil{T} end

struct MatrixPencil{T} <: AbstractMatrixPencil{T}
    A::AbstractMatrix{T}
    B::AbstractMatrix{T}
end

function MatrixPencil(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T<:Complex}
    @assert size(A) == size(B)
    MatrixPencil{T}(A, B)
end

function MatrixPencil(A::AbstractMatrix{T}, B::Union{AbstractMatrix{T},UniformScaling}=I) where {T<:Complex}
    if B isa UniformScaling
        B = Matrix{T}(I, size(A))
    end
    MatrixPencil(A, B)
end

struct SchurMatrixPencil{T} <: AbstractMatrixPencil{T}
    A::AbstractMatrix{T}
    Ac::AbstractMatrix{T}
    B::AbstractMatrix{T}
    Bc::AbstractMatrix{T}
end

function MatrixPencil(F::Schur)
    SchurMatrixPencil{eltype(F)}(Matrix{eltype(F)}(F.T), Matrix{eltype(F)}(F.T'), Matrix{eltype(F)}(I, size(F.T)), Matrix{eltype(F)}(I, size(F.T)))
end
function MatrixPencil(F::GeneralizedSchur)
    SchurMatrixPencil{eltype(F)}(Matrix{eltype(F)}(F.S), Matrix{eltype(F)}(F.S'), Matrix{eltype(F)}(F.T), Matrix{eltype(F)}(F.T'))
end

Base.size(x::AbstractMatrixPencil) = size(x.A)
Base.size(x::AbstractMatrixPencil, i) = size(x.A, i)
KernelAbstractions.get_backend(x::AbstractMatrixPencil) = get_backend(x.A)

Adapt.@adapt_structure MatrixPencil
Adapt.@adapt_structure SchurMatrixPencil

"""
    validate(zg, A, B, γ=missing, δ=missing)
    validate(zg, P, γ=missing, δ=missing)

Checks the following:
- the grid of shifts (zg) is not empty
- A and B are of the same size (if B!=UniformScaling)
- γ + δ == 1 -- normalization for the matrix-pencil pseudospectra (if given)
"""
function validate(zg, A::AbstractMatrix, B, γ=missing, δ=missing)
    @assert !isempty(zg)
    if B != I
        @assert size(A) == size(B)
    end
    if !(ismissing(γ) && ismissing(δ))
        @assert γ + δ == 1
    end
end
function validate(zg, P::AbstractMatrixPencil, γ=missing, δ=missing)
    validate(zg, P.A, P.B, γ, δ)
end
