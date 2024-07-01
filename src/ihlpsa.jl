using ArraysOfArrays
using ThreadsX
using ProgressBars

# triangular solves using kernel abstractions
include("KATRSM.jl/KATRSM.jl")
using .KATRSM

## KERNELS ##

# 1-tensor -> vector of vectors
@kernel function _v2v(V, W)
    I = @index(Global, Linear)
    g = length(V)
    m = length(V[1])
    if I <= g
        for i = 1:m
            W[I][i] = V[I, i]
        end
    end
end

# TODO polish
# each thread will take one grid point and do all of its calculations independently
# v is a length g vector of m-dimensional vectors
# βₙ₊₁v is a g-dimensional vector
# Qₙ₊₁v is a length g vector of m-dimensional vectors
@kernel function _qₙnext(v, βₙ₊₁, Qₙ₊₁)
    I = @index(Global, Linear)
    g = length(v)
    m = length(v[1])
    if I <= g
        # norm of v
        vnorm = zero(eltype(v[1]))
        for i = 1:m
            vnorm += conj(v[I][i]) * v[I][i]
        end
        vnorm = sqrt(vnorm)
        # update qₙ₊₁
        for j = 1:m
            Qₙ₊₁[I, j] = v[I][j] / vnorm
        end
        ## set βₙ₊₁
        βₙ₊₁[I] = vnorm
    end
end

# TODO polish
# kernel for the Lanczos 3-term recurance + qₙnext
# each thread block will handle a grid point
@kernel function _ihl_ttr_qₙnext(βₙ₋₁, Qₙ₋₁, αₙ, Qₙ, v, βₙ₊₁)
    I = @index(Global, Linear)
    g = length(v)
    m = length(v[1])
    if I <= g
        # ttr
        for i = 1:m
            v[I][i] -= βₙ₋₁[I] * Qₙ₋₁[I, i]
        end
        αₙ[I] = zero(eltype(v[1]))
        for i = 1:m
            αₙ[I] += conj(Qₙ[I, i]) * v[I][i]
        end
        for i = 1:m
            v[I][i] -= αₙ[I] * Qₙ[I, i]
            Qₙ₋₁[I, i] = Qₙ[I, i] # gvecv
        end
        # qₙnext
        vnorm = zero(real(eltype(v[I])))
        for i = 1:m
            vnorm += real(conj(v[I][i]) * v[I][i])
        end
        vnorm = sqrt(vnorm)
        for j = 1:m
            Qₙ[I, j] = v[I][j] / vnorm
            v[I][j] = Qₙ[I, j] # gvecv no2
        end
        βₙ₊₁[I] = vnorm
    end
end

## END KERNELS ##

struct IHLworkspace{T,B}
    maxbatch
    zv::AbstractVector{T}
    P::AbstractMatrixPencil{T}
    x₀
    Qv
    v
end

function IHLworkspace(P::AbstractMatrixPencil{T}, maxbatch, x₀=missing) where {T<:Complex}
    m = size(P, 1)
    zv = zeros(T, maxbatch)
    if ismissing(x₀)
        x = randn(T, m)
        x₀ = VectorOfSimilarVectors(repeat(x / norm(x), outer=(1, maxbatch)))
    elseif !(x₀ isa VectorOfSimilarVectors)
        x₀ = VectorOfSimilarVectors(repeat(x₀ / norm(x₀), outer=(1, maxbatch)))
    end
    Qv = VectorOfSimilarArrays(zeros(T, maxbatch, m, 2))
    v = VectorOfSimilarVectors(zeros(T, m, maxbatch))
    v .= deepcopy(x₀)
    IHLworkspace{T,get_backend(P)}(maxbatch, zv, P, x₀, Qv, v)
end

function Adapt.adapt_structure(to, ihl::IHLworkspace)
    zv = adapt(to, ihl.zv)
    P = adapt(to, ihl.P)
    x₀ = adapt(to, ihl.x₀)
    Qv = adapt(to, ihl.Qv)
    v = adapt(to, ihl.v)
    IHLworkspace{eltype(zv),get_backend(P)}(ihl.maxbatch, zv, P, x₀, Qv, v)
end

# extend get_backend for IHLworkspace
KernelAbstractions.get_backend(x::IHLworkspace{T,B}) where {T,B} = B

## DEVICE FUNCTIONS ##

# non-cpu solve step in lockstep_ihl!
function trsmIHL(backend, bV, zv, P::SchurMatrixPencil; wgs=256)
    g = length(zv)
    @views _batched_column_oriented_forward_solve_pencil(backend, wgs, (wgs, g))(bV, conj(zv), P.Ac, P.Bc)
    @views _batched_column_oriented_backward_solve_pencil(backend, wgs, (wgs, g))(bV, zv, P.A, P.B)
end

# cpu solve step in lockstep_ihl!
function trsmIHL(backend::CPU, bV, zv, P::SchurMatrixPencil)
    g = length(zv)
    _batched_forward_solve_pencil(backend)(bV, conj(zv), P.A', P.B', ndrange=g)
    _batched_backward_solve_pencil(backend)(bV, zv, P.A, P.B, ndrange=g)
end

function lockstep_ihl!(α, β, ihl::IHLworkspace, nit, g; wgs=256)
    backend = get_backend(ihl)
    ihl.v .= ihl.x₀
    _qₙnext(backend)(view(ihl.x₀, 1:g), view(β, 2, 1:g), view(ihl.Qv[2], 1:g, :), ndrange=g)
    _v2v(backend)(view(ihl.Qv[2], 1:g, :), view(ihl.v, 1:g), ndrange=g)
    for n = 1:nit
        trsmIHL(backend, view(ihl.v, 1:g), view(ihl.zv, 1:g), ihl.P; wgs)
        _ihl_ttr_qₙnext(backend)(view(β, n, 1:g), view(ihl.Qv[1], 1:g, :), view(α, n, 1:g), view(ihl.Qv[2], 1:g, :), view(ihl.v, 1:g), view(β, n + 1, 1:g), ndrange=g)
    end
    synchronize(backend)
end

# device operations "interface" for kernel abstractions
get_bgarray(B::CPU) = Array
device(B::CPU) = CPU()
devices(B::CPU) = CPU()
device!(B::CPU, dev) = CPU()
device_bytes_available(B::CPU) = (Sys.free_memory() |> Int)
device_reclaim(B::CPU) = GC.gc()

## END DEVICE FUNCTIONS ##

## HOST FUNCTIONS ##

# separate srg computations
function ihlsrg!(sr, zv, γ, δ, α, β)
    Threads.@threads for i in eachindex(zv)
        Tv = Hermitian(diagm(0 => real(α[:, i]), -1 => real(β[2:end, i]), 1 => real(β[2:end, i]))[1:end-1, 1:end-1])
        try
            sr[i] = (γ + δ * abs(zv[i])) / sqrt(eigmax(Tv))
        catch
            sr[i] = eps(real(eltype(zv)))
        end
    end
end

## END HOST FUNCTIONS ##

## WRAPPER FUNCTIONS ##

# single-device batched inverse lanczos pseudospectra
function sdihlpsa(
    backend;
    zg::AbstractArray{T,2},
    P::AbstractMatrixPencil{T},
    γ,
    δ,
    zpd::Integer,
    nit::Integer=ceil(Integer, log2(size(P, 1))),
    x₀::Union{Missing,AbstractVector{T},AbstractArrayOfSimilarArrays{T}}=missing,
    pchnl::Union{Missing,Channel}=missing,
    wgs=256
) where {T<:Complex}
    dev = device(backend)
    bgarray = get_bgarray(backend)
    zv = collect(Iterators.flatten(zg))
    gtotal = length(zv)
    sr = zeros(real(T), length(zv))
    idxbatches = Vector(collect(Iterators.partition(1:gtotal, min(gtotal, zpd))))
    batches = idxbatches
    dzv = adapt(bgarray, zv)
    α = adapt(bgarray, zeros(T, nit, gtotal))
    β = adapt(bgarray, zeros(T, nit + 1, gtotal))
    ihl = adapt(bgarray, IHLworkspace(P, zpd, x₀))
    _foreach = !KernelAbstractions.isgpu(backend) ? ThreadsX.foreach : Base.foreach
    @sync _foreach(batches) do idxb
        view(ihl.zv, 1:length(idxb)) .= view(dzv, idxb)
        lockstep_ihl!(view(α, :, idxb), view(β, :, idxb), ihl, nit, length(idxb); wgs)
        Threads.@spawn begin
            device!(backend, dev)
            if !ismissing(pchnl)
                put!(pchnl, length(idxb) * nit)
            end
            ihlsrg!(view(sr, idxb), view(zv, idxb), γ, δ, adapt(Array, α[:, idxb]), adapt(Array, β[:, idxb]))
        end
    end
    return Matrix{real(T)}(reshape(sr, size(zg)))
end

function sdihlpsa(
    backend,
    zg::AbstractArray{T,2},
    P::AbstractMatrixPencil{T},
    γ,
    δ,
    zpd::Integer,
    nit::Integer=ceil(Integer, log2(size(P, 1))),
    x₀::Union{Missing,AbstractVector{T},AbstractArrayOfSimilarArrays{T}}=missing,
    pchnl::Union{Missing,Channel}=missing,
    wgs=256
) where {T<:Complex}
    sdargs = (; zg, P, γ, δ, zpd, nit, x₀, pchnl, wgs)
    sdihlpsa(backend; sdargs...)
end

# computes the largest maxbatch that will fit in the target device memory up to the specified margin of error (moe)
function findmaxbatchihl(backend, T, m; moe=0.1)
    device_reclaim(backend)
    floor(Integer, (device_bytes_available(backend) * (1 - moe) - (sizeof(T) * (4 * m * m + 1))) / (sizeof(T) * (1 + 4 * m)))
end

# multi-device, general purpose batched inverse lanczos pseudospectra caller
function ihlpsa(
    backend,
    zg::AbstractArray{T,2},
    P::AbstractMatrixPencil{T},
    nit::Integer=ceil(Integer, log2(size(P, 1))),
    γ=1,
    δ=0;
    x₀::Union{Missing,AbstractVector{T},AbstractMatrix{T}}=missing,
    progress=false,
    zpd=missing,
    devs=missing,
    wgs=256
) where {T<:Complex}
    m = size(P.A, 1)
    # this progress bar can probably be generalized to all psa methods via a channel -- TODO
    pbar = ProgressBar(total=nit * length(zg), printing_delay=0.001)
    pchnl = Channel()
    Threads.@spawn begin
        while (isopen(pchnl))
            wait(pchnl)
            ProgressBars.update(pbar, take!(pchnl))
        end
    end
    if KernelAbstractions.isgpu(backend)
        if ismissing(devs)
            devs = devices(backend)
        end
        set_description(pbar, "$(length(devs)) device(s), grid points * nit:")
        zgidxbatches = Vector(collect(Iterators.partition(1:size(zg, 2), ceil(Integer, size(zg, 2) / length(devs)))))
        results = Vector{Any}(undef, length(devs))
        @sync begin
            for (did, dev) in enumerate(devs)
                Threads.@spawn begin
                    device!(backend, dev)
                    zgb = zg[:, zgidxbatches[did]]
                    if ismissing(zpd)
                        zpd = min(findmaxbatchihl(backend, T, m), length(zgb))
                    end
                    if progress
                        results[did] = sdihlpsa(backend, zgb, P, γ, δ, zpd, nit, x₀, pchnl, wgs)
                    else
                        results[did] = sdihlpsa(backend, zgb, P, γ, δ, zpd, nit, x₀, missing, wgs)
                    end
                end
            end
        end
        result = (hcat(results...))::Matrix{real(T)}
    else
        # note, cpu CANNOT currently batch zg -- there are race conditions present due to pre-allocation of ihl for device codes
        # if you run out of memory here...you should have just used the gpu anyways!
        set_description(pbar, "CPU device, grid points * nit:")
        result = sdihlpsa(backend, zg, P, γ, δ, length(zg), nit, x₀, missing, wgs)
    end
    close(pchnl)
    return result'
end

## END WRAPPER FUNCTIONS ##