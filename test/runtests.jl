using Test, KAPseudospectra

if isempty(ARGS) || "all" in ARGS
    all_tests = true
else
    all_tests = false
end

include("eigtool_core.jl")

@testset "svdpsa" begin
    @testset "parter16" begin
        @testset "F32" begin
            fname = tdir * "F32parter16.mat"
            maxtol = 10^-6
            @test testsvdpsa(fname, maxtol)
        end
        @testset "F64" begin
            fname = tdir * "F64parter16.mat"
            maxtol = 10^-14
            @test testsvdpsa(fname, maxtol)
        end
    end
end

function test_ihlpsa(backend)
    @testset "ihlpsa -- $(backend)" begin
        @testset "parter16" begin
            @testset "F32" begin
                fname = tdir * "F32parter16.mat"
                maxtol = 10^-6
                @test testihlpsa(fname, backend, maxtol)
            end
            @testset "F64" begin
                fname = tdir * "F64parter16.mat"
                maxtol = 10^-14
                @test testihlpsa(fname, backend, maxtol)
            end
        end
    end
end

if all_tests || "cpu" in ARGS
    using KernelAbstractions
    test_ihlpsa(CPU())
end

if all_tests || "cuda" in ARGS
    using CUDA
    if CUDA.functional()
        test_ihlpsa(CUDABackend())
    end
end

if all_tests || "amdgpu" in ARGS
    using AMDGPU
    if AMDGPU.functional()
        test_ihlpsa(ROCBackend())
    end
end
