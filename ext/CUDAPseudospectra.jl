module CUDAPseudospectra

using KAPseudospectra, CUDA, PrecompileTools

if CUDA.functional()

    @eval begin
        KAPseudospectra.device(B::CUDA.CUDABackend) = CUDA.device()
        KAPseudospectra.device!(B::CUDA.CUDABackend, dev) = CUDA.device!(dev)
        KAPseudospectra.devices(B::CUDA.CUDABackend) = CUDA.devices()
        KAPseudospectra.get_bgarray(B::CUDA.CUDABackend) = CUDA.CuArray
        KAPseudospectra.device_bytes_available(B::CUDA.CUDABackend) = CUDA.free_memory()
        KAPseudospectra.device_reclaim(B::CUDA.CUDABackend) = CUDA.reclaim()
        ## precompile gpu code
        @setup_workload begin
            using LinearAlgebra
            for T in [ComplexF32, ComplexF64]
                m = 32
                g = 100
                gx, gy, zg = qgrid(T, (-4, 4), (-4, 4), (g, g))
                A = randn(T, m, m)
                P = MatrixPencil(schur(A))
                @compile_workload begin
                    ihlpsa(CUDABackend(), zg, P, 5; devs=[collect(CUDA.devices())[1]])
                end
            end
        end
    end

end

end