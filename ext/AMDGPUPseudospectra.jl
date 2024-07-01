module AMDGPUPseudospectra

using KAPseudospectra, AMDGPU

if AMDGPU.functional()

    @eval begin
        KAPseudospectra.device(B::AMDGPU.ROCBackend) = AMDGPU.device()
        KAPseudospectra.device!(B::AMDGPU.ROCBackend, dev) = AMDGPU.device!(dev)
        KAPseudospectra.devices(B::AMDGPU.ROCBackend) = AMDGPU.devices()
        KAPseudospectra.get_bgarray(B::AMDGPU.ROCBackend) = AMDGPU.ROCArray
        KAPseudospectra.device_bytes_available(B::AMDGPU.ROCBackend) = AMDGPU.Runtime.Mem.free()
        KAPseudospectra.device_reclaim(B::AMDGPU.ROCBackend) = AMDGPU.HIP.reclaim()
    end

end

end