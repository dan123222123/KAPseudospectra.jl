module MetalPseudospectra

using KAPseudospectra, Sys, Metal

if Sys.isapple()
    @eval KAPseudospectra.devices(B::Metal.MtlBackend) = Metal.devices()
    @eval KAPseudospectra.get_bgarray(B::Metal.MtlBackend) = Metal.MtlArray
end

end