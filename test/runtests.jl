using NaiveNASflux
using Test

function assertlayer(l, Wexp, bexp)
    @test size(Wexp) == size(weights(l))
    @test size(bexp) == size(bias(l))
    @test Wexp == weights(l)
    @test bexp == bias(l)
end

@testset "NaiveNASflux.jl" begin

    NaiveNASlib.set_defaultΔNoutStrategy(DefaultJuMPΔSizeStrategy())
    NaiveNASlib.set_defaultΔNinStrategy(DefaultJuMPΔSizeStrategy())


    @info "Testing util"
    include("util.jl")

    @info "Testing mutable"
    include("mutable.jl")

    @info "Testing vertex"
    include("vertex.jl")

    @info "Testing pruning"
    include("pruning.jl")

    @info "Testing examples"
    include("examples.jl")
end
