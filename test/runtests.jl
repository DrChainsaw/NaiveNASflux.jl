using NaiveNASflux, Flux, Test
using NaiveNASlib.Advanced, NaiveNASlib.Extend

function assertlayer(l, Wexp, bexp)
    @test size(Wexp) == size(weights(l))
    if bexp isa Flux.Zeros
        @test bias(l) isa Flux.Zeros
    else
        @test size(bexp) == size(bias(l))
    end
    @test Wexp == weights(l)
    @test bexp == bias(l)
end

@testset "NaiveNASflux.jl" begin

    @info "Testing util"
    include("util.jl")

    @info "Testing select"
    include("select.jl")

    @info "Testing mutable"
    include("mutable.jl")

    @info "Testing vertex"
    include("vertex.jl")

    @info "Testing pruning"
    include("pruning.jl")

    @info "Testing examples"
    include("examples.jl")
end
