using NaiveNASflux, Flux, Test
using NaiveNASlib.Advanced, NaiveNASlib.Extend

function assertlayer(l, Wexp, bexp)
    @test size(Wexp) == size(weights(l))
    @test size(bexp) == size(bias(l))
    
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

    @info "Testing chainrules"
    include("chainrules.jl")

    @info "Testing neuronutility"
    include("neuronutility.jl")

    @info "Testing examples"
    include("examples.jl")

    if Int !== Int32
        import Documenter
        Documenter.doctest(NaiveNASflux)
    end
end
