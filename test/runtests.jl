using NaiveNASflux
using Test

function assertlayer(l, Wexp, bexp)
    @test size(Wexp) == size(weights(l))
    @test size(bexp) == size(bias(l))
    @test Wexp == weights(l)
    @test bexp == bias(l)
end

@testset "NaiveNASflux.jl" begin

    include("util.jl")
    include("mutable.jl")
    include("vertex.jl")

end
