import NaiveNASflux
using Flux

@testset "Utils" begin
    @testset "Sizes" begin

        @test nin(Dense(3,4)) == 3
        @test nout(Dense(3,4)) == 4

    end
end
