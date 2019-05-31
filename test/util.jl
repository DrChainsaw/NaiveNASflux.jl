import NaiveNASflux
using Flux

@testset "Utils" begin
    @testset "Sizes" begin

        @test nin(Dense(3,4)) == 3
        @test nout(Dense(3,4)) == 4

        @test nin(Conv((2,3), 4=>5)) == 4
        @test nout(Conv((2,3), 4=>5)) == 5
    end
end
