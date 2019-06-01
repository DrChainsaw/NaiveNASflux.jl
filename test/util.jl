import NaiveNASflux
using Flux

@testset "Utils" begin
    @testset "Sizes" begin

        @test nin(Dense(3,4)) == 3
        @test nout(Dense(3,4)) == 4

        for c in (Conv, ConvTranspose, DepthwiseConv)
            @info "\ttest size for $c"
            @test nin(c((1,2), 3=>6)) == 3
            @test nout(c((1,2), 3=>6)) == 6
        end

        @test nin(Flux.Diagonal(3)) == nout(Flux.Diagonal(3)) == 3
    end
end
