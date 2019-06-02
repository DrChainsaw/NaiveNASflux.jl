import NaiveNASflux
using Flux

@testset "Utils" begin
    @testset "Sizes" begin

        @test nin(Dense(3,4)) == 3
        @test nout(Dense(3,4)) == 4

        @test nin(Conv((1,2), 3=>6)) == 3
        @test nout(Conv((1,2), 3=>6)) == 6

        @test nin(ConvTranspose((1,2), 3=>6)) == 3
        @test nout(ConvTranspose((1,2), 3=>6)) == 6

        @test nin(DepthwiseConv((1,2), 3=>6)) == 3
        @test nout(DepthwiseConv((1,2), 3=>6)) == 6

        @test nin(Flux.Diagonal(3)) == nout(Flux.Diagonal(3)) == 3

        @test nin(LayerNorm(3)) == nout(LayerNorm(3)) == 3
        @test nin(BatchNorm(3)) == nout(BatchNorm(3)) == 3
        @test nin(InstanceNorm(3)) == nout(InstanceNorm(3)) == 3
        @test nin(GroupNorm(3,1)) == nout(GroupNorm(3,1)) == 3

        @test nin(RNN(3,4)) == 3
        @test nout(RNN(3,4)) == 4
        @test nin(LSTM(3,4)) == 3
        @test nout(LSTM(3,4)) == 4
        @test nin(GRU(3,4)) == 3
        @test nout(GRU(3,4)) == 4

    end
end
