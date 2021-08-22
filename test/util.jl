@testset "Utils" begin

    struct BogusLayer <: NaiveNASflux.FluxLayer end

    @testset "Sizes" begin

        @test nin(Dense(3,4)) == [3]
        @test nout(Dense(3,4)) == 4

        @test nin(Conv((2,), 3=>4)) == [3]
        @test nout(Conv((2,), 3=>4)) == 4
        @test nin(Conv((1,2), 3=>6)) == [3]
        @test nout(Conv((1,2), 3=>6)) == 6
        @test nin(Conv((1,2,3), 4=>5)) == [4]
        @test nout(Conv((1,2,3), 4=>5)) == 5

        @test nin(ConvTranspose((2,), 3=>4)) == [3]
        @test nout(ConvTranspose((2,), 3=>4)) == 4
        @test nin(ConvTranspose((1,2), 3=>6)) == [3]
        @test nout(ConvTranspose((1,2), 3=>6)) == 6
        @test nin(ConvTranspose((1,2,3), 4=>5)) == [4]
        @test nout(ConvTranspose((1,2,3), 4=>5)) == 5

        @test nin(DepthwiseConv((2,), 3=>4*3)) == [3]
        @test nout(DepthwiseConv((2,), 3=>4*3)) == 12
        @test nin(DepthwiseConv((1,2), 3=>6*3)) == [3]
        @test nout(DepthwiseConv((1,2), 3=>6*3)) == 18
        @test nin(DepthwiseConv((1,2,3), 4=>5*4)) == [4]
        @test nout(DepthwiseConv((1,2,3), 4=>5*4)) == 20

        @test nin(CrossCor((2,), 3=>4)) == [3]
        @test nout(CrossCor((2,), 3=>4)) == 4
        @test nin(CrossCor((1,2), 3=>6)) == [3]
        @test nout(CrossCor((1,2), 3=>6)) == 6
        @test nin(CrossCor((1,2,3), 4=>5)) == [4]
        @test nout(CrossCor((1,2,3), 4=>5)) == 5

        @test nin(Flux.Diagonal(3)) == [nout(Flux.Diagonal(3))] == [3]

        @test nin(LayerNorm(3)) == [nout(LayerNorm(3))] == [3]
        @test nin(BatchNorm(3)) == [nout(BatchNorm(3))] == [3]
        @test nin(InstanceNorm(3)) == [nout(InstanceNorm(3))] == [3]
        @test nin(GroupNorm(3,1)) == [nout(GroupNorm(3,1))] == [3]

        @test nin(RNN(3,4)) == [3]
        @test nout(RNN(3,4)) == 4
        @test nin(LSTM(3,4)) == [3]
        @test nout(LSTM(3,4)) == 4
        @test nin(GRU(3,4)) == [3]
        @test nout(GRU(3,4)) == 4
    end

    @testset "Dims" begin
        import NaiveNASflux: actdim, indim, outdim
        import NaiveNASflux: GenericFlux2D, GenericFluxConvolutional, GenericFluxRecurrent
        @test actdim(Dense(3,4)) == actdim(GenericFlux2D()) == 1

        @test actdim(GenericFluxConvolutional{2}()) ==  3
        @test actdim(Conv((1,2), 3=>6)) == 3
        @test actdim(ConvTranspose((1,2), 3=>6)) == 3
        @test actdim(DepthwiseConv((1,2), 3=>6)) == 3
        @test actdim(CrossCor((1,2), 3=>6)) == 3

        @test actdim(Flux.Diagonal(1)) == indim(Flux.Diagonal(2)) == outdim(Flux.Diagonal(3)) == 1

        @test actdim(GenericFluxRecurrent()) == 1
        @test actdim(RNN(3,4)) ==  1
        @test actdim(LSTM(3,4)) == 1
        @test actdim(GRU(3,4)) == 1

        @test_throws ArgumentError actdim(BogusLayer())
        @test_throws ArgumentError indim(BogusLayer())
        @test_throws ArgumentError outdim(BogusLayer())
    end
end
