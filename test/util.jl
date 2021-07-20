@testset "Utils" begin

    struct BogusLayer <: NaiveNASflux.FluxLayer end

    _nin(l) = nin(layertype(l), l)[]
    _nout(l) = nout(layertype(l), l)

    @testset "Sizes" begin

        @test _nin(Dense(3,4)) == 3
        @test _nout(Dense(3,4)) == 4

        @test _nin(Conv((2,), 3=>4)) == 3
        @test _nout(Conv((2,), 3=>4)) == 4
        @test _nin(Conv((1,2), 3=>6)) == 3
        @test _nout(Conv((1,2), 3=>6)) == 6
        @test _nin(Conv((1,2,3), 4=>5)) == 4
        @test _nout(Conv((1,2,3), 4=>5)) == 5

        @test _nin(ConvTranspose((2,), 3=>4)) == 3
        @test _nout(ConvTranspose((2,), 3=>4)) == 4
        @test _nin(ConvTranspose((1,2), 3=>6)) == 3
        @test _nout(ConvTranspose((1,2), 3=>6)) == 6
        @test _nin(ConvTranspose((1,2,3), 4=>5)) == 4
        @test _nout(ConvTranspose((1,2,3), 4=>5)) == 5

        @test _nin(DepthwiseConv((2,), 3=>4*3)) == 3
        @test _nout(DepthwiseConv((2,), 3=>4*3)) == 12
        @test _nin(DepthwiseConv((1,2), 3=>6*3)) == 3
        @test _nout(DepthwiseConv((1,2), 3=>6*3)) == 18
        @test _nin(DepthwiseConv((1,2,3), 4=>5*4)) == 4
        @test _nout(DepthwiseConv((1,2,3), 4=>5*4)) == 20

        @test _nin(CrossCor((2,), 3=>4)) == 3
        @test _nout(CrossCor((2,), 3=>4)) == 4
        @test _nin(CrossCor((1,2), 3=>6)) == 3
        @test _nout(CrossCor((1,2), 3=>6)) == 6
        @test _nin(CrossCor((1,2,3), 4=>5)) == 4
        @test _nout(CrossCor((1,2,3), 4=>5)) == 5

        @test _nin(Flux.Diagonal(3)) == _nout(Flux.Diagonal(3)) == 3

        @test _nin(LayerNorm(3)) == _nout(LayerNorm(3)) == 3
        @test _nin(BatchNorm(3)) == _nout(BatchNorm(3)) == 3
        @test _nin(InstanceNorm(3)) == _nout(InstanceNorm(3)) == 3
        @test _nin(GroupNorm(3,1)) == _nout(GroupNorm(3,1)) == 3

        @test _nin(RNN(3,4)) == 3
        @test _nout(RNN(3,4)) == 4
        @test _nin(LSTM(3,4)) == 3
        @test _nout(LSTM(3,4)) == 4
        @test _nin(GRU(3,4)) == 3
        @test _nout(GRU(3,4)) == 4
    end

    @testset "Dims" begin
        import NaiveNASflux: actdim, indim, outdim
        @test actdim(Dense(3,4)) == 1

        @test actdim(Conv((1,2), 3=>6)) == 3

        @test actdim(ConvTranspose((1,2), 3=>6)) == 3

        @test actdim(DepthwiseConv((1,2), 3=>6)) == 3

        @test actdim(CrossCor((1,2), 3=>6)) == 3

        @test actdim(Flux.Diagonal(1)) == indim(Flux.Diagonal(2)) == outdim(Flux.Diagonal(3)) == 1

        @test actdim(RNN(3,4)) == 1
        @test actdim(LSTM(3,4)) == 1
        @test actdim(GRU(3,4)) == 1

        @test_throws ArgumentError actdim(BogusLayer())
        @test_throws ArgumentError indim(BogusLayer())
        @test_throws ArgumentError outdim(BogusLayer())
    end
end
