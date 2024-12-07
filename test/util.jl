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

        @test nin(Flux.Scale(3)) == [nout(Flux.Scale(3))] == [3]

        @test nin(LayerNorm(3)) == [nout(LayerNorm(3))] == [3]
        @test nin(BatchNorm(3)) == [nout(BatchNorm(3))] == [3]
        @test nin(InstanceNorm(3)) == [nout(InstanceNorm(3))] == [3]
        @test nin(GroupNorm(3,1)) == [nout(GroupNorm(3,1))] == [3]

        @test nin(RNN(3 => 4)) == nin(RNN(3 => 4).cell) ==  [3]
        @test nout(RNN(3 => 4)) == nout(RNN(3 => 4).cell) == 4
        @test nin(LSTM(3 => 4)) == nin(LSTM(3 => 4).cell) == [3]
        @test nout(LSTM(3 => 4)) == nout(LSTM(3 => 4).cell) == 4
        @test nin(GRU(3 => 4)) == nin(GRU(3 => 4).cell) == [3]
        @test nout(GRU(3 => 4)) == nout(GRU(3 => 4).cell) == 4
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

        @test actdim(Flux.Scale(1)) == indim(Flux.Scale(2)) == outdim(Flux.Scale(3)) == 1

        @test actdim(GenericFluxRecurrent()) == 1
        @test actdim(RNN(3 => 4)) == actdim(RNN(3 => 4).cell) == 1
        @test actdim(LSTM(3 => 4)) == actdim(LSTM(3 => 4).cell) == 1
        @test actdim(GRU(3 => 4)) == actdim(GRU(3 => 4).cell) == 1

        @test_throws ArgumentError actdim(BogusLayer())
        @test_throws ArgumentError indim(BogusLayer())
        @test_throws ArgumentError outdim(BogusLayer())
    end

    @testset "ngroups" begin
        import NaiveNASflux: ngroups

        @test ngroups(DepthwiseConv((2,), 3 => 9)) == ngroups(Conv((2,), 3 => 9; groups=3)) == ngroups(ConvTranspose((2,), 3 => 9; groups=3)) == 3
        @test ngroups(Conv((3,3), 10 => 30; groups=5)) == ngroups(ConvTranspose((3,3), 10 => 30; groups=5)) == 5
        @test ngroups(Conv((3,3), 10 => 30; groups=2)) == ngroups(ConvTranspose((3,3), 10 => 30; groups=2)) == 2
    end 
end
