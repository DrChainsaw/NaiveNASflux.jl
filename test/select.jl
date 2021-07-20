@testset "Select" begin

    @testset "KernelSizeAligned" begin
        import NaiveNASflux: selectfilters
        pars = reshape(1:2*3*4*5, 5,4,3,2)
        ps = selectfilters(KernelSizeAligned(2, -1), pars)

        @test ps[1] == (1 => [-1, 1, 2, 3, 4, 5, -1])
        @test ps[2] == (2 => [2,3,4])
    end

    @testset "KernelSizeAligned MutableLayer $convtype" for convtype in (Conv, ConvTranspose, DepthwiseConv)
        import NaiveNASflux: weights
        m = fluxvertex(convtype((3,4), 5=>5, pad=(1,1,1,2)), inputvertex("in", 5), layerfun=identity)
        indata = ones(Float32, 5,5,5,2)

        @test size(weights(layer(m)))[1:2] == (3,4)
        @test size(m(indata)) == size(indata)

        mutate_weights(m, KernelSizeAligned((-1, 2), (1,0,3,2)))

        @test size(weights(layer(m)))[1:2] == (2,6)
        @test size(m(indata)) == size(indata)
    end

    @testset "KernelSizeAligned LazyMutable" begin
        import NaiveNASflux: weights
        m = fluxvertex(Conv((3,4), 5=>6, pad=(1,1,1,2)), inputvertex("in", 5), layerfun=LazyMutable)
        indata = ones(Float32, 5,5,5,2)

        @test size(weights(layer(m))) == (3,4,5,6)
        @test size(m(indata))[1:2] == size(indata)[1:2]

        mutate_weights(m, KernelSizeAligned((-1, 2), (1,0,3,2)))

        @test size(m(indata)) == (size(indata,1), size(indata,2), 6, size(indata,4))
        @test size(weights(layer(m))) == (2,6,5,6)

        mutate_weights(m, KernelSizeAligned((2, -1), (1,2,2,2)))
        NaiveNASlib.Δsize!(NaiveNASlib.NeuronIndices(), NaiveNASlib.OnlyFor(), m, [1:nin(m)[]], [1,3,5])

        @test size(m(indata)) == (size(indata,1), size(indata,2), 3, size(indata,4))
        @test size(weights(layer(m))) == (4,5,5,3)
    end

    @testset "KernelSizeAligned Dense is Noop with layerfun $lfun" for lfun in (identity, LazyMutable)
        m = fluxvertex(Dense(3,4), inputvertex("in", 3), layerfun = lfun)
        @test nin(m) == [3]
        indata = ones(Float32, nin(m)[], 2)

        @test size(m(indata)) == (4,2)
        mutate_weights(m, KernelSizeAligned((-1,-1), (1,1)))
        @test size(m(indata)) == (4,2)
    end
end
