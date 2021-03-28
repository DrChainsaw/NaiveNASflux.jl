@testset "Select" begin

    @testset "Select parameters" begin
        import NaiveNASflux: select
        mat = reshape(collect(1:3*4), 3, 4)

        @test select(mat, 2 => [2,-1,-1,4]; newfun = (args...) -> 0) == [4 0 0 10;5 0 0 11;6 0 0 12]
        @test select(mat, 1 => [-1,1,3]; newfun = (args...) -> 0) == [0 0 0 0;1 4 7 10;3 6 9 12]

        @test select(mat, 1 => [2,-1,3,-1], 2 => [-1,1,-1,4]; newfun = (args...) -> 0) == [0 2 0 11;0 0 0 0;0 3 0 12;0 0 0 0]

        dfun = (T, d, s...) ->  d == 1 ? 10 : -10

        @test select(ones(2,2), 1 => [-1, 1], 2 => [1 , -1], newfun=dfun) == [10 -10; 1 -10]
        @test select(ones(2,2), 2 => [1 , -1], 1 => [-1, 1], newfun=dfun) == [10 10; 1 -10]

        @test select(Flux.Zeros(), 1 => [1,2,3]) == Flux.Zeros()
    end


    @testset "KernelSizeAligned" begin
        import NaiveNASflux: selectfilters
        pars = reshape(1:2*3*4*5, 5,4,3,2)
        ps = selectfilters(KernelSizeAligned(2, -1), pars)

        @test ps[1] == (1 => [-1, 1, 2, 3, 4, 5, -1])
        @test ps[2] == (2 => [2,3,4])
    end

    @testset "KernelSizeAligned MutableLayer $convtype" for convtype in (Conv, ConvTranspose, DepthwiseConv)
        import NaiveNASflux: weights
        m = mutable(convtype((3,4), 5=>5, pad=(1,1,1,2)), inputvertex("in", 5), layerfun=identity)
        indata = ones(Float32, 5,5,5,2)

        @test size(weights(layer(m)))[1:2] == (3,4)
        @test size(m(indata)) == size(indata)

        mutate_weights(m, KernelSizeAligned((-1, 2), (1,0,3,2)))

        @test size(weights(layer(m)))[1:2] == (2,6)
        @test size(m(indata)) == size(indata)
    end

    @testset "KernelSizeAligned LazyMutable" begin
        import NaiveNASflux: weights
        m = mutable(Conv((3,4), 5=>6, pad=(1,1,1,2)), inputvertex("in", 5), layerfun=LazyMutable)
        indata = ones(Float32, 5,5,5,2)

        @test size(weights(layer(m))) == (3,4,5,6)
        @test size(m(indata))[1:2] == size(indata)[1:2]

        mutate_weights(m, KernelSizeAligned((-1, 2), (1,0,3,2)))

        @test size(m(indata)) == (size(indata,1), size(indata,2), 6, size(indata,4))
        @test size(weights(layer(m))) == (2,6,5,6)

        mutate_weights(m, KernelSizeAligned((2, -1), (1,2,2,2)))
        mutate_outputs(m, [1,3,5])

        @test size(m(indata)) == (size(indata,1), size(indata,2), 3, size(indata,4))
        @test size(weights(layer(m))) == (4,5,5,3)
    end

    @testset "KernelSizeAligned Dense is Noop with layerfun $lfun" for lfun in (identity, LazyMutable)
        m = mutable(Dense(3,4), inputvertex("in", 3), layerfun = lfun)
        indata = ones(Float32, nin(m)[], 2)

        @test size(m(indata)) == (4,2)
        mutate_weights(m, KernelSizeAligned((-1,-1), (1,1)))
        @test size(m(indata)) == (4,2)
    end
end
