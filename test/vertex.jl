import NaiveNASflux: weights, bias

@testset "InputShapeVertex" begin
    v = inputvertex("in", 3, FluxDense())
    @test layertype(v) == FluxDense()
    @test name(v) == "in"
    @test nout(v) == 3

    c = clone(v)
    @test layertype(c) == FluxDense()
    @test name(c) == "in"
    @test nout(c) == 3
end

@testset "Size mutations" begin

    @testset "Dense to Dense" begin
        inpt = inputvertex("in", 4)
        dl1 = Dense(4, 5)
        dl2 = Dense(5, 3)
        bias(dl1)[1:end] = 1:5
        bias(dl2)[1:end] = 1:3
        dense1 = mutable(dl1, inpt)
        dense2 = mutable(dl2, dense1)

        @test inputs(dense2) == [dense1]
        @test outputs(dense1) == [dense2]
        @test inputs(dense1) == [inpt]
        @test outputs(inpt) == [dense1]

        @test layer(dense1) == dl1
        @test layer(dense2) == dl2
        @test layertype(dense1) == FluxDense()
        @test layertype(dense2) == FluxDense()

        @test [nout(inpt)] == nin(dense1) == [4]
        @test [nout(dense1)] == nin(dense2) == [5]
        @test nout(dense2) == 3

        inds = Bool[1, 1, 0, 1, 1]
        NaiveNASlib.Δsize!(dense1) do v
            v == dense1 || return 1
            return inds .- 0.5
        end

        @test [nout(inpt)] == nin(dense1) == [4]
        @test [nout(dense1)] == nin(dense2) == [4]
        @test nout(dense2) == 3

        W2exp, b2exp = weights(dl2)[:, inds], bias(dl2)
        W1exp, b1exp = weights(dl1)[inds, :], bias(dl1)[inds]

        @test size(CompGraph([inpt], [dense2])(collect(Float32, 1:nout(inpt)))) == (3,)

        assertlayer(layer(dense2), W2exp, b2exp)
        assertlayer(layer(dense1), W1exp, b1exp)
    end

    @testset "Invariant parametric layer" begin
        inpt = inputvertex("in", 3)
        cv = mutable(Conv((1,1), nout(inpt) => 4), inpt)
        bv = mutable(BatchNorm(nout(cv)), cv)

        @test nin(bv) == [nout(cv)] == [4]

        Δnin!(v -> 1, bv, -1)

        @test nin(bv) == [nout(cv)] == [3]
    end

    @testset "Invariant non-parametric layer" begin
        inpt = inputvertex("in", 3)
        cv = mutable(Conv((1,1), nout(inpt) => 4), inpt)
        bv = mutable(MeanPool((2,2)), cv)

        @test nin(bv) == [nout(cv)] == [4]

        Δnin!(v -> 1, bv, -1)

        @test nin(bv) == [nout(cv)] == [3]
    end

    @testset "DepthwiseConv" begin
        import NaiveNASflux: outdim, wrapped
        lazymutable(v::AbstractVertex) = lazymutable(base(v))
        lazymutable(v::CompVertex) = lazymutable(v.computation)
        lazymutable(m::AbstractMutableComp) = lazymutable(wrapped(m)) 
        lazymutable(m::LazyMutable) = m
        lazyouts(v) = lazymutable(v).outputs
        lazyins(v) = lazymutable(v).inputs

        @testset "Depthwise single layer" begin
            # just to check that I have understood the wiring of the weight
            @testset "4 inputs times 2" begin
                inpt = inputvertex("in", 4)
                dc = mutable("dc", DepthwiseConv(reshape(Float32[1 1 1 1;2 2 2 2], 1, 1, 2, 4), Float32[0,0,0,0,1,1,1,1]), inpt)
                @test neuron_value(dc) == [1,2,1,2,2,3,2,3]
                @test reshape(dc(fill(10, (1,1,4,1))), :) == [10, 20, 10, 20, 11, 21, 11, 21]
                @test Δnout!(dc => -4)
                @test lazyouts(dc) == [2, 4, 6, 8] 
                @test reshape(dc(fill(10, (1,1,4,1))), :) == [20, 20, 21, 21] 
                @test Δnout!(dc, 4)   
                @test lazyouts(dc) == [1,-1, 2,-1, 3,-1, 4,-1] 
                # TODO: Add kwargs to NaiveNASlib mutation functions
                # In the meantime, we just create a new MutableLayer instead of trying to dig up the right one from dc
                mdc = MutableLayer(layer(dc))
                NaiveNASflux.mutate(mdc, inputs=lazyins(dc)[1], outputs=lazyouts(dc), insert=(args...) -> (args...) -> 0)
                @test reshape(mdc(fill(10, (1,1,4,1))), :) == [20, 0, 20, 0, 21, 0, 21, 0] 
            end

            @testset "2 inputs times 3" begin
                inpt = inputvertex("in", 2)
                dc = mutable("dc", DepthwiseConv(reshape(Float32[1 1;2 2;3 3], 1, 1, 3, 2), Float32[0,0,1,1,2,2]), inpt)
                @test reshape(dc(fill(10, (1,1,2,1))), :) == [10, 20, 31, 11, 22, 32]
                @test Δnout!(dc => -2)
                @test lazyouts(dc) == [2,3,5,6] 
                @test reshape(dc(fill(10, (1,1,2,1))), :) == [20, 31, 22, 32]
                @test Δnout!(dc, 4)   
                @test lazyouts(dc) == [1, 2, -1, -1, 3, 4, -1, -1]
                mdc = MutableLayer(layer(dc))
                NaiveNASflux.mutate(mdc, inputs=lazyins(dc)[1], outputs=lazyouts(dc), insert=(args...) -> (args...) -> 0)
                @test reshape(mdc(fill(10, (1,1,2,1))), :) == [20, 31, 0, 0, 22, 32, 0, 0]
            end

            @testset "1 input times 5" begin
                inpt = inputvertex("in", 1)
                dc = mutable("dc", DepthwiseConv(reshape(Float32.(1:5), 1, 1, 5, 1), Float32.(1:5)), inpt)
                @test reshape(dc(fill(10, (1,1,1,1))), :) == [11, 22, 33, 44, 55]
                @test Δnout!(dc=>-2)
                @test lazyouts(dc) == 3:5 
                @test reshape(dc(fill(10, (1,1,1,1))), :) == [33, 44, 55]
                @test Δnout!(dc=>3)
                @test lazyouts(dc) == vcat(1:3, fill(-1, 3)) 
                mdc = MutableLayer(layer(dc))
                NaiveNASflux.mutate(mdc, inputs=lazyins(dc)[1], outputs=lazyouts(dc), insert=(args...) -> (args...) -> 0)
                @test reshape(mdc(fill(10, (1,1,1,1))), :) == [33, 44, 55, 0, 0, 0]
            end

            @testset "3 inputs times 7" begin
                inpt = inputvertex("in", 3)
                dc = mutable("dc", DepthwiseConv(reshape(repeat(Float32.(1:7), 3), 1,1,7,3), Float32.(1:21)), inpt)
                @test reshape(dc(fill(100, (1,1,3,1))), :) == repeat(100:100:700, 3) .+ (1:21)
                @test Δnout!(dc => -9) do v
                    v == dc || return 1
                    val = ones(nout(v))
                    val[[2,13,14]] .= -10
                    return val
                end
                @test lazyouts(dc) == [1,3,4,5,8,10,11,12,15,17,18,19]
                @test reshape(dc(fill(100, (1,1,3,1))), :) == [101,303,404,505,108,310,411,512,115,317,418,519]
                @test Δnout!(dc => 6)
                @test lazyouts(dc) == vcat(1:4, -1, -1, 5:8, -1, -1, 9:12, -1, -1)
                mdc = MutableLayer(layer(dc))
                NaiveNASflux.mutate(mdc, inputs=lazyins(dc)[1], outputs=lazyouts(dc), insert=(args...) -> (args...) -> 0)
                @test reshape(mdc(fill(100, (1,1,3,1))), :) == [101,303,404,505,0,0,108,310,411,512,0,0,115,317,418,519,0,0]
            end
        end

        @testset "DepthwiseConv groupsize 2 into groupsize 1" begin
            
            inpt = inputvertex("in", 4)
            dc1 = mutable("dc1", DepthwiseConv((2,2), nout(inpt) => 2 * nout(inpt)), inpt)
            dc2 = mutable("dc2", DepthwiseConv((2,2), nout(dc1) => nout(dc1)), dc1)

            @test @test_logs (:warn, r"Could not change nout of") Δnout!(v -> 1, dc1, 2)
            @test [nout(dc1)] == nin(dc2) == [12]
            @test nout(dc2) == 24 #TODO: Why so big??

            # Add deterministic valuefunction which wants to do non-contiguous selection across groups
            @test @test_logs (:warn, r"Could not change nout of") Δnout!(v -> repeat([1, 2], nout(v) ÷ 2), dc1, -2)
            @test [nout(dc1)] == nin(dc2) == [8]

            @test lazyins(dc1) == [1:nout(inpt)]
            @test [lazyouts(dc1)] == lazyins(dc2) == [[2, -1, 4, -1, 6, -1, 8, -1]]

            # Test that we actually succeeded in making a valid model
            y1 = dc1(ones(Float32, 3,3, nout(inpt), 2))
            @test size(y1, outdim(dc1)) == nout(dc1)
            y2 = dc2(y1)
            @test size(y2, outdim(dc2)) == nout(dc2)
        end

        @testset "DepthwiseConv groupsize 3 into groupsize 5" begin
            import NaiveNASflux: outdim
            # TODO: More testing, old implementation seems to have worked by accident
            inpt = inputvertex("in", 4)
            dc1 = mutable("dc1", DepthwiseConv((2,2), nout(inpt) => 3 * nout(inpt)), inpt)
            dc2 = mutable("dc2", DepthwiseConv((2,2), nout(dc1) => 5 * nout(dc1)), dc1)
            dc3 = mutable("dc3", DepthwiseConv((2,2), nout(dc2) => nout(dc2)), dc2)

            # TODO: Check compgraph output pre and post?
            # Need to insert zeros then?

            @test @test_logs (:warn, r"Could not change nout of") Δnout!(v -> 1, dc1, 2)
            @test [nout(dc1)] == nin(dc2) == [16]
            @test [nout(dc2)] == nin(dc3) == [96] # TODO: Why so big??

           # Add deterministic valuefunction which wants to do non-contiguous selection across groups
            @test @test_logs (:warn, r"Could not change nout of") Δnout!(v -> repeat([1, 2], nout(v) ÷ 2), dc1, -2)
            @test [nout(dc1)] == nin(dc2) == [12]
            @test [nout(dc2)] == nin(dc3) == [96]

            @test lazyins(dc1) == [1:nout(inpt)]
            @test [lazyouts(dc1)] == lazyins(dc2) == [[2, 3, -1, 5, 6, -1, 8, 9, -1, 11, 12, -1]]

            # All neurons had a positive value, so NaiveNASlib should inrease to next valid size
            @test [lazyouts(dc2)] == lazyins(dc3)
            
            # Test that we actually succeeded in making a valid model
            y1 = dc1(ones(Float32,5,5, nout(inpt), 2))
            @test size(y1, outdim(dc1)) == nout(dc1)
            y2 = dc2(y1)
            @test size(y2, outdim(dc2)) == nout(dc2)
            y3 = dc3(y2)
            @test size(y3, outdim(dc3)) == nout(dc3)
        end
    end

    @testset "Concatenate activations" begin

        function testgraph(layerfun, nin1, nin2)
            vfun = (v, s) -> mutable(layerfun(nout(v), s), v)
            return testgraph_vfun(vfun, nin1, nin2)
        end

        function testgraph_vfun(vertexfun, nin1::Integer, nin2::Integer)
            in1 = inputvertex("in1", nin1)
            in2 = inputvertex("in2", nin2)
            return testgraph_vfun(vertexfun, in1, in2)
        end

        function testgraph_vfun(vertexfun, in1, in2)
            l1 = vertexfun(in1, 3)
            l2 = vertexfun(in2, 7)

            joined = concat(l1, l2)
            l3 = vertexfun(joined, 9)

            @test nout(joined) == nout(l1) + nout(l2) == 10
            return CompGraph([in1, in2], l3)
        end

        @testset "Concatenate Dense" begin
            nin1 = 2
            nin2 = 5
            @test size(testgraph(Dense, nin1, nin2)(ones(nin1), ones(nin2))) == (9,)
        end

        @testset "Concatenate $rnntype" for rnntype in (RNN, GRU, LSTM)
            nin1 = 2
            nin2 = 5
            indata1 = reshape(collect(Float32, 1:nin1*4), nin1, 4)
            indata2 = reshape(collect(Float32, 1:nin2*4), nin2, 4)

            @test size(testgraph(rnntype, nin1, nin2)(indata1, indata2)) == (9,4)
        end

        @testset "Concatenate $convtype" for convtype in (Conv, ConvTranspose, CrossCor)
            nin1 = 2
            nin2 = 5
            indata1 = reshape(collect(Float32, 1:nin1*4*4), 4, 4, nin1, 1)
            indata2 = reshape(collect(Float32, 1:nin2*4*4), 4, 4, nin2, 1)

            convfun = (nin,nout) -> convtype((3,3), nin=>nout, pad = (1,1))
            @test size(testgraph(convfun, nin1, nin2)(indata1, indata2)) == (4,4,9,1)
        end

        @testset "Concatenate Pooled Conv" begin
            nin1 = 2
            nin2 = 5
            indata1 = reshape(collect(Float32, 1:nin1*4*4), 4, 4, nin1, 1)
            indata2 = reshape(collect(Float32, 1:nin2*4*4), 4, 4, nin2, 1)
            function vfun(v, s)
                cv = mutable(Conv((3,3), nout(v)=>s, pad = (1,1)), v)
                return mutable(MaxPool((3,3), pad=(1,1), stride=(1,1)), cv)
            end
            @test size(testgraph_vfun(vfun, nin1, nin2)(indata1, indata2)) == (4,4,9,1)
        end

        @testset "Concatenate InputShapeVertex" begin
            nin1 = 6
            nin2 = 4
            in1 = inputvertex("in1", nin1, FluxDense())
            in2 = inputvertex("in2", nin2, FluxDense())
            @test size(testgraph_vfun((v,s) -> v, in1, in2)(ones(nin1), ones(nin2))) == (10,)
        end

        @testset "Concatenate BatchNorm only" begin
            nin1 = 3
            nin2 = 7
            indata1 = reshape(collect(Float32, 1:nin1*4*4), 4, 4, nin1, 1)
            indata2 = reshape(collect(Float32, 1:nin2*4*4), 4, 4, nin2, 1)
            in1 = inputvertex("in1", nin1, FluxConv{2}())
            in2 = inputvertex("in2", nin2, FluxConv{2}())
            vfun(v,s) = mutable(BatchNorm(nout(v)), v)

            @test size(testgraph_vfun(vfun, in1, in2)(indata1, indata2)) == (4,4,10,1)
        end

        @testset "Concatentate dimension mismatch fail" begin
            d1 = mutable(Dense(2,3), inputvertex("in1", 2))
            c1 = mutable(Conv((3,3), 4=>5), inputvertex("in2", 4))
            r1 = mutable(RNN(6,7), inputvertex("in3", 6))
            @test_throws DimensionMismatch concat(d1, c1)
            @test_throws DimensionMismatch concat(r1, c1)
            @test_throws DimensionMismatch concat(d1, r1)
        end

        @testset "Concat with name" begin
            d1 = mutable(Dense(2,3), inputvertex("in1", 2))
            d2 = mutable(Dense(2,5), inputvertex("in1", 2))
            c = concat("c", d1, d2)

            @test name(c) == "c"
            @test nout(c) == 8
            @test nin(c) == [3, 5]
        end
    end

    @testset "Tricky structures" begin

        mutable struct Probe
            activation
        end
        function (p::Probe)(x)
            p.activation = x
            return x
        end

        function probe(in)
            p = Probe(nothing)
            return invariantvertex(NoParams(p), in), p
        end

        conv3x3(inpt::AbstractVertex, nch::Integer) = mutable(Conv((3,3), nout(inpt)=>nch, pad=(1,1)), inpt)
        batchnorm(inpt) = mutable(BatchNorm(nout(inpt)), inpt)
        mmaxpool(inpt) = mutable(MaxPool((2,2)), inpt)

        @testset "Residual Conv block" begin
            inpt = inputvertex("in", 3)
            conv1 = conv3x3(inpt, 5)
            pv1, p1 = probe(conv1)
            bn1 = batchnorm(pv1)
            conv2 = conv3x3(bn1, 5)
            pv2, p2 = probe(conv2)
            bn2 = batchnorm(pv2)
            add = bn2 + bn1
            mp = mmaxpool(add)
            out = conv3x3(mp, 3)

            graph = CompGraph([inpt], [out])

            # Test that the graph works to begin with
            indata = reshape(collect(Float32, 1:3*4*4), 4, 4, 3, 1)
            @test size(graph(indata)) == (2, 2, 3, 1)
            @test size(p1.activation) == (4, 4, 5, 1)
            @test size(p2.activation) == (4, 4, 5, 1)

            Δnin!(v -> 1:nout(v), out => -1)

            @test size(graph(indata)) == (2, 2, 3, 1)
            @test size(p1.activation) == (4, 4, 4, 1)
            @test size(p2.activation) == (4, 4, 4, 1)
        end

        @testset "Residual fork Conv block" begin
            inpt = inputvertex("in", 3)
            conv1 = conv3x3(inpt, 5)
            pv1, p1 = probe(conv1)
            bn1 = batchnorm(pv1)
            conv1a = conv3x3(bn1, 3)
            pv1a, p1a = probe(conv1a)
            bn1a = batchnorm(pv1a)
            conv1b = conv3x3(bn1, 2)
            pv1b, p1b = probe(conv1b)
            bn1b = batchnorm(pv1b)
            mv = concat(bn1a, bn1b)
            add = mv + bn1
            mp = mmaxpool(add)
            out = conv3x3(mp, 3)

            graph = CompGraph([inpt], [out])

            # Test that the graph works to begin with
            indata = reshape(collect(Float32, 1:3*4*4), 4, 4, 3, 1)
            @test size(graph(indata)) == (2, 2, 3, 1)
            @test size(p1.activation) == (4, 4, 5, 1)
            @test size(p1a.activation) == (4, 4, 3, 1)
            @test size(p1b.activation) == (4, 4, 2, 1)

            Δnin!(v -> 1:nout(v), out => -1)

            @test size(graph(indata)) == (2, 2, 3, 1)
            @test size(p1.activation) == (4, 4, 4, 1)
            @test size(p1a.activation) == (4, 4, 2, 1)
            @test size(p1b.activation) == (4, 4, 2, 1)
        end

        rnnvertex(inpt, outsize) = mutable("rnn", RNN(nout(inpt), outsize), inpt)
        densevertex(inpt, outsize) = mutable("dense", Dense(nout(inpt), outsize), inpt)

        @testset "RNN to Dense" begin
            inpt = inputvertex("in", 4)
            rnn = rnnvertex(inpt, 5)
            pv, p = probe(rnn)
            dnn = densevertex(pv, 3)

            graph = CompGraph([inpt], [dnn])

            indata = [collect(Float32, 1:nin(rnn)[]) for i =1:10]
            @test size(hcat(graph.(indata)...)) == (3,10)
            @test size(p.activation) == (5,)

            Δnin!(v -> 1:nout(v), dnn, 1)

            @test size(hcat(graph.(indata)...)) == (3,10)
            @test size(p.activation) == (6,)

            Δnout!(v -> 1:nout(v), rnn, -2)

            @test size(hcat(graph.(indata)...)) == (3,10)
            @test size(p.activation) == (4,)
        end
    end
end

@testset "Trait functions" begin
    @test named("test")(SizeAbsorb()) == NamedTrait(SizeAbsorb(), "test")
    @test validated()(SizeAbsorb()) == SizeChangeValidation(SizeAbsorb())
    @test logged(level=Base.CoreLogging.Info, info=NameInfoStr())(SizeAbsorb()) == SizeChangeLogger(Base.CoreLogging.Info, NameInfoStr(), SizeAbsorb())
end

@testset "Flux functor" begin
    import Flux:functor
    import NaiveNASflux: weights, bias
    inpt = inputvertex("in", 2, FluxDense())
    v1 = mutable(Dense(2, 3), inpt)
    v2 = mutable(Dense(3, 4), v1)
    g1 = CompGraph(inpt, v2)

    @test functor(g1)[1] == (inpt, v1, v2)

    pars1 = params(g1).order
    @test pars1[1] == weights(layer(v1))
    @test pars1[2] == bias(layer(v1))
    @test pars1[3] == weights(layer(v2))
    @test pars1[4] == bias(layer(v2))

    g2 = copy(g1)
    # Basically what Flux.gpu does except function is CuArrays.cu(x) instead of 2 .* x
    testfun(x) = x
    testfun(x::AbstractArray) = 2 .* x
    fmap(testfun, g2)

    pars2 = params(g2).order.data
    @test pars2 == 2 .* pars1
end

@testset "Trainable insert values" begin
    using Random
    import NaiveNASflux: weights, bias

    @testset "Dense-Dense-Dense" begin
        Random.seed!(0)
        iv = inputvertex("in", 3, FluxDense())
        v1 = mutable("v1", Dense(3,3), iv)
        v2 = mutable("v2", Dense(3,4), v1)
        v3 = mutable("v3", Dense(4,2), v2)

        g = CompGraph(iv, v3)

        indata = randn(3,4)
        expectedout = g(indata)

        Δnout!(v -> 1, v1 => 2)
        NaiveNASflux.forcemutation(g)

        @test g(indata) ≈ expectedout

        Flux.train!((x,y) -> Flux.mse(g(x), y), params(g), [(randn(nin(v1)[],8), randn(nout(v3) ,8))], Descent(0.5))

        @test minimum(abs.(weights(layer(v1)))) > 0
        @test minimum(abs.(weights(layer(v2)))) > 0
        @test minimum(abs.(weights(layer(v3)))) > 0
    end

    @testset "Conv-Bn-Conv" begin
        Random.seed!(0)
        iv = inputvertex("in", 2, FluxConv{2}())
        v1 = mutable("v1", Conv((1,1), 2 => 2), iv)
        v2 = mutable("v2", BatchNorm(2), v1)
        v3 = mutable("v3", Conv((1,1), 2 => 2), v2)

        g = CompGraph(iv, v3)

        indata = randn(Float32, 2,2,2,8)
        expectedout = g(indata)

        Δnout!(v -> 1, v1 => 2)
        NaiveNASflux.forcemutation(g)

        @test g(indata) == expectedout

        Flux.train!((x,y) -> Flux.mse(g(x), y), params(g), [(randn(Float32,2,2,2,8), randn(Float32,2,2,2,8))], Descent(0.5))

        @test minimum(abs.(weights(layer(v1)))) > 0
        @test minimum(abs.(weights(layer(v3)))) > 0
    end

    @testset "Conv-Conv-Conv" begin
        Random.seed!(0)
        iv = inputvertex("in", 2, FluxConv{2}())
        v1 = mutable("v1", Conv((1,1), 2 => 2), iv; layerfun=ActivationContribution ∘ LazyMutable)
        v2 = mutable("v2", Conv((1,1), 2 => 2), v1; layerfun=ActivationContribution ∘ LazyMutable)
        v3 = mutable("v3", Conv((1,1), 2 => 2), v2; layerfun=ActivationContribution ∘ LazyMutable)

        g = CompGraph(iv, v3)

        indata = randn(Float32, 2,2,2,8)
        expectedout = g(indata)

        Δnout!(v->1, v1 => 2, v2 => 1)
        NaiveNASflux.forcemutation(g)

        @test g(indata) == expectedout

        Flux.train!((x,y) -> Flux.mse(g(x), y), params(g), [(randn(Float32,2,2,2,8), randn(Float32,2,2,2,8))], Descent(0.5))

        @test minimum(abs.(weights(layer(v1)))) > 0
        @test minimum(abs.(weights(layer(v2)))) > 0
        @test minimum(abs.(weights(layer(v3)))) > 0
    end
end
