using NaiveNASflux
import NaiveNASflux: weights, bias
using NaiveNASlib
using Flux

@testset "Size mutations" begin

    NaiveNASflux.layer(v::AbstractVertex) = layer(base(v))
    NaiveNASflux.layer(v::CompVertex) = layer(v.computation)

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

        @test [nout(inpt)] == nin(dense1) == [4]
        @test [nout(dense1)] == nin(dense2) == [5]
        @test nout(dense2) == 3

        inds = [1, 2, 4, 5]
        Δnin(dense2, inds)

        @test [nout(inpt)] == nin(dense1) == [4]
        @test [nout(dense1)] == nin(dense2) == [4]
        @test nout(dense2) == 3

        W2exp, b2exp = weights(dl2)[:, inds], bias(dl2)
        W1exp, b1exp = weights(dl1)[inds, :], bias(dl1)[inds]
        apply_mutation.(flatten(dense2))

        @test size(CompGraph([inpt], [dense2])(collect(Float32, 1:nin(inpt)))) == (3,)

        assertlayer(layer(dense2), W2exp, b2exp)
        assertlayer(layer(dense1), W1exp, b1exp)
    end

    @testset "Invariant parametric layer" begin
        inpt = inputvertex("in", 3)
        cv = mutable(Conv((1,1), nout(inpt) => 4), inpt)
        bv = mutable(BatchNorm(nout(cv)), cv)

        @test nin(bv) == [nout(cv)] == [4]

        Δnin(bv, [1,3,4])
        apply_mutation.(flatten(bv))

        @test nin(bv) == [nout(cv)] == [3]
    end

    @testset "Invariant non-parametric layer" begin
        inpt = inputvertex("in", 3)
        cv = mutable(Conv((1,1), nout(inpt) => 4), inpt)
        bv = mutable(MeanPool((2,2)), cv)

        @test nin(bv) == [nout(cv)] == [4]

        Δnin(bv, [1,3,4])
        apply_mutation.(flatten(bv))

        @test nin(bv) == [nout(cv)] == [3]
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

        @testset "Concatenate RNN, GRU and LSTM" begin
            nin1 = 2
            nin2 = 5
            indata1 = reshape(collect(Float32, 1:nin1*4), nin1, 4)
            indata2 = reshape(collect(Float32, 1:nin2*4), nin2, 4)
            for rnntype in [RNN, GRU, LSTM]
                @test size(testgraph(rnntype, nin1, nin2)(indata1, indata2)) == (9,4)
            end
        end

        @testset "Concatenate Conv and ConvTranspose" begin
            nin1 = 2
            nin2 = 5
            indata1 = reshape(collect(Float32, 1:nin1*4*4), 4, 4, nin1, 1)
            indata2 = reshape(collect(Float32, 1:nin2*4*4), 4, 4, nin2, 1)
            for convtype in [Conv, ConvTranspose]
                convfun = (nin,nout) -> convtype((3,3), nin=>nout, pad = (1,1))
                @test size(testgraph(convfun, nin1, nin2)(indata1, indata2)) == (4,4,9,1)
            end
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
            in1 = inputvertex("in1", nin1, FluxConv())
            in2 = inputvertex("in2", nin2, FluxConv())
            vfun(v,s) = mutable(BatchNorm(nout(v)), v)

            @test size(testgraph_vfun(vfun, in1, in2)(indata1, indata2)) == (4,4,10,1)
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
        maxpool(inpt) = mutable(MaxPool((2,2)), inpt)

        @testset "Residual Conv block" begin
            inpt = inputvertex("in", 3)
            conv1 = conv3x3(inpt, 5)
            pv1, p1 = probe(conv1)
            bn1 = batchnorm(pv1)
            conv2 = conv3x3(bn1, 5)
            pv2, p2 = probe(conv2)
            bn2 = batchnorm(pv2)
            add = bn2 + bn1
            mp = maxpool(add)
            out = conv3x3(mp, 3)

            graph = CompGraph([inpt], [out])

            # Test that the graph works to begin with
            indata = reshape(collect(Float32, 1:3*4*4), 4, 4, 3, 1)
            @test size(graph(indata)) == (2, 2, 3, 1)
            @test size(p1.activation) == (4, 4, 5, 1)
            @test size(p2.activation) == (4, 4, 5, 1)

            Δnin(out, [-1, 2, 3, -1])
            apply_mutation(graph)

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
            mv = StackingVertex(CompVertex((x,y) -> cat(x,y,dims=3), bn1a, bn1b))
            add = mv + bn1
            mp = maxpool(add)
            out = conv3x3(mp, 3)

            graph = CompGraph([inpt], [out])

            # Test that the graph works to begin with
            indata = reshape(collect(Float32, 1:3*4*4), 4, 4, 3, 1)
            @test size(graph(indata)) == (2, 2, 3, 1)
            @test size(p1.activation) == (4, 4, 5, 1)
            @test size(p1a.activation) == (4, 4, 3, 1)
            @test size(p1b.activation) == (4, 4, 2, 1)

            Δnin(out, [-1, 2, -1, 4])
            apply_mutation(graph)

            @test size(graph(indata)) == (2, 2, 3, 1)
            @test size(p1.activation) == (4, 4, 4, 1)
            @test size(p1a.activation) == (4, 4, 3, 1)
            @test size(p1b.activation) == (4, 4, 1, 1)
        end

        rnnvertex(inpt, outsize) = mutable(RNN(nout(inpt), outsize), inpt)
        densevertex(inpt, outsize) = mutable(Dense(nout(inpt), outsize), inpt)

        @testset "RNN with last time step to Dense" begin
            inpt = inputvertex("in", 4)
            rnn = rnnvertex(inpt, 5)
            pv, p = probe(rnn)
            lasttimestep = invariantvertex(x -> x[:,end], pv)
            dnn = densevertex(lasttimestep, 3)

            graph = CompGraph([inpt], [dnn])

            indata = reshape(collect(Float32, 1:nin(rnn)[1]*10), nin(rnn)[1],10)
            @test size(graph(indata)) == (3,)
            @test size(p.activation) == (5, 10)


            Δnin(dnn, [1, 2, 3, -1])
            apply_mutation(graph)

            @test size(graph(indata)) == (3,)
            @test size(p.activation) == (4, 10)

            Δnout(rnn, [-1, 1, 3])
            apply_mutation(graph)

            @test size(graph(indata)) == (3,)
            @test size(p.activation) == (3, 10)
        end

    end
end
