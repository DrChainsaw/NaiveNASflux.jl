import NaiveNASflux
import NaiveNASflux: AbstractMutableComp, MutableLayer, LazyMutable, weights, bias, select, layer, mutate, hiddenweights, hiddenstate, state, outscale
using Flux
import Flux: mapchildren
using NaiveNASlib
import InteractiveUtils:subtypes

@testset "Mutable computation" begin

    @testset "Select parameters" begin
        mat = reshape(collect(1:3*4), 3, 4)

        @test select(mat, 2 => [2,-1,-1,4]) == [4 0 0 10;5 0 0 11;6 0 0 12]
        @test select(mat, 1 => [-1,1,3]) == [0 0 0 0;1 4 7 10;3 6 9 12]

        @test select(mat, 1 => [2,-1,3,-1], 2 => [-1,1,-1,4]) == [0 2 0 11;0 0 0 0;0 3 0 12;0 0 0 0]
    end

    @testset "Dense MutableLayer" begin

        m = MutableLayer(Dense(2,3))

        @test nin(m) == nin(m.layer) == 2
        @test nout(m) == nout(m.layer) == 3
        @test minΔninfactor(m) == 1
        @test minΔnoutfactor(m) == 1
        @test m([1.0, 2.0]) == m.layer([1.0, 2.0])

        m.layer = Dense(3,4)
        bias(m.layer)[1:end] = 1:4
        @test nin(m) == nin(m.layer) == 3
        @test nout(m) == nout(m.layer) == 4
        @test m([1.0, 2.0, 3.0]) == m.layer([1.0, 2.0, 3.0])

        inds = [1,3]
        Wexp, bexp = weights(m.layer)[:, inds], bias(m.layer)
        mutate_inputs(m, inds)
        assertlayer(m.layer, Wexp, bexp)

        inds = [1,2,4]
        Wexp, bexp = weights(m.layer)[inds, :], bias(m.layer)[inds]
        mutate_outputs(m, inds)
        assertlayer(m.layer, Wexp, bexp)

        inds = [1,-1, 2]
        Wexp = hcat(weights(m.layer)[:, 1], zeros(Float32, 3), weights(m.layer)[:, 2])
        mutate_inputs(m, inds)
        assertlayer(m.layer, Wexp, bexp)

        inds = [-1, 1, -1, 3, -1]
        Wexp = permutedims(hcat(zeros(Float32, 3), weights(m.layer)[1, :], zeros(Float32, 3), weights(m.layer)[3, :], zeros(Float32,3)))
        bexp = Float32[0, bias(m.layer)[1], 0, bias(m.layer)[3], 0]
        mutate_outputs(m, inds)
        assertlayer(m.layer, Wexp, bexp)
    end
    @testset "Convolutional layers" begin

        @testset "Conv MutableLayer" begin
            m = MutableLayer(Conv((2,3),(4=>5)))

            @test nin(m) == nin(m.layer) == 4
            @test nout(m) == nout(m.layer) == 5
            @test minΔninfactor(m) == 1
            @test minΔnoutfactor(m) == 1
            input = reshape(collect(Float32, 1:3*4*4), 3, 4, 4, 1)
            @test m(input) == m.layer(input)

            inds = [1,3]
            Wexp, bexp = weights(m.layer)[:,:,inds,:], bias(m.layer)
            mutate_inputs(m, inds)
            assertlayer(m.layer, Wexp, bexp)

            inds = [1,2,4]
            Wexp, bexp = weights(m.layer)[:,:,:,inds], bias(m.layer)[inds]
            mutate_outputs(m, inds)
            assertlayer(m.layer, Wexp, bexp)

            inds = [1,-1, 2]
            wsize = deleteat!(collect(size(weights(m.layer))), 3)

            # Nothing beats working in four dimensions...
            # Stack 3D arrays in a 4:th dimension and then swap dim 3 and 4
            Wexp = permutedims(cat(
            weights(m.layer)[:,:,1,:],
            zeros(Float32, wsize...),
            weights(m.layer)[:,:,2,:], dims=4), [1,2,4,3])

            mutate_inputs(m, inds)
            assertlayer(m.layer, Wexp, bexp)

            inds = [-1, 1, -1, 3, -1]
            wsize = deleteat!(collect(size(weights(m.layer))), 4)

            Wexp = cat(
            zeros(Float32, wsize...),
            weights(m.layer)[:,:,:,1],
            zeros(Float32, wsize...),
            weights(m.layer)[:,:,:,3],
            zeros(Float32,wsize...), dims=4)

            bexp = Float32[0, bias(m.layer)[1], 0, bias(m.layer)[3], 0]
            mutate_outputs(m, inds)
            assertlayer(m.layer, Wexp, bexp)
        end

        @testset "ConvTranspose MutableLayer" begin
            m = MutableLayer(ConvTranspose((2,3),(4=>5)))

            @test nin(m) == nin(m.layer) == 4
            @test nout(m) == nout(m.layer) == 5
            @test minΔninfactor(m) == 1
            @test minΔnoutfactor(m) == 1
            input = reshape(collect(Float32, 1:3*4*4), 3, 4, 4, 1)
            @test m(input) == m.layer(input)

            inputs = [1,3]
            outputs = [1,2,4]
            Wexp, bexp = weights(m.layer)[:,:,outputs,inputs], bias(m.layer)[outputs]
            mutate(m, inputs=inputs, outputs=outputs)
            assertlayer(m.layer, Wexp, bexp)
        end

        @testset "DepthwiseConv MutableLayer" begin
            m = MutableLayer(DepthwiseConv((2,2),(3=>6)))

            @test nin(m) == nin(m.layer) == 3
            @test nout(m) == nout(m.layer) == 6
            @test_throws ErrorException minΔninfactor(m)
            @test_throws ErrorException minΔnoutfactor(m)
            input = reshape(collect(Float32, 1:3*3*3), 3, 3, 3, 1)
            @test m(input) == m.layer(input)

            inputs = [1, 3]
            outputs = [1, 2, 4, 5]
            Wexp, bexp = weights(m.layer)[:,:,outputs,inputs], bias(m.layer)[outputs]
            mutate(m, inputs=inputs, outputs=outputs)
            assertlayer(m.layer, Wexp, bexp)
        end
    end

    @testset "Diagonal MutableLayer" begin
        m = MutableLayer(Flux.Diagonal(4))

        @test nin(m) == nin(m.layer) == nout(m) == nout(m.layer) == 4
        @test minΔninfactor(m) == 1
        @test minΔnoutfactor(m) == 1
        weights(m.layer)[1:end] = 1:4
        bias(m.layer)[1:end] = 1:4

        @test m(Float32[1,2,3,4]) == m.layer(Float32[1,2,3,4])

        inds = [1,3]
        Wexp, bexp = weights(m.layer)[inds], bias(m.layer)[inds]
        mutate_inputs(m, inds)
        assertlayer(m.layer, Wexp, bexp)

        inds = [-1, 2, -1]
        Wexp, bexp = Float32[0, weights(m.layer)[2], 0], Float32[0, bias(m.layer)[2], 0]
        mutate_outputs(m, inds)
        assertlayer(m.layer, Wexp, bexp)
    end

    @testset "Normalization layers" begin

        @testset "LayerNorm MutableLayer" begin
            m = MutableLayer(LayerNorm(3))

            @test nin(m) == nin(m.layer) == nout(m) == nout(m.layer) == 3
            @test minΔninfactor(m) == 1
            @test minΔnoutfactor(m) == 1
            weights(m.layer.diag)[1:end] = 1:3
            bias(m.layer.diag)[1:end] = 1:3

            @test m(Float32[1,2,3]) == m.layer(Float32[1,2,3])

            inds = [1,3]
            Wexp, bexp = weights(m.layer.diag)[inds], bias(m.layer.diag)[inds]
            mutate_inputs(m, inds)
            @test typeof(layer(m)) <: LayerNorm
            assertlayer(m.layer.diag, Wexp, bexp)


            inds = [-1, 2, -1]
            Wexp, bexp = Float32[0, weights(m.layer.diag)[2], 0], Float32[0, bias(m.layer.diag)[2], 0]
            mutate_outputs(m, inds)
            @test typeof(layer(m)) <: LayerNorm
            assertlayer(m.layer.diag, Wexp, bexp)
        end

        array(arr) = arr
        array(arr::TrackedArray) = arr.data
        setpar(::Any, x) = x
        setpar(::TrackedArray, x) = param(x)

        function assertnorm(l, meanexp, varexp)
            @test l.β.data == meanexp
            @test l.γ.data == varexp
            @test l.μ == meanexp
            @test l.σ² == varexp
        end

        @testset "$l MutableLayer" for l in (BatchNorm, InstanceNorm, n -> GroupNorm(n,n))
            m = MutableLayer(l(5))
            l_orig = layer(m)

            @test nin(m) == nin(m.layer) == nout(m) == nout(m.layer) == 5
            @test minΔninfactor(m) == 1
            @test minΔnoutfactor(m) == 1
            m.layer = mapchildren(par -> setpar(par, collect(Float32, 1:5)), layer(m))

            inds = [1,3,4]
            expall = Float32.(inds)
            mutate_inputs(m, inds)
            assertnorm(m.layer, inds, inds)

            mutate_outputs(m, [-1, 2, -1])
            assertnorm(m.layer, [0, 3, 0], [1, 3, 1])
        end

        @testset "GroupNorm MutableLayer with groups" begin

            #Test with groups of 2
            m = MutableLayer(GroupNorm(6,3))
            m.layer = mapchildren(par -> setpar(par, collect(Float32, 1:length(par))), layer(m))
            mutate_inputs(m, [1,2,5,6])
            @test layer(m).μ == [1, 3]
            @test layer(m).σ² == [1, 3]

            # Now when dimensions don't add up: size 8 becomes size 9
            m = MutableLayer(GroupNorm(8,4))
            mutate_inputs(m, [1,3,-1,-1,4,-1,7,-1,8])
            # Current alg for selecting which group to pick in this case is poor, don't wanna test it :)
            @test length(layer(m).μ) == 3
            @test length(layer(m).σ²) == 3

        end
    end

    @testset "Recurrent layers" begin

        function assertrecurrent(l, Wiexp, Whexp, bexp, hexp, sexp)
            assertlayer(l, Wiexp, bexp)
            @test hiddenweights(l) == Whexp
            @test hiddenstate(l) == hexp
            @test state(l) == sexp
        end

        function setparsrnn(l)
            bias(l)[1:end] = collect(Float32, 1:nout(l)*outscale(l))
            hiddenstate(l)[1:end] = collect(Float32, 1:nout(l))
            state(l)[1:end] = collect(Float32, 1:nout(l))
        end

        function setparslstm(l)
            bias(l)[1:end] = collect(Float32, 1:nout(l)*outscale(l))
            foreach(h -> h[1:end] = collect(Float32, 1:nout(l)), hiddenstate(l))
            foreach(h -> h[1:end] = collect(Float32, 1:nout(l)), state(l))
        end

        @testset "RNN MutableLayer" begin
            m = MutableLayer(RNN(3, 4))
            setparsrnn(layer(m))

            @test nin(m) == nin(m.layer) == 3
            @test nout(m) == nout(m.layer) == 4
            @test minΔninfactor(m) == 1
            @test minΔnoutfactor(m) == 1

            inds = [1, 3]
            Wiexp = weights(layer(m))[:, inds]
            Whexp = copy(hiddenweights(layer(m)))
            bexp = copy(bias(layer(m)))
            hexp = copy(hiddenstate(layer(m)))
            sexp = copy(state(layer(m)))
            mutate_inputs(m, inds)
            assertrecurrent(layer(m), Wiexp, Whexp, bexp, hexp, sexp)

            inds = [1,-1, 2]
            Wiexp = permutedims(hcat(weights(layer(m))[1, :], zeros(Float32, 2), weights(layer(m))[2, :]))
            wh = hiddenweights(layer(m))
            Whexp = [wh[1, 1] 0 wh[1, 2]; zeros(Float32, 1, 3); wh[2, 1] 0 wh[2, 2]]
            bexp = Float32[bias(layer(m))[1], 0, bias(layer(m))[2]]
            hexp = Float32[hiddenstate(layer(m))[1], 0, hiddenstate(layer(m))[2]]
            sexp = Float32[state(layer(m))[1], 0, state(layer(m))[2]]
            mutate_outputs(m, inds)
            assertrecurrent(layer(m), Wiexp, Whexp, bexp, hexp, sexp)

            #Sanity check that the layer still seems to work after mutation
            output = m(reshape(collect(Float32, 1:2*10), 2,10))
            @test size(output) == (3, 10)
            @test isnan.(output) == falses(size(output))
        end

        @testset "LSTM MutableLayer" begin
            m = MutableLayer(LSTM(3, 4))
            setparslstm(layer(m))

            @test nin(m) == nin(m.layer) == 3
            @test nout(m) == nout(m.layer) == 4
            @test minΔninfactor(m) == 1
            @test minΔnoutfactor(m) == 1

            inds = [1, 3]
            Wiexp = weights(layer(m))[:, inds]
            Whexp = copy(hiddenweights(layer(m)))
            bexp = copy(bias(layer(m)))
            hexp = copy(hiddenstate(layer(m)))
            sexp = copy(state(layer(m)))
            mutate_inputs(m, inds)
            assertrecurrent(layer(m), Wiexp, Whexp, bexp, hexp, sexp)

            inds = [1,-1, 2]
            wi = weights(layer(m))
            scalerange = (0:outscale(layer(m))-1) .* nout(layer(m))
            Wiexp = permutedims(mapfoldl(offs -> hcat(wi[1+offs, :], zeros(Float32, 2), wi[2+offs, :]), hcat, scalerange))
            wh = hiddenweights(layer(m))
            Whexp = mapfoldl(offs -> [wh[1+offs, 1] 0 wh[1+offs, 2]; zeros(Float32, 1, 3); wh[2+offs, 1] 0 wh[2+offs, 2]], vcat, scalerange)
            bexp = mapfoldl(offs -> Float32[bias(layer(m))[1+offs], 0, bias(layer(m))[2+offs]], vcat, scalerange)
            hexp = map(hs -> Float32[hs[1], 0, hs[2]], hiddenstate(layer(m)))
            sexp = map(hs -> Float32[hs[1], 0, hs[2]], state(layer(m)))
            mutate_outputs(m, inds)
            assertrecurrent(layer(m), Wiexp, Whexp, bexp, hexp, sexp)

            #Sanity check that the layer still seems to work after mutation
            output = m(reshape(collect(Float32, 1:2*10), 2,10))
            @test size(output) == (3, 10)
            @test isnan.(output) == falses(size(output))
        end

        @testset "GRU MutableLayer" begin
            m = MutableLayer(GRU(3, 4))
            setparsrnn(layer(m))

            @test nin(m) == nin(m.layer) == 3
            @test nout(m) == nout(m.layer) == 4
            @test minΔninfactor(m) == 1
            @test minΔnoutfactor(m) == 1

            inds = [1, 3]
            Wiexp = weights(layer(m))[:, inds]
            Whexp = copy(hiddenweights(layer(m)))
            bexp = copy(bias(layer(m)))
            hexp = copy(hiddenstate(layer(m)))
            sexp = copy(state(layer(m)))
            mutate_inputs(m, inds)
            assertrecurrent(layer(m), Wiexp, Whexp, bexp, hexp, sexp)

            inds = [1,-1, 2]
            wi = weights(layer(m))
            scalerange = (0:outscale(layer(m))-1) .* nout(layer(m))
            Wiexp = permutedims(mapfoldl(offs -> hcat(wi[1+offs, :], zeros(Float32, 2), wi[2+offs, :]), hcat, scalerange))
            wh = hiddenweights(layer(m))
            Whexp = mapfoldl(offs -> [wh[1+offs, 1] 0 wh[1+offs, 2]; zeros(Float32, 1, 3); wh[2+offs, 1] 0 wh[2+offs, 2]], vcat, scalerange)
            bexp = mapfoldl(offs -> Float32[bias(layer(m))[1+offs], 0, bias(layer(m))[2+offs]], vcat, scalerange)
            hexp = Float32[hiddenstate(layer(m))[1], 0, hiddenstate(layer(m))[2]]
            sexp = Float32[state(layer(m))[1], 0, state(layer(m))[2]]
            mutate_outputs(m, inds)
            assertrecurrent(layer(m), Wiexp, Whexp, bexp, hexp, sexp)

            #Sanity check that the layer still seems to work after mutation
            output = m(reshape(collect(Float32, 1:2*10), 2,10))
            @test size(output) == (3, 10)
            @test isnan.(output) == falses(size(output))
        end
    end

    @testset "Clone MutableLayer" begin
            m = MutableLayer(Dense(2,3))
            cloned = clone(m)
            @test layer(cloned) !== layer(m)
            @test cloned([1, 2]) == m([1, 2])
    end

    @testset "LazyMutable" begin
        @testset "LazyMutable Dense factory" begin

            struct DenseFactory end
            function NaiveNASflux.dispatch!(m::LazyMutable, ::DenseFactory, x)
                m.mutable = MutableLayer(Dense(nin(m), nout(m)))
                return m(x)
            end
            m = LazyMutable(DenseFactory(), 2, 3)

            @test typeof(m.mutable) == DenseFactory
            expected = m(Float32[2,3])

            @test typeof(m.mutable) == MutableLayer
            @test m(Float32[2,3]) == expected

            #Now mutate before create
            m = LazyMutable(DenseFactory(), 2, 3)

            mutate_inputs(m, 5)
            mutate_outputs(m, 4)

            #No eagerness allowed :)
            @test m.mutable != MutableLayer
            expected = m(Float32[0,1,2,3,4])

            @test typeof(m.mutable) == MutableLayer
            @test m(Float32[0,1,2,3,4]) == expected
        end

        @testset "LazyMutable with Dense MutableLayer" begin
            m = MutableLayer(Dense(3,4))
            mlazy = LazyMutable(m)

            Wexp = weights(layer(m))
            bexp = bias(layer(m))

            @test minΔninfactor(m) == 1
            @test minΔnoutfactor(m) == 1

            mutate_inputs(mlazy, [1, 3])
            assertlayer(layer(m), Wexp, bexp)

            mutate_outputs(mlazy, [2, 4, -1])
            assertlayer(layer(m), Wexp, bexp)

            @test layer(m) == layer(mlazy)
            @test layertype(m) == layertype(mlazy)

            expected = mlazy(Float32[2,3])

            @test nin(mlazy) == nin(m) == 2
            @test nout(mlazy) == nout(m) == 3

            @test expected == m(Float32[2,3])
        end

        @testset "LazyMutable reselect" begin
            m = LazyMutable(MutableLayer(Dense(5,5)))

            mutate_inputs(m, [-1, 1, 3, -1, 4])
            @test m.inputs == [-1, 1, 3, -1, 4]

            mutate_inputs(m, [1,2,4,5])
            @test m.inputs == [-1, 1,-1, 4]

            mutate_outputs(m, [2, -1, 3, -1, 4])
            @test m.outputs == [2, -1, 3, -1, 4]

            mutate_outputs(m, [1, 2, 5,-1])
            @test m.outputs == [2, -1, 4, -1]

            @test m(Float32[1,3,5,7]) == layer(m)(Float32[1,3,5,7])
        end

        @testset "Clone" begin
            mlazy = LazyMutable(MutableLayer(Dense(2,3)))
            cloned = clone(mlazy)
            @test layer(cloned) !== layer(mlazy)
            @test cloned([1, 2]) == mlazy([1, 2])
        end

        @testset "Treelike" begin
            m = LazyMutable(MutableLayer(Dense(2,3)))
            visitfun(x) = x
            visitdense = false
            function visitfun(l::TrackedArray)
                visitdense = true
                return l
            end

            mutate_inputs(m, [-1, -1, 1, 2])

            Flux.mapleaves(visitfun, m)
            @test visitdense
        end
    end
end
