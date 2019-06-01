import NaiveNASflux
import NaiveNASflux: AbstractMutableComp, MutableLayer, LazyMutable, weights, bias, select, layer, mutate
using Flux
using NaiveNASlib
import InteractiveUtils:subtypes

@testset "Mutable computation" begin

    @testset "Method contracts" begin
        for subtype in subtypes(AbstractMutableComp)
            @info "\ttest method contracts for AbstractMutableComp $subtype"
            @test hasmethod(nin, (subtype,))
            @test hasmethod(nout, (subtype,))
        end
    end

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

    @testset "Conv MutableLayer" begin
        m = MutableLayer(Conv((2,3),(4=>5)))

        @test nin(m) == nin(m.layer) == 4
        @test nout(m) == nout(m.layer) == 5
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
        input = reshape(collect(Float32, 1:3*3*3), 3, 3, 3, 1)
        @test m(input) == m.layer(input)

        inputs = [1, 3]
        outputs = [1, 2, 4, 5]
        Wexp, bexp = weights(m.layer)[:,:,outputs,inputs], bias(m.layer)[outputs]
        mutate(m, inputs=inputs, outputs=outputs)
        assertlayer(m.layer, Wexp, bexp)
    end

    @testset "Diagonal MutableLayer" begin
        m = MutableLayer(Flux.Diagonal(4))

        @test nin(m) == nin(m.layer) == nout(m) == nout(m.layer) == 4
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

    @testset "LayerNorm MutableLayer" begin
        m = MutableLayer(LayerNorm(3))

        @test nin(m) == nin(m.layer) == nout(m) == nout(m.layer) == 3
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

        mutate_inputs(mlazy, [1, 3])
        assertlayer(layer(m), Wexp, bexp)

        mutate_outputs(mlazy, [2, 4, -1])
        assertlayer(layer(m), Wexp, bexp)

        expected = mlazy(Float32[2,3])

        @test nin(mlazy) == nin(m) == 2
        @test nout(mlazy) == nout(m) == 3

        @test expected == m(Float32[2,3])
    end

end
