import NaiveNASflux
import NaiveNASflux: AbstractMutableComp, MutableLayer, LazyMutable, weights, bias
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

    @testset "MutableLayer dense" begin

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

    @testset "MutableLayer Conv" begin

        m = MutableLayer(Conv((2,3),(4=>5)))

        @test nin(m) == nin(m.layer) == 4
        @test nout(m) == nout(m.layer) == 5
        input = reshape(collect(Float32, 1:4*12), 3, 4, 4, 1)
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

    @testset "LazyMutable factory dense" begin

        struct DenseFactory <: AbstractMutableComp{Dense} end
        function NaiveNASflux.dispatch(m::LazyMutable, mut::DenseFactory, x)
            m.mutable = MutableLayer{Dense}(Dense(nin(m), nout(m)))
            return m.mutable(x)
        end
        m = LazyMutable(DenseFactory(), 2, 3)

        @test typeof(m.mutable) == DenseFactory
        expected = m(Float32[2,3])

        @test typeof(m.mutable) == MutableLayer{Dense}
        @test m(Float32[2,3]) == expected

        #Now mutate before create
        m = LazyMutable(DenseFactory(), 2, 3)

        mutate_inputs(m, 5)
        mutate_outputs(m, 4)

        @test typeof(m.mutable) == DenseFactory
        expected = m(Float32[0,1,2,3,4])

        @test typeof(m.mutable) == MutableLayer{Dense}
        @test m(Float32[0,1,2,3,4]) == expected
    end

end
