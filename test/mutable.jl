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

    function NaiveNASlib.mutate_inputs(m::LazyMutable, inputs::AbstractArray{<:Integer,1}...)
        @assert length(inputs) == 1 "Only one input per layer!"
        m.inputs = inputs[1]
    end

    function NaiveNASlib.mutate_outputs(m::LazyMutable, outputs::AbstractArray{<:Integer,1})
        m.outputs = outputs
    end

    function NaiveNASlib.mutate_inputs(m::LazyMutable, nin::Integer...)
        mutate_inputs(m, map(n -> collect(1:n), nin)...)
    end

    function NaiveNASlib.mutate_outputs(m::LazyMutable, nout::Integer)
        mutate_outputs(m, 1:nout)
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
