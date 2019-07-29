
@testset "Neuron value tests" begin

    ml(l, lfun=LazyMutable; insize=nin(l)[]) = mutable(l, inputvertex("in", insize, layertype(l)), layerfun = lfun)

    function tr(l, data)
        outshape = collect(size(data))
        outshape[[actdim(ndims(data))]] .= nout(l)
        example = [(data, ones(Float32, outshape...))];
        Flux.train!((x,y) -> Flux.mse(l(x), y), params(l), example, Descent(0.1))
    end

    @testset "Neuron value Dense default" begin
        l = ml(Dense(3,5))
        @test size(neuron_value(l)) == (5,)
    end

    @testset "Neuron value RNN default" begin
        l = ml(RNN(3,5))
        @test size(neuron_value(l)) == (5,)
    end

    @testset "Neuron value Conv default" begin
        l = ml(Conv((2,3), 4=>5))
        @test size(neuron_value(l)) == (5,)
    end

    @testset "Neuron value Dense act contrib" begin
        l = ml(Dense(3,5), ActivationContribution)
        @test neuron_value(l) == zeros(5)
        tr(l, ones(Float32, 3, 4))
        @test size(neuron_value(l)) == (5,)
    end

    @testset "Neuron value RNN act contrib" begin
        l = ml(RNN(3,5), ActivationContribution)
        @test neuron_value(l) == zeros(5)
        tr(l, ones(Float32, 3, 8))
        @test size(neuron_value(l)) == (5,)
    end

    @testset "Neuron value Conv act contrib" begin
        l = ml(Conv((3,3), 2=>5, pad=(1,1)), ActivationContribution)
        @test neuron_value(l) == zeros(5)
        tr(l, ones(Float32, 4,4,2,5))
        @test size(neuron_value(l)) == (5,)
    end

    @testset "Neuron value MaxPool act contrib" begin
        l = ml(MaxPool((3,3)), ActivationContribution, insize=2)
        @test ismissing(neuron_value(l))
        l(ones(Float32, 4,4,2,5))
        @test size(neuron_value(l)) == (2,)
    end

    @testset "Mutate ActivationContribution" begin
        l = ml(Dense(3,5), ActivationContribution ∘ LazyMutable)
        Δnout(l, [1,2,3,-1])
        apply_mutation(l)
        @test size(l(ones(Float32, 3,2))) == (4, 2)
        @test size(neuron_value(l)) == (4,)
    end

    @testset "Mutate ActivationContribution MaxPool" begin
        l1 = ml(Conv((3,3), 2=>5, pad=(1,1)), ActivationContribution ∘ LazyMutable)
        l2 = mutable(MaxPool((3,3), pad=(1,1)), l1, layerfun = ActivationContribution ∘ LazyMutable)
        g = CompGraph(inputs(l1), l2)

        # Mutate before activation contribution for l2 has been initialized
        Δnout(l1, [1,2,3,4,-1,-1])
        apply_mutation(g)
        @test size(neuron_value(l1)) == (6,)
        @test ismissing(neuron_value(l2))

        # This will initialize it
        @test size(g(ones(Float32, 4,4,2,3))) == (2,2,6,3)

        Δnout(l1, [1,2,3,4])
        apply_mutation(g)
        @test size(g(ones(Float32, 4,4,2,3)))  == (2,2,4,3)
        @test size(neuron_value(l1)) == size(neuron_value(l2)) == (4,)

        Δnin(l2, [1,2, -1])
        apply_mutation(g)
        @test size(g(ones(Float32, 4,4,2,3)))  == (2,2,3,3)
        @test size(neuron_value(l1)) == size(neuron_value(l2)) == (3,)

    end
end
