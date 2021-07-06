
@testset "Neuron value tests" begin
    import NaiveNASflux: neuron_value_safe

    ml(l, lfun=LazyMutable; insize=nin(l)[]) = mutable(l, inputvertex("in", insize, layertype(l)), layerfun = lfun)

    function tr(l, data; loss=Flux.mse)
        outshape = collect(size(data))
        outshape[[actdim(ndims(data))]] .= nout(l)
        example = [(data, ones(Float32, outshape...))];
        Flux.train!((x,y) -> loss(l(x), y), params(l), example, Descent(0.1))
    end

    function tr(l, output, inputs...)
        example = [(inputs, output)];
        Flux.train!((x,y) -> Flux.mse(l(x...), y), params(l), example, Descent(0.1))
    end


    @testset "Utils" begin
        actonly(curr, act, grad) = act

        @testset "Ewma" begin
            m = Ewma(0.3, actonly)
            @test m(missing, [1,2,3,4], :ignored) == [1,2,3,4]
            @test m([1,2,3,4], [5,6,7,8], :ignored) ≈ [3.8, 4.8, 5.8, 6.8]
        end

        @testset "NeuronValueEvery{3}" begin
            m = NeuronValueEvery(3, actonly)
            @test m(:old, :new, :ignored) == :new
            @test m(:old, :new, :ignored) == :old
            @test m(:old, :new, :ignored) == :old
            @test m(:old, :new, :ignored) == :new
            @test m(:old, :new, :ignored) == :old
            @test m(:old, :new, :ignored) == :old
            @test m(:old, :new, :ignored) == :new
            @test m(:old, :new, :ignored) == :old
            @test m(:old, :new, :ignored) == :old
        end
    end

    @testset "Neuron value Dense default" begin
        l = ml(Dense(3,5))
        @test size(neuron_value(l)) == (5,)
        Δnout!(v -> 1, l => 3)
        @test size(neuron_value(l)) == (8,)
        Δnout!(v -> 1, l => -4)
        @test size(neuron_value(l)) == (4,)
        @test neuron_value(l) ≈ neuron_value_safe(l)
    end

    @testset "Neuron value Dense default no bias" begin
        l = ml(Dense(ones(5, 3), Flux.Zeros()))
        @test size(neuron_value(l)) == (5,)
        @test neuron_value(l) ≈ neuron_value_safe(l)
    end

    @testset "Neuron value RNN default" begin
        l = ml(RNN(3,5))
        @test size(neuron_value(l)) == (5,)
        @test neuron_value(l) ≈ neuron_value_safe(l)
    end

    @testset "Neuron value Conv default" begin
        l = ml(Conv((2,3), 4=>5))
        @test size(neuron_value(l)) == (5,)
        @test neuron_value(l) ≈ neuron_value_safe(l)
    end

    @testset "Neuron value unkown default" begin
        l = ml(MeanPool((2,2)); insize = 3)
        @test ismissing(neuron_value(l))
        @test neuron_value_safe(l) == ones(nout(l)) 
    end

    @testset "ActivationContribution no grad" begin
        f(x) = 2 .* x .^ 2
        Flux.Zygote.@nograd f
        l = ml(Dense(2,3), ActivationContribution)
        @test neuron_value(l) == zeros(3)
        tr(l, ones(Float32, 2, 1), loss = f ∘ Flux.mse)
        @test neuron_value(l) == zeros(3)
        @test length(params(l)) == length(params(layer(l)))
    end

    @testset "Neuron value Dense act contrib" begin
        l = ml(Dense(3,5), ActivationContribution)
        @test neuron_value(l) == zeros(5)
        tr(l, ones(Float32, 3, 4))
        @test size(neuron_value(l)) == (5,)
        @test length(params(l)) == length(params(layer(l)))
    end

    @testset "Neuron value Dense act contrib every 4" begin
        l = ml(Dense(3,5), l -> ActivationContribution(l, NeuronValueEvery(4)))
        @test neuron_value(l) == zeros(5)
        nvprev = copy(neuron_value(l))
        tr(l, ones(Float32, 3, 4))
        @test neuron_value(l) != nvprev
        nvprev = copy(neuron_value(l))

        tr(l, ones(Float32, 3, 4))
        @test nvprev == neuron_value(l)

        tr(l, ones(Float32, 3, 4))
        tr(l, ones(Float32, 3, 4))
        tr(l, ones(Float32, 3, 4))
        @test nvprev != neuron_value(l)
    end

    @testset "Neuron value RNN act contrib" begin
        l = ml(RNN(3,5), ActivationContribution)
        @test neuron_value(l) == zeros(5)
        tr(l, ones(Float32, 3, 8))
        @test size(neuron_value(l)) == (5,)
        @test length(params(l)) == length(params(layer(l)))
    end

    @testset "Neuron value Conv act contrib" begin
        l = ml(Conv((3,3), 2=>5, pad=(1,1)), ActivationContribution)
        @test neuron_value(l) == zeros(5)
        tr(l, ones(Float32, 4,4,2,5))
        @test size(neuron_value(l)) == (5,)
        @test length(params(l)) == length(params(layer(l)))
    end

    @testset "Neuron value MaxPool act contrib" begin
        l = ml(MaxPool((3,3)), ActivationContribution, insize=2)
        @test ismissing(neuron_value(l))
        tr(l, ones(Float32, 4,4,2,5))

        @test ismissing(minΔninfactor(l))
        @test ismissing(minΔnoutfactor(l))
        @test size(neuron_value(l)) == (2,)
    end

    @testset "Neuron value GlobalMeanPool act contrib" begin
        l = ml(GlobalMeanPool(), ActivationContribution, insize=2)
        @test ismissing(neuron_value(l))
        tr(l, ones(Float32, 4,4,2,5))

        @test ismissing(minΔninfactor(l))
        @test ismissing(minΔnoutfactor(l))
        @test size(neuron_value(l)) == (2,)
    end

    @testset "Elem add ActivationContribution" begin
        ac(l) = ActivationContribution(l)
        v = ac >> ml(Dense(2,3)) + ml(Dense(4,3))
        tr(v, [1 1 1]', [1 2 3]', [4 5 6]')
        @test size(neuron_value(v)) == (3,)

        @test minΔninfactor(v) == 1
        @test minΔnoutfactor(v) == 1

        g = CompGraph(vcat(inputs.(inputs(v))...), v)
        @test size(g(ones(Float32, 2,2), ones(Float32, 4, 2))) == (nout(v), 2)
    end

    @testset "Concat ActivationContribution" begin
        v = concat(ml(Dense(2,3)), ml(Dense(4,5)), layerfun=ActivationContribution)
        tr(v,ones(nout(v), 1), [1 2 3]', [4 5 6 7 8]')
        @test size(neuron_value(v)) == (nout(v),)

        @test minΔninfactor(v) == 1
        @test minΔnoutfactor(v) == 1

        g = CompGraph(vcat(inputs.(inputs(v))...), v)
        @test size(g(ones(Float32, 2,2), ones(Float32, 4, 2))) == (nout(v), 2)
    end

    @testset "Function ActivationContribution" begin
        # Not really an absorbvertex, but ActivationContribution should work on stuff which is not FLux layers.
        v = absorbvertex(ActivationContribution(x -> 2x), 3, ml(Dense(2,3)))
        tr(v, ones(nout(v), 1))
        @test size(neuron_value(v)) == (nout(v),)

        @test minΔninfactor(v) == 1
        @test minΔnoutfactor(v) == 1

        g = CompGraph(vcat(inputs.(inputs(v))...), v)
        @test size(g(ones(Float32, 2,2))) == (nout(v), 2)
    end

    @testset "Mutate ActivationContribution" begin
        l = ml(Dense(3,5), ActivationContribution ∘ LazyMutable)
        Δnout!(l, -1)
        Δoutputs(l, v -> 1:nout_org(v))
        apply_mutation(l)
        @test size(l(ones(Float32, 3,2))) == (4, 2)
        @test size(neuron_value(l)) == (4,)
    end

    @testset "Mutate ActivationContribution MaxPool" begin
        l1 = ml(Conv((3,3), 2=>5, pad=(1,1)), ActivationContribution ∘ LazyMutable)
        l2 = mutable(MaxPool((3,3), pad=(1,1)), l1, layerfun = ActivationContribution ∘ LazyMutable)
        g = CompGraph(inputs(l1), l2)

        # Mutate before activation contribution for l2 has been initialized
        Δnout!(l1, 1)
        Δoutputs(l1, v -> 1:nout_org(v))
        apply_mutation(g)
        @test size(neuron_value(l1)) == (6,)
        @test ismissing(neuron_value(l2))

        # This will initialize it
        @test size(g(ones(Float32, 4,4,2,3))) == (2,2,6,3)

        Δnout!(l1, -2)
        Δoutputs(l1, v -> 1:nout_org(v))
        apply_mutation(g)
        tr(g, ones(Float32, 2,2,4,3), ones(Float32, 4,4,2,3))
        @test size(neuron_value(l1)) == size(neuron_value(l2)) == (4,)

        Δnin!(l2, -1)
        Δoutputs(l1, v -> 1:nout_org(v))
        apply_mutation(g)
        tr(g, ones(Float32, 2,2,3,3), ones(Float32, 4,4,2,3))
        @test size(neuron_value(l1)) == size(neuron_value(l2)) == (3,)
    end
end
