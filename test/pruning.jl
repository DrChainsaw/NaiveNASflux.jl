
@testset "Neuron value tests" begin
    import NaiveNASflux: neuron_value_safe, neuron_value

    ml(l, lfun=LazyMutable; insize=nin(l)[]) = fluxvertex(l, inputvertex("in", insize, layertype(l)), layerfun = lfun)

    function tr(l, data; loss=Flux.mse)
        example = [(data, 1)];
        Flux.train!((x,y) -> loss(l(x), y), params(l), example, Descent(0.1))
    end

    function tr(l, output, inputs...)
        example = [(inputs, output)];
        Flux.train!((x,y) -> Flux.mse(l(x...), y), params(l), example, Descent(0.1))
    end


    @testset "Utils" begin
        actonly(curr, act, grad) = act

        @testset "Ewma" begin
            m = NaiveNASflux.Ewma(0.3, actonly)
            @test m(missing, [1,2,3,4], :ignored) == [1,2,3,4]
            @test m([1,2,3,4], [5,6,7,8], :ignored) ≈ [3.8, 4.8, 5.8, 6.8]
            @test m(Float32[1,2,3,4], Float32[5,6,7,8], :ignored) ≈ Float32[3.8, 4.8, 5.8, 6.8]
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
        @test neuron_value_safe(l) == 1 
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

    @testset "Functor and trainable" begin
        import NaiveNASflux: weights, bias
        l = ml(Dense(2,3), ActivationContribution)

        neuron_value_org = neuron_value(l)

        @test params(l) == params(layer(l))
        l2 = fmap(x -> x isa AbstractArray ? fill(17, size(x)) : x, l)
        @test unique(neuron_value(l2)) == unique(bias(layer(l2))) == unique(weights(layer(l2))) == [17]
        @test neuron_value_org === neuron_value(l) == zeros(nout(l))
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
        v = invariantvertex(ActivationContribution(x -> 2x), ml(Dense(2,3)))
        tr(v, ones(nout(v), 1))
        @test size(neuron_value(v)) == (nout(v),)

        @test minΔninfactor(v) == 1
        @test minΔnoutfactor(v) == 1

        g = CompGraph(vcat(inputs.(inputs(v))...), v)
        @test size(g(ones(Float32, 2,2))) == (nout(v), 2)
    end

    @testset "Mutate ActivationContribution" begin
        l = ml(Dense(3,5), ActivationContribution ∘ LazyMutable)
        Δnout!(v -> 1:nout(v), l, -1)
        @test size(l(ones(Float32, 3,2))) == (4, 2)
        @test size(neuron_value(l)) == (4,)
    end

    @testset "Mutate ActivationContribution MaxPool" begin
        l1 = ml(Conv((3,3), 2=>5, pad=(1,1)), ActivationContribution ∘ LazyMutable)
        l2 = fluxvertex(MaxPool((3,3), pad=(1,1)), l1, layerfun = ActivationContribution ∘ LazyMutable)
        g = CompGraph(inputs(l1), l2)

        # Mutate before activation contribution for l2 has been initialized
        Δnout!(v -> 1:nout(v), l1 => 1)
        @test size(neuron_value(l1)) == (6,)
        @test ismissing(neuron_value(l2))

        # This will initialize it
        @test size(g(ones(Float32, 4,4,2,3))) == (2,2,6,3)

        Δnout!(v -> 1:nout(v), l1 => -2)
        tr(g, ones(Float32, 2,2,4,3), ones(Float32, 4,4,2,3))
        @test size(neuron_value(l1)) == size(neuron_value(l2)) == (4,)

        Δnin!(v -> 1:nout(v), l2 => -1)
        tr(g, ones(Float32, 2,2,3,3), ones(Float32, 4,4,2,3))
        @test size(neuron_value(l1)) == size(neuron_value(l2)) == (3,)
    end

    @testset "Add input edge to ActivationContribution concat" begin
        v0 = denseinputvertex("in", 3)
        v1 = fluxvertex("v1", Dense(nout(v0), 4), v0; layerfun=ActivationContribution)
        v2 = fluxvertex("v2", Dense(nout(v0), 3), v0; layerfun=ActivationContribution)
        v3 = concat("v3", v1; layerfun=ActivationContribution)
        v4 = fluxvertex("v4", Dense(nout(v3), 2), v3; layerfun=ActivationContribution)
        v5 = concat("v5", v4, v3, v3; layerfun=ActivationContribution)

        g = CompGraph(v0, v5)
        Flux.gradient(() -> sum(g(ones(Float32, nout(v0), 1))))

        # make sure values have materialized so we don't accidentally have a scalar value
        @test length(NaiveNASlib.default_outvalue(v3)) == nout(v3) == nout(v1)

        @test create_edge!(v2, v3)

        @test length(NaiveNASlib.default_outvalue(v3)) == nout(v3) == nout(v1) + nout(v2)
        @test length(NaiveNASlib.default_outvalue(v5)) == nout(v5) == 2 * nout(v3) + nout(v4)

        @test size(g(ones(Float32, nout(v0), 1))) == (nout(v5), 1)
    end

    @testset "Add input edge to ActivationContribution concat fail" begin
        import NaiveNASlib: PostAlign, ΔSizeFailNoOp, FailAlignSizeNoOp
        import NaiveNASflux: neuron_value
        v0 = denseinputvertex("in", 3)
        v1 = fluxvertex("v1", Dense(nout(v0), 4), v0; layerfun=ActivationContribution)
        v2 = fluxvertex("v2", Dense(nout(v0), 3), v0; layerfun=ActivationContribution)
        v3 = concat("v3", v1; layerfun=ActivationContribution)
        v4 = fluxvertex("v4", Dense(nout(v3), 2), v3; layerfun=ActivationContribution)
        v5 = concat("v5", v4, v3, v3; layerfun=ActivationContribution)

        g = CompGraph(v0, v5)
        Flux.gradient(() -> sum(g(ones(Float32, nout(v0), 1))))

        # make sure values have materialized so we don't accidentally have a scalar value
        @test length(NaiveNASlib.default_outvalue(v3)) == nout(v3) == nout(v1)

        nvbefore_v3 = copy(neuron_value(v3))
        nvbefore_v5 = copy(neuron_value(v5))

        @test create_edge!(v2, v3; strategy=PostAlign(ΔSizeFailNoOp(), FailAlignSizeNoOp())) == false

        @test length(NaiveNASlib.default_outvalue(v3)) == nout(v3) == nout(v1)
        @test length(NaiveNASlib.default_outvalue(v5)) == nout(v5) == 2 * nout(v3) + nout(v4)

        @test nvbefore_v3 == neuron_value(v3)
        @test nvbefore_v5 == neuron_value(v5)

        @test size(g(ones(Float32, nout(v0), 1))) == (nout(v5), 1)
    end

    @testset "Add input edge to ActivationContribution concat maze" begin
        import NaiveNASlib: PostAlign
        v0 = denseinputvertex("in", 3)
        v1 = fluxvertex("v1", Dense(nout(v0), 4), v0; layerfun=ActivationContribution)
        v2 = fluxvertex("v2", Dense(nout(v0), 3), v0; layerfun=ActivationContribution)
        v3 = concat("v3", v1; layerfun=ActivationContribution)
        v4 = concat("v4", v3; layerfun=ActivationContribution)
        v5 = concat("v5", v3; layerfun=ActivationContribution)
        v6 = concat("v6", v3,v4,v5; layerfun=ActivationContribution)

        g = CompGraph(v0, v6)
        Flux.gradient(() -> sum(g(ones(Float32, nout(v0), 1))))

        # make sure values have materialized so we don't accidentally have a scalar value
        @test length(NaiveNASlib.default_outvalue(v3)) == nout(v3) == nout(v1)

        @test create_edge!(v2, v3; strategy=PostAlign())

        @test length(NaiveNASlib.default_outvalue(v3)) == nout(v3) == nout(v1) + nout(v2)
        @test length(NaiveNASlib.default_outvalue(v4)) == nout(v4) == nout(v3)
        @test length(NaiveNASlib.default_outvalue(v5)) == nout(v5) == nout(v3)
        @test length(NaiveNASlib.default_outvalue(v6)) == nout(v6) == nout(v3) + nout(v4) + nout(v5)

        @test size(g(ones(Float32, nout(v0), 1))) == (nout(v6), 1)
    end

    @testset "Remove input edge to ActivationContribution concat" begin
        v0 = denseinputvertex("in", 3)
        v1 = fluxvertex("v1", Dense(nout(v0), 4), v0; layerfun=ActivationContribution)
        v2 = fluxvertex("v2", Dense(nout(v0), 3), v0; layerfun=ActivationContribution)
        v3 = concat("v3", v1, v2; layerfun=ActivationContribution)
        v4 = fluxvertex("v4", Dense(nout(v3), 2), v3; layerfun=ActivationContribution)
        v5 = concat("v5", v4, v3, v3; layerfun=ActivationContribution)

        g = CompGraph(v0, v5)
        Flux.gradient(() -> sum(g(ones(Float32, nout(v0), 1))))

        # make sure values have materialized so we don't accidentally have a scalar value
        @test length(NaiveNASlib.default_outvalue(v3)) == nout(v3) == nout(v1) + nout(v2)

        @test remove_edge!(v2, v3)

        @test length(NaiveNASlib.default_outvalue(v3)) == nout(v3) == nout(v1)
        @test length(NaiveNASlib.default_outvalue(v5)) == nout(v5) == 2 * nout(v3) + nout(v4)

        @test size(g(ones(Float32, nout(v0), 1))) == (nout(v5), 1)
    end

    
    @testset "Remove input edge to ActivationContribution concat fail" begin
        import NaiveNASlib: PostAlign, ΔSizeFailNoOp, FailAlignSizeNoOp
        import NaiveNASflux: neuron_value

        v0 = denseinputvertex("in", 3)
        v1 = fluxvertex("v1", Dense(nout(v0), 4), v0; layerfun=ActivationContribution)
        v2 = fluxvertex("v2", Dense(nout(v0), 3), v0; layerfun=ActivationContribution)
        v3 = concat("v3", v1, v2; layerfun=ActivationContribution)
        v4 = fluxvertex("v4", Dense(nout(v3), 2), v3; layerfun=ActivationContribution)
        v5 = concat("v5", v4, v3, v3; layerfun=ActivationContribution)

        g = CompGraph(v0, v5)
        Flux.gradient(() -> sum(g(ones(Float32, nout(v0), 1))))

        # make sure values have materialized so we don't accidentally have a scalar value
        @test length(NaiveNASlib.default_outvalue(v3)) == nout(v3) == nout(v1) + nout(v2)

        nvbefore_v3 = copy(neuron_value(v3))
        nvbefore_v5 = copy(neuron_value(v5))

        @test remove_edge!(v2, v3; strategy= PostAlign(ΔSizeFailNoOp(), FailAlignSizeNoOp())) == false

        @test length(NaiveNASlib.default_outvalue(v3)) == nout(v3) == nout(v1) + nout(v2)
        @test length(NaiveNASlib.default_outvalue(v5)) == nout(v5) == 2 * nout(v3) + nout(v4)

        @test nvbefore_v3 == neuron_value(v3)
        @test nvbefore_v5 == neuron_value(v5)

        @test size(g(ones(Float32, nout(v0), 1))) == (nout(v5), 1)
    end

    @testset "Remove input edge to ActivationContribution concat maze" begin
        import NaiveNASlib: PostAlign
        v0 = denseinputvertex("in", 3)
        v1 = fluxvertex("v1", Dense(nout(v0), 4), v0; layerfun=ActivationContribution)
        v2 = fluxvertex("v2", Dense(nout(v0), 3), v0; layerfun=ActivationContribution)
        v3 = concat("v3", v1, v2; layerfun=ActivationContribution)
        v4 = concat("v4", v3; layerfun=ActivationContribution)
        v5 = concat("v5", v3; layerfun=ActivationContribution)
        v6 = concat("v6", v3,v4,v5; layerfun=ActivationContribution)

        g = CompGraph(v0, v6)
        Flux.gradient(() -> sum(g(ones(Float32, nout(v0), 1))))

        # make sure values have materialized so we don't accidentally have a scalar value
        @test length(NaiveNASlib.default_outvalue(v3)) == nout(v3) == nout(v1) + nout(v2)

        @test remove_edge!(v2, v3; strategy=PostAlign())

        @test length(NaiveNASlib.default_outvalue(v3)) == nout(v3) == nout(v1)
        @test length(NaiveNASlib.default_outvalue(v4)) == nout(v4) == nout(v3)
        @test length(NaiveNASlib.default_outvalue(v5)) == nout(v5) == nout(v3)
        @test length(NaiveNASlib.default_outvalue(v6)) == nout(v6) == nout(v3) + nout(v4) + nout(v5)

        @test size(g(ones(Float32, nout(v0), 1))) == (nout(v6), 1)
    end
end
