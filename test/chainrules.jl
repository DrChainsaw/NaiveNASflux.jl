@testset "ChainRules" begin
    
    @testset "nograd" begin
        import NaiveNASflux.nograd
        @test Flux.gradient(+, 1,2) == (1,1)
        @test Flux.gradient(1,2) do x,y
            nograd() do
                x+y
            end
        end == (nothing,nothing)
    end
    
    import Optimisers
    function with_explicit_grads(f)
        try
            NaiveNASlib.enable_explicit_gradients[] = true
            f()   
        finally
            NaiveNASlib.enable_explicit_gradients[] = false
        end
    end

    teststructs(g::CompGraph, res, exp; seen=Base.IdSet()) = foreach(enumerate(outputs(g))) do (i, vo)
        teststructs(vo, seen, res.outputs[i] ,exp)
    end

    function teststructs(v::AbstractVertex, seen, res, exp) 
        v in seen && return
        push!(seen, v)
        _teststructs(v, seen, res, exp, Symbol(name(v)))
    end

    function _teststructs(::NaiveNASflux.InputShapeVertex, seen, res, exp, name) end

    function _teststructs(v::AbstractVertex, seen, res::RT, exp, name) where RT 
        @testset "Check structure for $(name) of type $(typeof(v))" begin
            @test hasfield(RT, :base)
        end
        if hasfield(RT, :base)
            _teststructs(base(v), seen, res.base, exp, name)
        end
    end
    function _teststructs(v::CompVertex, seen, res::RT, exp, name) where RT 
        @testset "Check structure for $(name) of type $(typeof(v))" begin
            @test hasfield(RT, :computation)
            _teststructs(v.computation, res.computation, exp, name)
        end
        foreach(enumerate(inputs(v))) do (i, vi)
            teststructs(vi, seen, res.inputs[i], exp)
        end            
    end
    function _teststructs(m::LazyMutable, res::RT, exp, name) where RT 
        @testset "Check structure for $(name) of type $(typeof(m))" begin
            @test hasfield(RT, :mutable)
        end
        _teststructs(m.mutable, res.mutable, exp, name)
    end

    function _teststructs(m::ActivationContribution, res::RT, exp, name) where RT 
        @testset "Check structure for $(name) of type $(typeof(m))" begin
            @test hasfield(RT, :layer)
        end
        _teststructs(m.layer, res.layer, exp, name)
    end

    function _teststructs(m::NaiveNASflux.MutableLayer, res::RT, exp, name) where RT 
        @testset "Check structure for $(name)" begin
            @test hasfield(RT, :layer)

            explayer = getindex(exp, name)
            reslayer = res.layer
            
            _teststructs(reslayer, explayer)
        end
    end

    _teststructs(res, exp) = @test res == exp 
    _teststructs(res::Dense, exp::Dense) = _teststructs(Optimisers.trainable(res), Optimisers.trainable(exp))

    function _teststructs(res::T, exp::T) where T <: Optimisers.Leaf 
        _teststructs(res.rule, exp.rule)
        _teststructs(res.state, exp.state)
    end

    function _teststructs(nt1::T, nt2::T) where T <: NamedTuple
        @testset "Check param $p" for p in keys(nt1)
            _teststructs(getfield(nt1, p), getfield(nt2, p))
        end
    end

    @testset "Model gradient $(lfun == identity ? "" : " with layerfun=$lfun")" for lfun in (
        identity,
        LazyMutable,
        ActivationContribution,
        LazyMutable ∘ ActivationContribution
    )
        chain = Chain(v1 = Dense(2,3), v2 = Dense(3, 4)) 
        indata = reshape(collect(Float32, 1:6),2,3)

        iv = denseinputvertex("in", 2)
        v1 = fluxvertex("v1", chain[1], iv; layerfun=lfun)
        v2 = fluxvertex("v2", chain[2], v1; layerfun=lfun)
        graph = CompGraph(iv, v2)

        ps = Flux.params(chain)

        exp = Flux.gradient(() -> sum(chain(indata)), ps)
        res = Flux.gradient(() -> sum(graph(indata)), ps)

        for p in ps
            @test exp[p] == res[p]
        end

        with_explicit_grads() do 
            @test gradient(sum ∘ graph, indata) == gradient(sum ∘ chain, indata)

            expex = Flux.gradient(c -> sum(c(indata)), chain)
            resex = Flux.gradient(g -> sum(g(indata)), graph)
            teststructs(graph, resex..., expex[1].layers) 

            @testset "Optimisers" begin
                graphstate = Optimisers.setup(Optimisers.Adam(), graph)
                chainstate = Optimisers.setup(Optimisers.Adam(), chain)
                @testset "Setup state" begin
                    teststructs(graph, graphstate, chainstate.layers)
                end
                # TODO: Why deepcopy needed? fmap(copy, graph) does not seem to work?
                graphstate, newgraph = Optimisers.update(graphstate, deepcopy(graph), resex...)
                chainstate, newchain = Optimisers.update(chainstate, chain, expex...)
                @testset "New state" begin
                    teststructs(newgraph, graphstate, chainstate.layers)
                end
                @testset "New model" begin
                    teststructs(newgraph, Optimisers.trainable(newgraph), Optimisers.trainable(newchain).layers)
                end
            end
        end

        @test Δnout!(v1, 1)
        resinc = Flux.gradient(() -> sum(graph(indata)), Flux.params(graph))
        
        for (p1, p2) in zip(Flux.params(chain[1]).order, Flux.params(v1).order)
            if ndims(p1) == 1
                @test exp[p1] == resinc[p2][1:end-1]
            else
                @test exp[p1] == resinc[p2][1:end-1,:]
            end
        end

        for (p1, p2) in zip(Flux.params(chain[2]).order, Flux.params(v2).order)
            if ndims(p1) == 1
                @test exp[p1] == resinc[p2]
            else
                @test exp[p1] == resinc[p2][:, 1:end-1]
            end
        end
    end

end