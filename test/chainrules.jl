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

    teststructs(g::CompGraph{<:Any, <:Tuple}, res, exp; seen=Base.IdSet()) = foreach(enumerate(outputs(g))) do (i, vo)
        teststructs(vo, seen, res.outputs[i] ,exp)
    end

    teststructs(g::CompGraph{<:Any, <:AbstractVertex}, res, exp; seen=Base.IdSet()) = teststructs(g.outputs, seen, res.outputs ,exp) 

    function teststructs(v::AbstractVertex, seen, res, exp) 
        v in seen && return
        push!(seen, v)
        _teststructs(v, seen, res, exp, Symbol(name(v)))
    end

    function _teststructs(::NaiveNASflux.InputShapeVertex, seen, res, exp, name) end

    function _teststructs(v::AbstractVertex, seen, res::RT, exp, name) where RT 
        if layertype(v) isa NaiveNASflux.FluxParLayer
            @testset "Check structure for $(name) of type $(typeof(v))" begin
                @test hasfield(RT, :base)
            end
            if hasfield(RT, :base)
                _teststructs(base(v), seen, res.base, exp, name)
            end
        end
    end
    function _teststructs(v::CompVertex, seen, res::RT, exp, name) where RT 
        @testset "Check structure for $(name) of type $(typeof(v))" begin
            @test hasfield(RT, :computation)
            _teststructs(v.computation, res.computation, exp, name)
        end
        foreach(enumerate(inputs(v))) do (i, vi)
            isnothing(res.inputs) && return
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

        @test Δnout!(v1, 1)
        resincex = Flux.gradient(g -> sum(g(indata)), graph)

        if lfun == identity
            @test expex[1].layers.v1.weight == resincex[1].outputs.base.base.inputs[1].base.base.computation.layer.weight[1:end-1,:]
            @test expex[1].layers.v1.bias == resincex[1].outputs.base.base.inputs[1].base.base.computation.layer.bias[1:end-1]
        
            @test expex[1].layers.v2.weight == resincex[1].outputs.base.base.computation.layer.weight[:, 1:end-1]
            @test expex[1].layers.v2.bias == resincex[1].outputs.base.base.computation.layer.bias
        elseif lfun == LazyMutable
            @test expex[1].layers.v1.weight == resincex[1].outputs.base.base.inputs[1].base.base.computation.mutable.layer.weight[1:end-1,:]
            @test expex[1].layers.v1.bias == resincex[1].outputs.base.base.inputs[1].base.base.computation.mutable.layer.bias[1:end-1]
        
            @test expex[1].layers.v2.weight == resincex[1].outputs.base.base.computation.mutable.layer.weight[:, 1:end-1]
            @test expex[1].layers.v2.bias == resincex[1].outputs.base.base.computation.mutable.layer.bias
        elseif lfun == ActivationContribution
            @test expex[1].layers.v1.weight == resincex[1].outputs.base.base.inputs[1].base.base.computation.layer.layer.weight[1:end-1,:]
            @test expex[1].layers.v1.bias == resincex[1].outputs.base.base.inputs[1].base.base.computation.layer.layer.bias[1:end-1]
        
            @test expex[1].layers.v2.weight == resincex[1].outputs.base.base.computation.layer.layer.weight[:, 1:end-1]
            @test expex[1].layers.v2.bias == resincex[1].outputs.base.base.computation.layer.layer.bias
        else
            @test expex[1].layers.v1.weight == resincex[1].outputs.base.base.inputs[1].base.base.computation.mutable.layer.layer.weight[1:end-1,:]
            @test expex[1].layers.v1.bias == resincex[1].outputs.base.base.inputs[1].base.base.computation.mutable.layer.layer.bias[1:end-1]
        
            @test expex[1].layers.v2.weight == resincex[1].outputs.base.base.computation.mutable.layer.layer.weight[:, 1:end-1]
            @test expex[1].layers.v2.bias == resincex[1].outputs.base.base.computation.mutable.layer.layer.bias
        end
    end

end