@testset "ChainRules" begin
    
   chain = Chain(Dense(2,3), Dense(3, 4)) 
   indata = reshape(collect(Float32, 1:6),2,3)

   @testset "Model gradient $(lfun == identity ? "" : " with layerfun=$lfun")" for lfun in (
    identity,
    LazyMutable,
    ActivationContribution,
    LazyMutable ∘ ActivationContribution
   )
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

        # TODO: Test explicit gradients
   end

end