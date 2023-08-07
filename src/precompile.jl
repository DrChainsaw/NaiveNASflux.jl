using PrecompileTools

let 
    @setup_workload begin
        iv1 = denseinputvertex("iv1", 1)
        v1 = fluxvertex("v1", Dense(nout(iv1) => 1), iv1)
        v2 = concat("v2", v1, v1; layerfun=ActivationContribution)
        v3 = concat("v3", v2,v1,iv1)
        v4 = "v4" >> v3 + v3
        v5 = "v5" >> v4 + v4 + v4
        v6 = fluxvertex("v6", Dense(nout(v5) => 1), v5; layerfun = ActivationContribution ∘ LazyMutable)

        g1 = CompGraph(iv1, v6)
        x1 = ones(Float32, 1, 1)

        @compile_workload begin
            iv1 = denseinputvertex("iv1", 1)
            fluxvertex("v1", Dense(nout(iv1) => 1), iv1)

            g1(x1)
            Flux.@code_adjoint g1(x1)
            #Optimisers.setup(Optimisers.Descent(0.1f0), g1)
            #Flux.gradient((g,x) -> sum(g(x)), g1, x1)

            Δnout!(v3 => relaxed(2))
        end
    end
end