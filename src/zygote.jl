# Stuff (mostly adjoints) to make things play nice with Zygote

nograd(f) = f()
Flux.Zygote.@nograd nograd

Flux.Zygote.@nograd mutate
Flux.Zygote.@adjoint! function dispatch!(lm::LazyMutable, m::ResetLazyMutable, x...)
     dispatch!(lm, m, x...), Δ -> nothing
end

# Basically a workaround as the pullback for the recursive implementation in NaiveNASlib which
#       1) uses get! which does not have a pullback function
#        and
#       2) was in some cases extremely slow or even stalled completely
Flux.Zygote.@adjoint function output!(memo::AbstractDict, v::AbstractVertex)
    return Flux.Zygote.pullback(__context__, output_loop!, memo, v)
end


function output_loop!(memo, v)
    vs = nograd() do
        # ancestors returns all input ancestors to v in topological order
        # We also provide all vertices for which we have the output already in memo
        # so we don't do unnecessary calculations.
        ancestors(v, collect(AbstractVertex, keys(memo)))[length(memo)+1:end]
    end

    for vn in vs
        vnins = inputs(vn) # Types don't seem to be inferred if put in map
        inpt = map(iv -> memo[iv], vnins)
        memo[vn] = vn(inpt...)
    end
    return memo[v]
end
