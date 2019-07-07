
@testset "Examples" begin

    @testset "Pruning xor example" begin
        using NaiveNASflux
        import Flux: train!, mse
        import Random
        Random.seed!(666)
        niters = 100

        # First lets define a simple architecture
        dense(in, outsize, act) = mutable(Dense(nout(in),outsize, act), in, layerfun=ActivationContribution)

        invertex = inputvertex("input", 2, FluxDense())
        layer1 = dense(invertex, 32, relu)
        layer2 = dense(layer1, 1, sigmoid)
        original = CompGraph(invertex, layer2)

        # Training params, nothing to see here
        opt = ADAM(0.1)
        loss(g) = (x, y) -> mse(g(x), y)

        # Xor truth table: y = xor(x)
        x = Float32[0 0 1 1;
                    0 1 0 1]
        y = Float32[0 1 1 0]

        # Train the model
        for iter in 1:niters
            train!(loss(original), params(original), [(x,y)], opt)
        end
        @test loss(original)(x, y) < 0.001

        # Now, lets try three different ways to prune the network
        nprune = 5

        # Prune randomly
        pruned_random = copy(original)
        Δnin(pruned_random.outputs[], rand(1:nout(layer1), nout(layer1) - nprune))
        apply_mutation(pruned_random)

        pruneorder = sortperm(neuron_value(layer1))

        # Prune the least valuable neurons according to the metric in ActivationContribution
        pruned_least = copy(original)
        Δnin(pruned_least.outputs[], pruneorder[nprune:end])
        apply_mutation(pruned_least)

        # Prune the most valuable neurons according to the metric in ActivationContribution
        pruned_most = copy(original)
        Δnin(pruned_most.outputs[], pruneorder[1:end-nprune])
        apply_mutation(pruned_most)

        # Can I have my free lunch now please?!
        @test loss(pruned_most)(x, y) > loss(pruned_random)(x, y) > loss(pruned_least)(x, y) > loss(original)(x, y)

        # The metric calculated by ActivationContribution is actually quite good (in this case)!
        @test loss(pruned_least)(x, y) ≈ loss(original)(x, y) rtol = 1e-5
    end
end
