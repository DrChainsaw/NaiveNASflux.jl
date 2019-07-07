
@testset "Examples" begin

    @testset "Pruning xor example" begin
        import Flux: train!, mse
        import Flux.Tracker: hook
        using Statistics
        import Random
        Random.seed!(666)
        niters = 100

        #Calculate value of activations according to https://arxiv.org/abs/1611.06440.
        # Short summary is that the first order taylor approximation of the optimization problem:
        # "which neurons shall I remove to minimize impact on the loss function"
        # boils down to:
        # "the ones which minimize abs(gradient * activation)" (assuming parameter independence).
        mutable struct ActivationContribution <: AbstractMutableComp
            layer
            contribution
            # Inner constructor is workaround for julia issue #32325
            ActivationContribution(l::AbstractMutableComp) = ActivationContribution(l, zeros(Float32, nout(l)))
            ActivationContribution(l::AbstractMutableComp, c) = new(l, c)
        end

        function(m::ActivationContribution)(x)
            act = layer(m)(x)
            return hook(act) do grad
                m.contribution[1:end] += mean(abs.(act .* grad).data, dims=2)
                return grad
            end
        end
        NaiveNASflux.layer(m::ActivationContribution) = m.layer

        # To peel the onion...
        activation_contribution(v::AbstractVertex) = activation_contribution(base(v))
        activation_contribution(v::CompVertex) = activation_contribution(v.computation)
        activation_contribution(m::ActivationContribution) = m.contribution

        Flux.@treelike ActivationContribution

        dense(in, outsize, act) = mutable(Dense(nout(in),outsize, act), in, layerfun=ActivationContribution)

        invertex = inputvertex("input", 2, FluxDense())
        layer1 = dense(invertex, 32, relu)
        layer2 = dense(layer1, 1, sigmoid)

        graph = CompGraph(invertex, layer2)
        opt = ADAM(0.1)
        loss(g) = (x, y) -> mse(g(x), y)

        x = Float32[0 0 1 1;
                    0 1 0 1]
        y = Float32[0 1 1 0]

        for iter in 1:niters
            train!(loss(graph), params(graph), [(x,y)], opt)
        end
        @test loss(graph)(x, y) < 0.001

        nprune = 5

        pruned_random = copy(graph)
        # Nothing has changed yet
        @test loss(pruned_random)(x, y) ≈ loss(graph)(x, y) atol = 1e-10

        Δnin(pruned_random.outputs[], rand(1:nout(layer1), nout(layer1) - nprune))
        apply_mutation(pruned_random)

        contribution = sortperm(activation_contribution(layer1))

        # Prune the least contributing neurons according to the metric in ActivationContribution
        pruned_least = copy(graph)
        Δnin(pruned_least.outputs[], contribution[nprune:end])
        apply_mutation(pruned_least)

        # Prune the most contributing neurons according to the metric in ActivationContribution
        pruned_most = copy(graph)
        Δnin(pruned_most.outputs[], contribution[1:end-nprune])
        apply_mutation(pruned_most)

        # Can I have my free lunch please?!
        @test loss(pruned_most)(x, y) > loss(pruned_random)(x, y) > loss(pruned_least)(x, y) > loss(graph)(x, y)
        @test loss(pruned_least)(x, y) ≈ loss(graph)(x, y) rtol = 1e-5
    end
end
