
@testset "Examples" begin

    @testset "Pruning xor example" begin
        using NaiveNASflux, Test
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

        # Training data: xor truth table: y = xor(x)
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

        # Prune randomly selected neurons
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

        # The metric calculated by ActivationContribution is actually quite good (in this case).
        @test loss(pruned_least)(x, y) ≈ loss(original)(x, y) rtol = 1e-5
    end

    @testset "Conv 2D xor example" begin
        using NaiveNASflux, Test
        import Flux: train!, logitbinarycrossentropy, glorot_uniform
        using Statistics
        import Random
        Random.seed!(666)
        niters = 20

        # Layers used in this example
        conv(in, outsize, act; init=glorot_uniform) = mutable(Conv((3,3),nout(in)=>outsize, act, pad=(1,1), init=init), in)
        avgpool(in, h, w) = mutable(MeanPool((h, w)), in)
        dense(in, outsize, act) = mutable(Dense(nout(in),outsize, act), in)

        # Size of the input
        height = 4
        width = 4

        function model(nconv)
            invertex = inputvertex("in", 1, FluxConv())
            l = invertex
            for i in 1:nconv
                l = conv(l, 16, relu)
            end
            l = avgpool(l, height, width)
            l = invariantvertex(x -> x[1,1,:,:], l)
            l = dense(l, 2, identity)
            return CompGraph(invertex, l)
        end
        original = model(1)

        # Training params, nothing to see here
        opt_org = ADAM(0.01)
        loss(g) = (x, y) -> mean(logitbinarycrossentropy.(g(x), y))

        # Training data: 2D xor(-ish)
        # Class 1 are matrices A where A[n,m] ⊻ A[n+h÷2, m]) ⩓ A[n,m] ⊻ A[n, m+w÷2] ∀ n<h÷2, m<w÷2 is true, e.g. [1 0; 0, 1]
        function xy1(h,w)
            q1_3 = rand(Bool,h÷2,w÷2)
            q2_4 = .!q1_3
            return (x = Float32.(vcat(hcat(q2_4, q1_3), hcat(q1_3, q2_4))), y = Float32[0, 1])
        end
        xy0(h,w) = (x = rand(Float32[0,1], h,w), y = Float32[1, 0]) #Joke's on me when this generates false negatives :)
        # Generate 50% class 1 and 50% class 0 examples in one batch
        batch(h,w,batchsize) = mapfoldl(i -> i==0 ? xy0(h,w) : xy1(h,w), (ex1, ex2) -> (x = cat(ex1.x, ex2.x, dims=4), y = hcat(ex1.y,ex2.y)), (1:batchsize) .% 2)

        x_test, y_test = batch(height, width, 1024)
        startloss = loss(original)(x_test, y_test)

        # Train the model
        for iter in 1:niters
            train!(loss(original), params(original), [batch(height, width, 64)], opt_org)
        end
        # That didn't work so well...
        @test loss(original)(x_test, y_test) ≈ startloss atol=1e-1

        # Lets try three things:
        # 1. Just train the same model some more
        # 2. Add two more conv-layers to the already trained model
        # 3. Create a new model with three conv layers from scratch

        # Disclaimer: This experiment is intended to show usage of this library.
        # It is not meant to give evidence that method 2 is the better option.
        # Hyperparameters are tuned to strongly favor 2 in order to avoid sporadic failures

        # Add two layers after the conv layer
        add_layers = copy(original)
        function add2conv(in)
            l = conv(in, nout(in), relu, init=idmapping)
            return conv(l, nout(in), relu, init=idmapping)
        end
        insert!(vertices(add_layers)[2], add2conv)

        # New layers are initialized to identity mapping weights
        # We basically have the same model as before, just with more potential
        # Not guaranteed to be a good idea as it relies on existing layers to provide gradient diversity
        @test add_layers(x_test) == original(x_test)

        # Create a new model with three conv layers
        new_model = model(3)

        opt_add = deepcopy(opt_org)
        opt_new = ADAM(0.01)

        # Lets try again
        for iter in 1:niters
            b = batch(height, width, 64)
            train!(loss(original), params(original), [b], opt_org)
            train!(loss(add_layers), params(add_layers), [b], opt_add)
            train!(loss(new_model), params(new_model), [b], opt_new)
        end

        @test loss(add_layers)(x_test,y_test) < loss(new_model)(x_test,y_test) < loss(original)(x_test,y_test)
    end
end
