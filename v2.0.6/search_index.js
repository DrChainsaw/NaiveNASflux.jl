var documenterSearchIndex = {"docs":
[{"location":"reference/createvertex/#Vertex-Creation","page":"Vertex Creation","title":"Vertex Creation","text":"","category":"section"},{"location":"reference/createvertex/","page":"Vertex Creation","title":"Vertex Creation","text":"The functions added by NaiveNASflux are basically Flux-tailored convenience wrappers around those exported by NaiveNASlib.","category":"page"},{"location":"reference/createvertex/","page":"Vertex Creation","title":"Vertex Creation","text":"denseinputvertex\nrnninputvertex\nconv1dinputvertex\nconv2dinputvertex\nconv3dinputvertex\nconvinputvertex\nfluxvertex\nconcat","category":"page"},{"location":"reference/createvertex/#NaiveNASflux.denseinputvertex","page":"Vertex Creation","title":"NaiveNASflux.denseinputvertex","text":"denseinputvertex(name, size)\n\nReturn an input type vertex with the given name which promises 2D shaped input with size number of features suitable for e.g. Fluxs Dense layer.\n\nProviding the input type is not strictly necessary for the package to work and in many cases a normal inputvertex  will do. \n\nOne example of when it is useful is the concat function which needs to know the input type to automatically determine which dimension to concatenate.\n\n\n\n\n\n","category":"function"},{"location":"reference/createvertex/#NaiveNASflux.rnninputvertex","page":"Vertex Creation","title":"NaiveNASflux.rnninputvertex","text":"rnninputvertex(name, size)\n\nReturn an input type vertex with the given name which promises 2D shaped input with size number of features suitable for Fluxs recurrent layers.\n\nProviding the input type is not strictly necessary for the package to work and in many cases a normal inputvertex  will do. \n\nOne example of when it is useful is the concat function which needs to know the input type to automatically determine which dimension to concatenate.\n\n\n\n\n\n","category":"function"},{"location":"reference/createvertex/#NaiveNASflux.conv1dinputvertex","page":"Vertex Creation","title":"NaiveNASflux.conv1dinputvertex","text":"conv1dinputvertex(name, nchannel)\n\nReturn an input type vertex with the given name which promises convolution shaped input  with nchannel channels suitable for Fluxs convolution layers.\n\nEquivalent to convinputvertex(name, nchannel, 1). \n\nProviding the input type is not strictly necessary for the package to work and in many cases a normal inputvertex  will do. \n\nOne example of when it is useful is the concat function which needs to know the input type to automatically determine which dimension to concatenate.\n\n\n\n\n\n","category":"function"},{"location":"reference/createvertex/#NaiveNASflux.conv2dinputvertex","page":"Vertex Creation","title":"NaiveNASflux.conv2dinputvertex","text":"conv2dinputvertex(name, nchannel)\n\nReturn an input type vertex with the given name which promises convolution shaped input  with nchannel channels suitable for Fluxs convolution layers.\n\nEquivalent to convinputvertex(name, nchannel, 2). \n\nProviding the input type is not strictly necessary for the package to work and in many cases a normal inputvertex  will do. \n\nOne example of when it is useful is the concat function which needs to know the input type to automatically determine which dimension to concatenate.\n\n\n\n\n\n","category":"function"},{"location":"reference/createvertex/#NaiveNASflux.conv3dinputvertex","page":"Vertex Creation","title":"NaiveNASflux.conv3dinputvertex","text":"conv3dinputvertex(name, nchannel)\n\nReturn an input type vertex with the given name which promises convolution shaped input  with nchannel channels suitable for Fluxs convolution layers.\n\nEquivalent to convinputvertex(name, nchannel, 3). \n\nProviding the input type is not strictly necessary for the package to work and in many cases a normal inputvertex  will do. \n\nOne example of when it is useful is the concat function which needs to know the input type to automatically determine which dimension to concatenate.\n\n\n\n\n\n","category":"function"},{"location":"reference/createvertex/#NaiveNASflux.convinputvertex","page":"Vertex Creation","title":"NaiveNASflux.convinputvertex","text":"convinputvertex(name, nchannel, ndim)\n\nReturn an input type vertex with the given name which promises convolution shaped input  with nchannel channels and ndim number of dimensions for feature maps (e.g. 2 for images) suitable for Fluxs convolution layers.\n\nProviding the input type is not strictly necessary for the package to work and in many cases a normal inputvertex  will do. \n\nOne example of when it is useful is the concat function which needs to know the input type to automatically determine which dimension to concatenate.\n\n\n\n\n\n","category":"function"},{"location":"reference/createvertex/#NaiveNASflux.fluxvertex","page":"Vertex Creation","title":"NaiveNASflux.fluxvertex","text":"fluxvertex(l, in::AbstractVertex; layerfun=LazyMutable, traitfun=validated())\n\nReturn a vertex which wraps the layer l and has input vertex in.\n\nKeyword argument layerfun can be used to wrap the computation, e.g. in an ActivationContribution. \n\nKeyword argument traitfun can be used to wrap the MutationTrait of the vertex in a DecoratingTrait\n\n\n\n\n\nfluxvertex(name::AbstractString, l, in::AbstractVertex; layerfun=LazyMutable, traitfun=validated())\n\nReturn a vertex with name name which wraps the layer l and has input vertex in.\n\nName is only used when displaying or logging and does not have to be unique (although it probably is a good idea).\n\nKeyword argument layerfun can be used to wrap the computation, e.g. in an ActivationContribution. \n\nKeyword argument traitfun can be used to wrap the MutationTrait of the vertex in a DecoratingTrait\n\n\n\n\n\n","category":"function"},{"location":"reference/createvertex/#NaiveNASflux.concat","page":"Vertex Creation","title":"NaiveNASflux.concat","text":"concat(v::AbstractVertex, vs::AbstractVertex...; traitfun=identity, layerfun=identity)\n\nReturn a vertex which concatenates input along the activation (e.g. channel if convolution, first dimension if dense) dimension.\n\nInputs must have compatible activation shapes or an exception will be thrown.\n\nKeyword argument layerfun can be used to wrap the computation, e.g. in an ActivationContribution. \n\nKeyword argument traitfun can be used to wrap the MutationTrait of the vertex in a DecoratingTrait\n\nSee also NaiveNASlib.conc. \n\n\n\n\n\nconcat(name::AbstractString, v::AbstractVertex, vs::AbstractVertex...; traitfun=identity, layerfun=identity)\n\nReturn a vertex with name name which concatenates input along the activation (e.g. channel if convolution, first dimension if dense) dimension.\n\nName is only used when displaying or logging and does not have to be unique (although it probably is a good idea).\n\nInputs must have compatible activation shapes or an exception will be thrown.\n\nKeyword argument layerfun can be used to wrap the computation, e.g. in an ActivationContribution. \n\nKeyword argument traitfun can be used to wrap the MutationTrait of the vertex in a DecoratingTrait\n\nSee also NaiveNASlib.conc. \n\n\n\n\n\n","category":"function"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"EditURL = \"https://github.com/DrChainsaw/NaiveNASflux.jl/blob/master/test/examples/quicktutorial.jl\"","category":"page"},{"location":"examples/quicktutorial/#Quick-Tutorial","page":"Quick Tutorial","title":"Quick Tutorial","text":"","category":"section"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"Check out the basic usage of NaiveNASlib for less verbose examples.","category":"page"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"Here is a quick rundown of some common operations.","category":"page"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"using NaiveNASflux, Flux, Test","category":"page"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"Create an input vertex which tells its output vertices that they can expect 2D convolutional input (i.e 4D arrays).","category":"page"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"invertex = conv2dinputvertex(\"in\", 3)","category":"page"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"Vertex type for Flux-layers is automatically inferred through fluxvertex.","category":"page"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"conv = fluxvertex(Conv((3,3), 3 => 5, pad=(1,1)), invertex)\nbatchnorm = fluxvertex(BatchNorm(nout(conv), relu), conv)","category":"page"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"Explore the graph.","category":"page"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"@test inputs(conv) == [invertex]\n@test outputs(conv) == [batchnorm]\n\n@test nin(conv) == [3]\n@test nout(conv) == 5\n\n@test layer(conv) isa Flux.Conv\n@test layer(batchnorm) isa Flux.BatchNorm","category":"page"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"Naming vertices is a good idea for debugging and logging purposes.","category":"page"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"namedconv = fluxvertex(\"namedconv\", Conv((5,5), 3=>7, pad=(2,2)), invertex)\n\n@test name(namedconv) == \"namedconv\"","category":"page"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"Concatenate activations. Dimension is automatically inferred.","category":"page"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"conc = concat(\"conc\", namedconv, batchnorm)\n@test nout(conc) == nout(namedconv) + nout(batchnorm)","category":"page"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"No problem to combine with convenience functions from NaiveNASlib.","category":"page"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"residualconv = fluxvertex(\"residualconv\", Conv((3,3), nout(conc) => nout(conc), pad=(1,1)), conc)\nadd = \"add\" >> conc + residualconv\n\n@test name(add) == \"add\"\n@test inputs(add) == [conc, residualconv]","category":"page"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"Computation graph for evaluation. It is basically a more general version of Flux.Chain.","category":"page"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"graph = CompGraph(invertex, add)","category":"page"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"Access the vertices of the graph.","category":"page"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"@test vertices(graph) == [invertex, namedconv, conv, batchnorm, conc, residualconv, add]","category":"page"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"CompGraphs can be evaluated just like any function.","category":"page"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"x = ones(Float32, 7, 7, nout(invertex), 2)\n@test size(graph(x)) == (7, 7, nout(add), 2) == (7 ,7, 12 ,2)","category":"page"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"Mutate number of neurons.","category":"page"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"@test nout(add) == nout(residualconv) == nout(conv) + nout(namedconv) == 12\nΔnout!(add => -3)\n@test nout(add) == nout(residualconv) == nout(conv) + nout(namedconv) == 9","category":"page"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"Remove a layer.","category":"page"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"@test nvertices(graph) == 7\nremove!(batchnorm)\n@test nvertices(graph) == 6","category":"page"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"Add a layer.","category":"page"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"insert!(residualconv, v -> fluxvertex(BatchNorm(nout(v), relu), v))\n@test nvertices(graph) == 7","category":"page"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"Change kernel size (and supply new padding).","category":"page"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"namedconv |> KernelSizeAligned(-2,-2; pad=SamePad())","category":"page"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"Note: Parameters not changed yet...","category":"page"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"@test size(NaiveNASflux.weights(layer(namedconv))) == (5, 5, 3, 7)","category":"page"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"... because mutations are lazy by default so that no new parameters are created until the graph is evaluated.","category":"page"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"@test size(graph(x)) == (7, 7, nout(add), 2) == (7, 7, 9, 2)\n@test size(NaiveNASflux.weights(layer(namedconv))) == (3, 3, 3, 4)","category":"page"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"","category":"page"},{"location":"examples/quicktutorial/","page":"Quick Tutorial","title":"Quick Tutorial","text":"This page was generated using Literate.jl.","category":"page"},{"location":"reference/misc/#Misc.-Utilities","page":"Misc. Utilities","title":"Misc. Utilities","text":"","category":"section"},{"location":"reference/misc/","page":"Misc. Utilities","title":"Misc. Utilities","text":"layer","category":"page"},{"location":"reference/misc/#NaiveNASflux.layer","page":"Misc. Utilities","title":"NaiveNASflux.layer","text":"layer(v)\n\nReturn the computation wrapped inside v and inside any mutable wrappers.\n\nExamples\n\njulia> using NaiveNASflux, Flux\n\njulia> layer(fluxvertex(Dense(2,3), inputvertex(\"in\", 2)))\nDense(2 => 3)       # 9 parameters\n\n\n\n\n\n","category":"function"},{"location":"reference/misc/","page":"Misc. Utilities","title":"Misc. Utilities","text":"KernelSizeAligned","category":"page"},{"location":"reference/misc/#NaiveNASflux.KernelSizeAligned","page":"Misc. Utilities","title":"NaiveNASflux.KernelSizeAligned","text":"KernelSizeAligned(Δsize; pad)\nKernelSizeAligned(Δs::Integer...;pad)\n\nStrategy for changing kernel size of convolutional layers where filters remain phase aligned. In other words, the same  element indices are removed/added for all filters and only 'outer' elements are dropped or added.\n\nCall with vertex as input to change weights.\n\nExamples\n\njulia> using NaiveNASflux, Flux\n\njulia> cv = fluxvertex(Conv((3,3), 1=>1;pad=SamePad()), conv2dinputvertex(\"in\", 1));\n\njulia> cv(ones(Float32, 4,4,1,1)) |> size\n(4, 4, 1, 1)\n\njulia> layer(cv).weight |> size\n(3, 3, 1, 1)\n\njulia> cv |> KernelSizeAligned(-1, 1; pad=SamePad());\n\njulia> cv(ones(Float32, 4,4,1,1)) |> size\n(4, 4, 1, 1)\n\njulia> layer(cv).weight |> size\n(2, 4, 1, 1)\n\n\n\n\n\n","category":"type"},{"location":"reference/misc/","page":"Misc. Utilities","title":"Misc. Utilities","text":"NeuronUtilityEvery","category":"page"},{"location":"reference/misc/#NaiveNASflux.NeuronUtilityEvery","page":"Misc. Utilities","title":"NaiveNASflux.NeuronUtilityEvery","text":"NeuronUtilityEvery{N,T}\nNeuronUtilityEvery(n::Int, method::T)\n\nCalculate neuron utility using method every n:th call.\n\nUseful to reduce runtime overhead.\n\n\n\n\n\n","category":"type"},{"location":"examples/xorpruning/","page":"Model Pruning Example","title":"Model Pruning Example","text":"EditURL = \"https://github.com/DrChainsaw/NaiveNASflux.jl/blob/master/test/examples/xorpruning.jl\"","category":"page"},{"location":"examples/xorpruning/#Model-Pruning-Example","page":"Model Pruning Example","title":"Model Pruning Example","text":"","category":"section"},{"location":"examples/xorpruning/","page":"Model Pruning Example","title":"Model Pruning Example","text":"While NaiveNASflux does not come with any built in search policies, it is still possible to do some cool stuff with it. Below is a very simple example of parameter pruning.","category":"page"},{"location":"examples/xorpruning/","page":"Model Pruning Example","title":"Model Pruning Example","text":"First we need some boilerplate to create the model and do the training:","category":"page"},{"location":"examples/xorpruning/","page":"Model Pruning Example","title":"Model Pruning Example","text":"using NaiveNASflux, Flux, Test\nusing Flux: train!, mse\nimport Random\nRandom.seed!(0)\nniters = 50","category":"page"},{"location":"examples/xorpruning/","page":"Model Pruning Example","title":"Model Pruning Example","text":"To cut down on the verbosity, start by making a  helper function for creating a Dense layer as a graph vertex. The keyword argument layerfun=ActivationContribution will wrap the layer and compute an activity based neuron utility metric for it while the model trains.","category":"page"},{"location":"examples/xorpruning/","page":"Model Pruning Example","title":"Model Pruning Example","text":"densevertex(in, outsize, act) = fluxvertex(Dense(nout(in),outsize, act), in, layerfun=ActivationContribution)","category":"page"},{"location":"examples/xorpruning/","page":"Model Pruning Example","title":"Model Pruning Example","text":"Ok, lets create the model and train it. We overparameterize quite heavily to avoid sporadic test failures :)","category":"page"},{"location":"examples/xorpruning/","page":"Model Pruning Example","title":"Model Pruning Example","text":"invertex = denseinputvertex(\"input\", 2)\nlayer1 = densevertex(invertex, 32, relu)\nlayer2 = densevertex(layer1, 1, sigmoid)\noriginal = CompGraph(invertex, layer2)\n\n# Training params, nothing to see here\nopt = ADAM(0.1)\nloss(g) = (x, y) -> mse(g(x), y)\n\n# Training data: xor truth table: y = xor(x) just so we don't need to download a dataset.\nx = Float32[0 0 1 1;\n            0 1 0 1]\ny = Float32[0 1 1 0]\n\n# Train the model\ntrain!(loss(original), params(original), Iterators.repeated((x,y), niters), opt)\n@test loss(original)(x, y) < 0.001","category":"page"},{"location":"examples/xorpruning/","page":"Model Pruning Example","title":"Model Pruning Example","text":"With that out of the way, lets try three different ways to prune the hidden layer (vertex nr 2 in the graph). To make examples easier to compare, lets decide up front that we want to remove half of the hidden layer neurons and try out three different ways of how to select which ones to remove.","category":"page"},{"location":"examples/xorpruning/","page":"Model Pruning Example","title":"Model Pruning Example","text":"nprune = 16","category":"page"},{"location":"examples/xorpruning/","page":"Model Pruning Example","title":"Model Pruning Example","text":"Prune the neurons with lowest utility according to the metric in ActivationContribution. This is the default if no utility function is provided.","category":"page"},{"location":"examples/xorpruning/","page":"Model Pruning Example","title":"Model Pruning Example","text":"pruned_least = deepcopy(original)\nΔnout!(pruned_least[2] => -nprune)","category":"page"},{"location":"examples/xorpruning/","page":"Model Pruning Example","title":"Model Pruning Example","text":"Prune the neurons with higest utility according to the metric in ActivationContribution. This is obviously not a good idea if you want to preserve the accuracy.","category":"page"},{"location":"examples/xorpruning/","page":"Model Pruning Example","title":"Model Pruning Example","text":"pruned_most = deepcopy(original)\nΔnout!(pruned_most[2] => -nprune) do v\n    vals = NaiveNASlib.defaultutility(v)\n    return 2*sum(vals) .- vals # Ensure all values are still > 0, even for last vertex\nend","category":"page"},{"location":"examples/xorpruning/","page":"Model Pruning Example","title":"Model Pruning Example","text":"Prune randomly selected neurons by giving random utility.","category":"page"},{"location":"examples/xorpruning/","page":"Model Pruning Example","title":"Model Pruning Example","text":"pruned_random = deepcopy(original)\nΔnout!(v -> rand(nout(v)), pruned_random[2] => -nprune)","category":"page"},{"location":"examples/xorpruning/","page":"Model Pruning Example","title":"Model Pruning Example","text":"Free lunch anyone?","category":"page"},{"location":"examples/xorpruning/","page":"Model Pruning Example","title":"Model Pruning Example","text":"@test   loss(pruned_most)(x, y)   >\n        loss(pruned_random)(x, y) >\n        loss(pruned_least)(x, y)  >=\n        loss(original)(x, y)","category":"page"},{"location":"examples/xorpruning/","page":"Model Pruning Example","title":"Model Pruning Example","text":"The metric calculated by ActivationContribution is actually quite good in this case.","category":"page"},{"location":"examples/xorpruning/","page":"Model Pruning Example","title":"Model Pruning Example","text":"@test loss(pruned_least)(x, y) ≈ loss(original)(x, y) atol = 1e-5","category":"page"},{"location":"examples/xorpruning/","page":"Model Pruning Example","title":"Model Pruning Example","text":"","category":"page"},{"location":"examples/xorpruning/","page":"Model Pruning Example","title":"Model Pruning Example","text":"This page was generated using Literate.jl.","category":"page"},{"location":"#Introduction","page":"Introduction","title":"Introduction","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"NaiveNASflux is an extension of NaiveNASlib which adds primitives for Flux layers so that they can be used in a computation graph which NaiveNASlib can modify. Apart from this, it adds very little new functionality.","category":"page"},{"location":"#Reading-Guideline","page":"Introduction","title":"Reading Guideline","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"Due to how NaiveNASflux just glues Flux and NaiveNASlib, most of the things one can use NaiveNASflux for is described in the documentation for NaiveNASlib.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"The Quick Tutorial gives an overview of some basic operations while the Model Pruning Example show simple usage without bringing in full fledged neural architecture search.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"The API reference is split up into categories in an attempt to make it easy to answer \"how do I achieve X?\"-type questions.","category":"page"},{"location":"reference/layerwrappers/#Layer-Wrappers","page":"Layer Wrappers","title":"Layer Wrappers","text":"","category":"section"},{"location":"reference/layerwrappers/","page":"Layer Wrappers","title":"Layer Wrappers","text":"NaiveNASflux wraps Flux layers in mutable wrapper types by default so that the vertex operations can be mutated without having to recreate the whole model. Additional wrappers which might be useful are described here.","category":"page"},{"location":"reference/layerwrappers/","page":"Layer Wrappers","title":"Layer Wrappers","text":"ActivationContribution\nLazyMutable","category":"page"},{"location":"reference/layerwrappers/#NaiveNASflux.ActivationContribution","page":"Layer Wrappers","title":"NaiveNASflux.ActivationContribution","text":"ActivationContribution{L,M} <: AbstractMutableComp\nActivationContribution(l)\nActivationContribution(l, method)\n\nCalculate neuron utility based on activations and gradients using method.\n\nCan be a performance bottleneck in cases with large activations. Use NeuronUtilityEvery to mitigate.\n\nDefault method is described in https://arxiv.org/abs/1611.06440.\n\nShort summary is that the first order taylor approximation of the optimization problem: \"which neurons shall I remove to minimize impact on the loss function?\"  boils down to: \"the ones which minimize abs(gradient * activation)\" (assuming parameter independence).\n\n\n\n\n\n","category":"type"},{"location":"reference/layerwrappers/#NaiveNASflux.LazyMutable","page":"Layer Wrappers","title":"NaiveNASflux.LazyMutable","text":"LazyMutable\nLazyMutable(m::AbstractMutableComp)\n\nLazy version of MutableLayer in the sense that it does not perform any mutations until invoked to perform a computation.\n\nThis reduces the need to garbage collect when multiple mutations might be applied to a vertex before evaluating the model.\n\nAlso useable for factory-like designs where the actual layers of a computation graph are not instantiated until the graph is used.\n\nExamples\n\njulia> using NaiveNASflux, Flux\n\njulia> struct DenseConfig end\n\njulia> lazy = LazyMutable(DenseConfig(), 2, 3);\n\njulia> layer(lazy)\nDenseConfig()\n\njulia> function NaiveNASflux.dispatch!(m::LazyMutable, ::DenseConfig, x)\n       m.mutable = Dense(nin(m)[1], nout(m), relu)\n       return m.mutable(x)\n       end;\n\njulia> lazy(ones(Float32, 2, 5)) |> size\n(3, 5)\n\njulia> layer(lazy)\nDense(2 => 3, relu)  # 9 parameters\n\n\n\n\n\n","category":"type"}]
}
