

function NaiveNASlib.compconstraint!(case, s, ::FluxLayer, data) end
NaiveNASlib.compconstraint!(case, s::NaiveNASlib.DecoratingJuMPΔSizeStrategy, lt::FluxLayer, data) = NaiveNASlib.compconstraint!(case, NaiveNASlib.base(s), lt, data)
function NaiveNASlib.compconstraint!(case, ::NaiveNASlib.AbstractJuMPΔSizeStrategy, ::FluxDepthwiseConv, data)

  # Add constraint that nout(l) == n * nin(l) where n is integer
  ins = filter(vin -> vin in keys(data.noutdict), inputs(data.vertex))

   # "data.noutdict[data.vertex] == data.noutdict[ins[i]] * x" where x is an integer variable is not linear
   # Instead we use the good old big-M strategy to set up a series of "or" constraints (exactly one of them must be true).
   # This is combined with the abs value formulation to force equality.
   # multiplier[j] is false if and only if data.noutdict[data.vertex] == j*data.noutdict[ins[i]]
   # all but one of multiplier must be true
   # Each constraint below is trivially true if multiplier[j] is true since 1e6 is way bigger than the difference between the two variables 
  multipliers = @variable(data.model, [1:10], Bin)
  @constraint(data.model, sum(multipliers) == length(multipliers)-1)
  @constraint(data.model, [i=1:length(ins),j=1:length(multipliers)], data.noutdict[data.vertex] - j*data.noutdict[ins[i]] + multipliers[j] * 1e6 >= 0)
  @constraint(data.model, [i=1:length(ins),j=1:length(multipliers)], data.noutdict[data.vertex] - j*data.noutdict[ins[i]] - multipliers[j] * 1e6 <= 0)

  # Inputs which does not have a variable, possibly because all_in_Δsize_graph did not consider it to be part of the set of vertices which may change
  # We will constrain data.vertex to have integer multiple of its current size
  fixedins = filter(vin -> vin ∉ ins, inputs(data.vertex))
  if !isempty(fixedins) 
    fv_out = @variable(data.model, integer=true)
    @constraint(data.model, [i=1:length(fixedins)], data.noutdict[data.vertex] == nout(fixedins[i]) * fv_out)
  end
end

function NaiveNASlib.compconstraint!(::NaiveNASlib.NeuronIndices, s::NaiveNASlib.AbstractJuMPΔSizeStrategy, t::FluxDepthwiseConv, data)
  model = data.model
  v = data.vertex
  select = data.outselectvars[v]
  insert = data.outinsertvars[v]

  nin(v)[] == 1 && return # Special case, no restrictions as we only need to be an integer multple of 1?

  ngroups = div(nout(v), nin(v)[])
  maxgroupsize = nin(v)[]
  groupsizes = @variable(model, [1:maxgroupsize], Bin)

  @constraint(model, sum(groupsizes) == length(groupsizes)-1)
  # Neurons mapped to the same weight are interleaved, i.e layer.weight[:,:,1,:] maps to y[1:ngroups:end] where y = layer(x)
  for group in 1:ngroups
    select_in_group = select[group : ngroups : end]
    select_none_in_group = @variable(model, binary=true)
    @constraint(model, sum(select_in_group) <= 1e6 * (1-select_none_in_group))

    insert_in_group = insert[group : ngroups : end]
    for j in 1:length(groupsizes)
      @constraint(model, sum(select_in_group) + (groupsizes[j] + select_none_in_group) * 1e6 >= j) 
      @constraint(model, sum(select_in_group) - (groupsizes[j] + select_none_in_group) * 1e6 <= j)    
      
      @constraint(model, [i=2:j], insert_in_group[i] - insert_in_group[i-1] - groupsizes[j] * 1e6 <= 0)
      @constraint(model, [i=2:j], insert_in_group[i] - insert_in_group[i-1] + groupsizes[j] * 1e6 >= 0)
      @constraint(model, insert_in_group[1] - insert_in_group[j] - groupsizes[j] * 1e6 <= 0)
      @constraint(model, insert_in_group[1] - insert_in_group[j] + groupsizes[j] * 1e6 >= 0)

      @constraint(model, [i=j+1:maxgroupsize], insert_in_group[i] - groupsizes[j] * 1e6 <= 0)
      @constraint(model, [i=j+1:maxgroupsize], insert_in_group[i] + groupsizes[j] * 1e6 >= 0)  
    end
  end

  ins = filter(vin -> vin in keys(data.noutdict), inputs(data.vertex))
  inmultipliers = @variable(model, [1:maxgroupsize], Int)
  @constraint(model, 1 .<= inmultipliers .<= 100)
  @constraint(data.model, [i=1:length(ins),j=1:maxgroupsize], data.noutdict[ins[i]] - j * inmultipliers[j] + groupsizes[j] * 1e6 >= 0)
  @constraint(data.model, [i=1:length(ins),j=1:maxgroupsize], data.noutdict[ins[i]] - j * inmultipliers[j] - groupsizes[j] * 1e6 <= 0)

  @constraint(data.model, [i=1:length(ins)], data.noutdict[ins[i]] <= data.noutdict[v])

  #@constraint(data.model, [i=1:length(ins),j=1:length(groupsizes)], data.noutdict[ins[i]] / j + groupsizes[j] * 1e6 >= 1)
  #@constraint(data.model, [i=1:length(ins),j=1:length(groupsizes)], data.noutdict[ins[i]] / j - groupsizes[j] * 1e6 <= 1)

  #@constraint(data.model, [i=1:length(ins),j=1:length(groupsizes)], data.noutdict[ins[i]] / (maxgroupsize-j+1) + groupsizes[j] * 1e6 >= 1)
  #@constraint(data.model, [i=1:length(ins),j=1:length(groupsizes)], data.noutdict[ins[i]] / (maxgroupsize-j+1) - groupsizes[j] * 1e6 <= 1)

    # Add constraint that nout(l) == n * nin(l) where n is integer
  # "data.noutdict[data.vertex] == data.noutdict[ins[i]] * x" where x is an integer variable is not linear
  # Instead we use the good old big-M strategy to set up a series of "or" constraints (exactly one of them must be true).
  # This is combined with the abs value formulation to force equality.
  # multiplier[j] is false if and only if data.noutdict[data.vertex] == j*data.noutdict[ins[i]]
  # all but one of multiplier must be true
  # Each constraint below is trivially true if multiplier[j] is true since 1e6 is way bigger than the difference between the two variables 
#=   multipliers = @variable(data.model, [1:10], Bin)
  @constraint(data.model, sum(multipliers) == length(multipliers)-1)
  @constraint(data.model, [i=1:length(ins),j=1:length(multipliers)], data.noutdict[data.vertex] - j*data.noutdict[ins[i]] + multipliers[j] * 1e6 >= 0)
  @constraint(data.model, [i=1:length(ins),j=1:length(multipliers)], data.noutdict[data.vertex] - j*data.noutdict[ins[i]] - multipliers[j] * 1e6 <= 0)
 =#
  #@constraint(model, [i=1:ngroups], multipliers[i] == groupsizes[i])

  # Inputs which does not have a variable, possibly because all_in_Δsize_graph did not consider it to be part of the set of vertices which may change
  # We will constrain data.vertex to have integer multiple of its current size
  fixedins = filter(vin -> vin ∉ ins, inputs(data.vertex))
  if !isempty(fixedins) 
    fv_out = @variable(data.model, integer=true)
    @constraint(data.model, [i=1:length(fixedins)], data.noutdict[data.vertex] == nout(fixedins[i]) * fv_out)
  end
end

import JuMP
export testmod
function testmod()
  model = NaiveNASlib.selectmodel(NaiveNASlib.NeuronIndices(), NaiveNASlib.DefaultJuMPΔSizeStrategy(), 3, 4)
  select = @variable(model, [1:8], Bin)
  insert = @variable(model, [1:8], Int)
  ngroups = 2
  maxgroupsize = 4
  groupsizes = @variable(model, [1:maxgroupsize], Bin)
  select_none_in_group = @variable(model,[1:ngroups], Bin)
  insert_none_in_group = @variable(model,[1:ngroups], Bin)
  @constraint(model, sum(groupsizes) == length(groupsizes)-1)

  @constraint(model, 0 .<= insert .<= 100)

  for group in 1:ngroups
    select_in_group = select[group : ngroups : end]
    insert_in_group = insert[group : ngroups : end]

    @constraint(model,  sum(select_in_group) <= 1e6*(1-select_none_in_group[group]))
    #@constraint(model,  sum(insert_in_group) <= 1e6*(1-insert_none_in_group[group]))
    for j in 1:length(groupsizes)
      @constraint(model, sum(select_in_group) + (groupsizes[j] + select_none_in_group[group]) * 1e6 >= maxgroupsize - j + 1)
      @constraint(model, sum(select_in_group) - (groupsizes[j] + select_none_in_group[group]) * 1e6 <= maxgroupsize - j + 1)    
      
      #@constraint(model, sum(insert_in_group) + (groupsizes[j] + insert_none_in_group[group]) * 1e6 >= maxgroupsize - j + 1)
      #@constraint(model, sum(insert_in_group) - (groupsizes[j] + insert_none_in_group[group]) * 1e6 <= maxgroupsize - j + 1)    

      @constraint(model, [i=2:maxgroupsize-j+1], insert_in_group[i] - insert_in_group[i-1] - groupsizes[j] * 1e6 <= 0)
      @constraint(model, [i=2:maxgroupsize-j+1], insert_in_group[i] - insert_in_group[i-1] + groupsizes[j] * 1e6 >= 0)
      @constraint(model, insert_in_group[1] - insert_in_group[maxgroupsize - j + 1] - groupsizes[j] * 1e6 <= 0)
      @constraint(model, insert_in_group[1] - insert_in_group[maxgroupsize - j + 1] + groupsizes[j] * 1e6 >= 0)

      @constraint(model, [i=(maxgroupsize-j+2):maxgroupsize], insert_in_group[i] - groupsizes[j] * 1e6 <= 0)
      @constraint(model, [i=(maxgroupsize-j+2):maxgroupsize], insert_in_group[i] + groupsizes[j] * 1e6 >= 0)
      #@constraint(model, [i=maxgroupsize-j+2:maxgroupsize], insert_in_group[i] == 0)

#=       @constraint(model, insert_in_group[1] + (groupsizes[j] + insert_none_in_group[group]) * 1e6 >= insert_in_group[end]) 
      @constraint(model, insert_in_group[1] - (groupsizes[j] + insert_none_in_group[group]) * 1e6 <= insert_in_group[end]) 
      @constraint(model, [i=2:ngroups], insert_in_group[i] + (groupsizes[j] + insert_none_in_group[group]) * 1e6 >= insert_in_group[i-1]) 
      @constraint(model, [i=2:ngroups], insert_in_group[i] - (groupsizes[j] + insert_none_in_group[group]) * 1e6 <= insert_in_group[i-1])  =#
    end
  end
  #@constraint(model, select_none_in_group[1:1] .== 1)
  @constraint(model, sum(select) + sum(insert) == 9)
  #@constraint(model, sum(select) == 3)
  #@constraint(model, insert[4] == 0)
  #@constraint(model, insert[1:3] .== 1)
  #@constraint(model, [i=2:3], insert[i] .== insert[i-1])

  JuMP.@objective(model, Max, sum(select))

  JuMP.optimize!(model)
  if NaiveNASlib.accept(12, NaiveNASlib.DefaultJuMPΔSizeStrategy(), model)
    @show round.(Int, JuMP.value.(select))
    @show round.(Int, JuMP.value.(select_none_in_group))
    @show round.(Int, JuMP.value.(insert))
    @show round.(Int, JuMP.value.(insert_none_in_group))
    @show round.(Int, JuMP.value.(groupsizes))
  end

end
export testmod2
function testmod2()
  model = NaiveNASlib.selectmodel(NaiveNASlib.NeuronIndices(), NaiveNASlib.DefaultJuMPΔSizeStrategy(), 3, 4)
  select = @variable(model, [1:12], Bin)
  insert = @variable(model, [1:12], Int)
  ngroups = 3
  groupsizes = @variable(model, [1:4], Bin)
  @constraint(model, sum(groupsizes) == length(groupsizes)-1)


  for group in 1:ngroups
    select_in_group = select[group : ngroups : end]
    insert_in_group = insert[group : ngroups : end]
    select_none_in_group = @variable(model, binary=true)

    @constraint(model,  sum(select_in_group) <= 1e6*(1-select_none_in_group))
    for j in 1:length(groupsizes)
      @constraint(model, sum(select_in_group) + (groupsizes[j] + select_none_in_group) * 1e6 >= j)
      @constraint(model, sum(select_in_group) - (groupsizes[j] + select_none_in_group) * 1e6 <= j)    
      
      #@constraint(model, sum(insert_in_group) + (groupsizes[j] + insertgroup[group]) * 1e6 >= j)
      #@constraint(model, sum(insert_in_group) - (groupsizes[j] + insertgroup[group]) * 1e6 <= j)
    end
  end
  #@constraint(model, select_none_in_group[1:1] .== 1)
  @constraint(model, sum(select) == 4)

  JuMP.@objective(model, Max, sum(select))

  JuMP.optimize!(model)
  if NaiveNASlib.accept(12, NaiveNASlib.DefaultJuMPΔSizeStrategy(), model)
    @show round.(Int, JuMP.value.(select))
    #@show round.(Int, JuMP.value.(select_none_in_group))
    @show round.(Int, JuMP.value.(groupsizes))
  end

end