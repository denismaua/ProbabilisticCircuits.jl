#####################

#TODO This code seems to assume logspace flows as floating point numbers. if so, enforca that on type F
function marginal_pass_up(circuit::UpFlowΔ{O,F}, data::XData{E}) where {E <: eltype(F)} where {O,F}
    resize_flows(circuit, num_examples(data))
    cache = zeros(Float64, num_examples(data)) #TODO: fix type later
    marginal_pass_up_node(n::UpFlowΔNode, ::PlainXData) = ()

    function marginal_pass_up_node(n::UpFlowLiteral{O,F}, cache::Array{Float64}, data::PlainXData{E}) where {E <: eltype(F)} where {O,F}
        pass_up_node(n, data)
        # now override missing values by 1
        npr = pr(n)
        npr[feature_matrix(data)[:,variable(n)] .< zero(eltype(F))] .= 1
        npr .= log.( npr .+ 1e-300 )
        return nothing
    end

    function marginal_pass_up_node(n::UpFlow⋀Cached, cache::Array{Float64}, ::PlainXData)
        pr(n) .= 0
        for i=1:length(n.children)
            # pr(n) .+= pr(n.children[i])
            broadcast!(+, pr(n), pr(n), pr(n.children[i]))
        end
        return nothing
    end

    function marginal_pass_up_node(n::UpFlow⋁Cached, cache::Array{Float64}, ::PlainXData)
        pr(n) .= 1e-300
        for i=1:length(n.children)    
            cache .= 0
            # broadcast reduced memory allocation, though accessing prob_origin(n).log_thetas[i] still allocates lots of extra memory, 
            # it is proabably due to derefrencing the pointer
            broadcast!(+, cache, pr(n.children[i]), prob_origin(n).log_thetas[i])
            broadcast!(exp, cache, cache)
            broadcast!(+, pr(n), pr(n), cache)
        end
        broadcast!(log, pr(n), pr(n));
        return nothing
    end

    ## Pass Up on every node in order
    for n in circuit
        marginal_pass_up_node(n, cache, data)
    end
    return nothing
end



##### marginal_pass_down

function marginal_pass_down(circuit::DownFlowΔ{O,F}) where {O,F}
    resize_flows(circuit, flow_length(origin(circuit)))
    for n in circuit
        reset_downflow_in_progress(n)
    end
    for downflow in downflow_sinks(circuit[end])
        # initialize root flows to 1
        downflow.downflow .= one(eltype(F))
    end
    for n in Iterators.reverse(circuit)
        marginal_pass_down_node(n)
    end
end

marginal_pass_down_node(n::DownFlowΔNode) = () # do nothing
marginal_pass_down_node(n::DownFlowLeaf) = ()

function marginal_pass_down_node(n::DownFlow⋀Cached)
    # todo(pashak) might need some changes, not tested, also to convert to logexpsum later
     # downflow(n) = EF_n(e), the EF for edges or leaves are note stored
    for c in n.children
        for sink in downflow_sinks(c)
            if !sink.in_progress
                sink.downflow .= downflow(n)
                sink.in_progress = true
            else
                sink.downflow .+= downflow(n)
            end
        end
    end
end

function marginal_pass_down_node(n::DownFlow⋁Cached)
    # todo(pashak) might need some changes, not tested, also to convert to logexpsum later
    # downflow(n) = EF_n(e), the EF for edges or leaves are note stored
    for (ind, c) in enumerate(n.children)
        for sink in downflow_sinks(c)
            if !sink.in_progress
                sink.downflow .= downflow(n) .* exp.(prob_origin(n).log_thetas[ind] .+ pr(origin(c)) .- pr(origin(n)) )
                sink.in_progress = true
            else
                sink.downflow .+= downflow(n) .* exp.(prob_origin(n).log_thetas[ind] .+ pr(origin(c)) .- pr(origin(n)))
            end
        end
    end
end

#### marginal_pass_up_down

function marginal_pass_up_down(circuit::DownFlowΔ{O,F}, data::XData{E}) where {E <: eltype(F)} where {O,F}
    @assert !(E isa Bool)
    marginal_pass_up(origin(circuit), data)
    marginal_pass_down(circuit)
end

"""
Class dominane with missing values

"""
function dominance_pass_up(circuit::UpFlowΔ{O,F}, data::XData{E}, y::Integer) where {E <: eltype(F)} where {O,F}
    resize_flows(circuit, num_examples(data))
    node_cache = Dict{UpFlowΔNode, Int64}()
    for (i, n) in enumerate(circuit)
        node_cache[n] = i
    end
    # pr() is  a vector of triples (a,b,c), where a:Int8 is the type of sub-circuit,b:Float64 is minval, c:Float is the maxval 
    dominance_pass_up_node(n::UpFlowΔNode, ::PlainXData,  y::Integer) = ()
    function dominance_pass_up_node(n::UpFlowLiteral{O,F}, data::PlainXData{E}, y::Integer) where {E <: eltype(F)} where {O,F}
        nsize =  num_examples(data)
        npr = pr(n)
        val = zeros( length(npr))
        if positive(n)
            val = feature_matrix(data)[:,variable(n)]
        else
            val =  ones(Float64,length(npr)) - feature_matrix(data)[:,variable(n)] # using views here strangely seems to save a few allocations and a little time
        end
        type = zeros( length(npr))
        # Type 0 observed, 1 class, 2 missing
        #  override missing values by -1  
        val[feature_matrix(data)[:,variable(n)] .< zeros(Float64,length(npr))] .= 1
        type[feature_matrix(data)[:,variable(n)] .< zeros(Float64,length(npr))] .= 2
        # override class Y values by 1 and -1  
        if y in variable(n)
            if positive(n)
                val .= 1
                type .= 1
            else
                val .= -1
                type .= 1
            end
        end
        pr(n) .= [x for x in zip(val,val,type)]
        # println("i ", variable(n) , "  val ",pr(n))
        return nothing
    end

    function dominance_pass_up_node(n::UpFlow⋀Cached, ::PlainXData, y::Integer)
        # println(" *** ")
        npr = pr(n)
        val_min = ones( length(npr))
        val_max = ones( length(npr))
        type = zeros( length(npr))
        index_max = zeros(Bool,num_examples(data)) # neg class misture missing
        index_class = zeros(Bool,num_examples(data))
        index_obs = zeros(Bool,num_examples(data)) #Observed and class
        index_min = zeros(Bool,num_examples(data)) # missing and observables 
        # print("-***- ", node_cache[n])
        for i=1:length(n.children)
            val_min_i = getindex.(pr(n.children[i]),1)
            val_max_i = getindex.(pr(n.children[i]),2)
            type_i = getindex.(pr(n.children[i]),3)
            #Verify if some sub-circuit is class and is negative 
            index_max .= ((val_min_i .< 0.0) .& (type_i .== 1.0)) .| index_max
            index_class .= ((val_min_i .>= 0.0) .& (type_i .== 1.0)) .| index_class
            # println(" max ",val_min ," t ",type_i, index_max )
            index_min .= (.!(index_max .| index_class)) .& ((type_i .== 2.0).| index_min)
        end
        # println("indexmin", index_min)
        index_obs .= .!(index_max .| index_class .| index_min)
        type[index_min] .= 2.0
        type[index_max .| index_class] .= 1.0
        # type[index_obs] .= 0.0
        for i=1:length(n.children)
            val_min_i = getindex.(pr(n.children[i]),1)
            val_max_i = getindex.(pr(n.children[i]),2)
            # Combining negative class and Missing or observed
                #The child with the neg class   
            val_min[(index_max).& (val_min_i .< 0.0)] .= val_min[(index_max).& (val_min_i .< 0.0)] .* val_min_i[(index_max).& (val_min_i .< 0.0)]
                #The child with the othres types of variables 
            val_min[(index_max).& (val_min_i .>= 0.0)] .= val_min[(index_max).& (val_min_i .>= 0.0)] .* val_max_i[(index_max).& (val_min_i .>= 0.0)]
                # Valman=valmin= valmin(i)*valmax(j)
            val_max[index_max] .= val_min[index_max]
            
            # Combining possitive classes and/or  observed
            val_min[index_class.& index_obs] .= val_min[index_class.& index_obs] .* val_min_i[index_class.& index_obs]
            val_max[index_class.& index_obs] .= val_min[index_class.& index_obs]
            
            #Combining with missing variables
            val_min[index_min] .= val_min[index_min] .* val_min_i[index_min]
            val_max[index_min] .= val_max[index_min] .* val_max_i[index_min]

            #Combining only observable 
            val_min[index_class] .= val_min[index_class] .* val_min_i[index_class]
            val_max[index_class] .= val_min[index_class]  
            # print( "   arg ",i,"  ",pr(n.children[i]))
            # pr(n) .*= pr(n.children[i])
        end
        pr(n) .= [x for x in zip(val_min,val_max,type)]
        # println("  * ",pr(n))
        return nothing
    end

    function dominance_pass_up_node(n::UpFlow⋁Cached,  ::PlainXData, y::Integer)
        # println("  +++ ")
        # print("-+++- ", node_cache[n])
        npr = pr(n)
        prs_min = zeros( length(n.children), num_examples(data))
        prs_max = zeros( length(n.children), num_examples(data))
        index_min = zeros( length(npr)) #Missing values and observed type 2
        index_sum = zeros( length(npr)) # Observed and/or class values  type 0 or 1
        val_min = zeros( length(npr))
        val_max = zeros( length(npr))
        type = zeros( length(npr))
        for i=1:length(n.children) 
            val_min_i = getindex.(pr(n.children[i]),1)
            val_max_i = getindex.(pr(n.children[i]),2)
            type_i = getindex.(pr(n.children[i]),3)
            index_min = (type_i .== 2.0)
            index_sum = (type_i .== 1.0) .| (type_i .== 0.0)
            type .= type_i
            weight = exp.(prob_origin(n).log_thetas[i])          
            prs_min[i,:] .= val_min_i .* weight
            prs_max[i,:] .= val_max_i .* weight
            # print( "   arg ",i,"  ",pr(n.children[i]))
        end
        val_min[index_min] .= minimum(prs_min[:,index_min], dims = 1)[1,:]
        val_max[index_min] .= maximum(prs_max[:,index_min], dims = 1)[1,:]

        val_min[index_sum] .= sum(prs_min[:,index_sum], dims = 1)[1,:]
        val_max[index_sum] .= sum(prs_max[:,index_sum], dims = 1)[1,:]
        pr(n) .= [x for x in zip(val_min,val_max,type)]
        # print("-- ", node_cache[n])
        # println("  +    ",pr(n))
        return nothing
    end
    
    for n in circuit
        dominance_pass_up_node(n,data, y)
    end
    return nothing
end

