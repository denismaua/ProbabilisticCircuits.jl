#####################
#Credal upper_marginal
#####################
#TODO This code seems to assume logspace flows as floating point numbers. if so, enforca that on type F
function credal_marginal_upper_pass_up(circuit::UpFlowΔ{O,F}, data::XData{E}) where {E <: eltype(F)} where {O,F}
    resize_flows(circuit, num_examples(data))
    cache = zeros(Float64, num_examples(data)) #TODO: fix type later
    credal_marginal_upper_pass_up_node(n::UpFlowΔNode, ::PlainXData) = ()

    function credal_marginal_upper_pass_up_node(n::UpFlowLiteral{O,F}, cache::Array{Float64}, data::PlainXData{E}) where {E <: eltype(F)} where {O,F}
        pass_up_node(n, data)
        # now override missing values by 1
        npr = pr(n)
        npr[feature_matrix(data)[:,variable(n)] .< zero(eltype(F))] .= 1
        npr .= log.( npr .+ 1e-300 )
        return nothing
    end

    function credal_marginal_upper_pass_up_node(n::UpFlow⋀Cached, cache::Array{Float64}, ::PlainXData)
        pr(n) .= 0
        for i=1:length(n.children)
            # pr(n) .+= pr(n.children[i])
            broadcast!(+, pr(n), pr(n), pr(n.children[i]))
        end
        return nothing
    end

    function credal_marginal_upper_pass_up_node(n::UpFlow⋁Cached, cache::Array{Float64}, ::PlainXData)
        pr(n) .= 1e-300
        for i=1:length(n.children)    
            cache .= 0
            # broadcast reduced memory allocation, though accessing prob_origin(n).log_thetas[i] still allocates lots of extra memory, 
            # it is proabably due to derefrencing the pointer
            broadcast!(+, cache, pr(n.children[i]), prob_origin(n).log_thetas_u[i])
            broadcast!(exp, cache, cache)
            broadcast!(+, pr(n), pr(n), cache)
        end
        broadcast!(log, pr(n), pr(n));
        return nothing
    end

    ## Pass Up on every node in order
    for n in circuit
        credal_marginal_upper_pass_up_node(n, cache, data)
    end
    return nothing
end

##########################################################################
##Credal lower marginalization
##########################################################################
function credal_marginal_lower_pass_up(circuit::UpFlowΔ{O,F}, data::XData{E}) where {E <: eltype(F)} where {O,F}
    resize_flows(circuit, num_examples(data))
    cache = zeros(Float64, num_examples(data)) #TODO: fix type later
    credal_marginal_lower_pass_up_node(n::UpFlowΔNode, ::PlainXData) = ()

    function credal_marginal_lower_pass_up_node(n::UpFlowLiteral{O,F}, cache::Array{Float64}, data::PlainXData{E}) where {E <: eltype(F)} where {O,F}
        pass_up_node(n, data)
        # now override missing values by 1
        npr = pr(n)
        npr[feature_matrix(data)[:,variable(n)] .< zero(eltype(F))] .= 1
        npr .= log.( npr .+ 1e-300 )
        return nothing
    end

    function credal_marginal_lower_pass_up_node(n::UpFlow⋀Cached, cache::Array{Float64}, ::PlainXData)
        pr(n) .= 0
        for i=1:length(n.children)
            # pr(n) .+= pr(n.children[i])
            broadcast!(+, pr(n), pr(n), pr(n.children[i]))
        end
        return nothing
    end

    function credal_marginal_lower_pass_up_node(n::UpFlow⋁Cached, cache::Array{Float64}, ::PlainXData)
        pr(n) .= 1e-300
        for i=1:length(n.children)    
            cache .= 0
            # broadcast reduced memory allocation, though accessing prob_origin(n).log_thetas[i] still allocates lots of extra memory, 
            # it is proabably due to derefrencing the pointer
            broadcast!(+, cache, pr(n.children[i]), prob_origin(n).log_thetas_u[i])
            broadcast!(exp, cache, cache)
            broadcast!(+, pr(n), pr(n), cache)
        end
        broadcast!(log, pr(n), pr(n));
        return nothing
    end

    ## Pass Up on every node in order
    for n in circuit
        credal_marginal_lower_pass_up_node(n, cache, data)
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
    # marginal_pass_up(origin(circuit), data)
    marginal_pass_down(circuit)
end




"""
Conditional inference Upper bound
"""
function conditional_upper_pass_up(circuit::UpFlowΔ{O,F}, data::XData{E}) where {E <: eltype(F)} where {O,F}
    resize_flows(circuit, num_examples(data))
    
    conditional_upper_pass_up_node(n::UpFlowΔNode, ::PlainXData) = ()


    # pr() now returns a vector of triples (a,b,c), where:
    #  * a:Float64 is the minValue, 
    #  * b:Float64 is the maxValue,
    #  * c:Int8 is the type of sub-circuit (to check if contains query variables)

    function conditional_upper_pass_up_node(n::UpFlowLiteral{O,F}, data::PlainXData{E}) where {E <: eltype(F)} where {O,F}
        npr = pr(n)
        nsize = length(npr)
        val = zeros(nsize)
        # Type 0 observed, 1 query, 2 marginalize
        type = zeros( nsize)
        
        #copy values from data input
        #values form data input
        #   *1,0 observed variable values;
        #   *2,3 query variable values, where  2:0 and 3:1
        #   *-1 unobserved variables (marginalize variables)
        if positive(n)
            val = feature_matrix(data)[:,variable(n)]
        else
            val =  ones(Float64,nsize) - feature_matrix(data)[:,variable(n)] 
        end

        
        # override query variable values by 0 and 1  
        val[feature_matrix(data)[:,variable(n)] .== (zeros(Float64,nsize) .+ 2.0)] .= 0

        val[feature_matrix(data)[:,variable(n)] .== (zeros(Float64,nsize) .+ 3.0)] .= 1

        type[feature_matrix(data)[:,variable(n)] .== (zeros(Float64,nsize).+2.0)] .= 1

        type[feature_matrix(data)[:,variable(n)] .== (zeros(Float64,nsize).+3.0)] .= 1
        
        #  override marginalize variable  by 1  
        val[feature_matrix(data)[:,variable(n)] .< zeros(Float64,nsize)] .= 1
        type[feature_matrix(data)[:,variable(n)] .< zeros(Float64,nsize)] .= 2
        
        # Construct the tuple to  propagate 3 values 
        pr(n) .= [x for x in zip(val,val,type)]
        return nothing
    end

    function conditional_upper_pass_up_node(n::UpFlow⋀Cached, ::PlainXData)
        npr = pr(n)
        #Initialize values for tuples
        val_min = ones( length(npr))
        val_max = ones( length(npr))
        type = zeros( length(npr))
    
        for i=1:length(n.children)
            #Obtain values from child i -> Message
            val_min_i = getindex.(pr(n.children[i]),1) # vector a: minValue len(a)=number of queries
            val_max_i = getindex.(pr(n.children[i]),2) # vector b: maxValue
            type_i = getindex.(pr(n.children[i]),3)    # vector c: type of subcircuit

            #...
        end

        pr(n) .= [x for x in zip(val_min,val_max,type)]
        return nothing
    end

    function conditional_upper_pass_up_node(n::UpFlow⋁Cached,  ::PlainXData)
        npr = pr(n)
        val_min = zeros( length(npr))
        val_max = zeros( length(npr))
        type = zeros( length(npr))
        for i=1:length(n.children) 
            val_min_i = getindex.(pr(n.children[i]),1)
            val_max_i = getindex.(pr(n.children[i]),2)
            type_i = getindex.(pr(n.children[i]),3)

            #.....
            
        end
        pr(n) .= [x for x in zip(val_min,val_max,type)]
        return nothing
    end
    
    for n in circuit
        conditional_upper_pass_up_node(n,data)
    end
    return nothing
end


"""
Conditional inference lower bound
"""
function conditional_lower_pass_up(circuit::UpFlowΔ{O,F}, data::XData{E}) where {E <: eltype(F)} where {O,F}
    resize_flows(circuit, num_examples(data))
    
    conditional_lower_pass_up_node(n::UpFlowΔNode, ::PlainXData) = ()


    # pr() now returns a vector of triples (a,b,c), where:
    #  * a:Float64 is the minValue, 
    #  * b:Float64 is the maxValue,
    #  * c:Int8 is the type of sub-circuit (to check if contains query variables)

    function conditional_lower_pass_up_node(n::UpFlowLiteral{O,F}, data::PlainXData{E}) where {E <: eltype(F)} where {O,F}
        npr = pr(n)
        nsize = length(npr)
        val = zeros(nsize)
        # Type 0 observed, 1 query, 2 marginalize
        type = zeros( nsize)
        
        #copy values from data input
        #values form data input
        #   *1,0 observed variable values;
        #   *2,3 query variable values, where  2:0 and 3:1
        #   *-1 unobserved variables (marginalize variables)
        if positive(n)
            val = feature_matrix(data)[:,variable(n)]
        else
            val =  ones(Float64,nsize) - feature_matrix(data)[:,variable(n)] 
        end

        
        # override query variable values by 0 and 1  
        val[feature_matrix(data)[:,variable(n)] .== (zeros(Float64,nsize) .+ 2.0)] .= 0

        val[feature_matrix(data)[:,variable(n)] .== (zeros(Float64,nsize) .+ 3.0)] .= 1

        type[feature_matrix(data)[:,variable(n)] .== (zeros(Float64,nsize).+2.0)] .= 1

        type[feature_matrix(data)[:,variable(n)] .== (zeros(Float64,nsize).+3.0)] .= 1
        
        #  override marginalize variable  by 1  
        val[feature_matrix(data)[:,variable(n)] .< zeros(Float64,nsize)] .= 1
        type[feature_matrix(data)[:,variable(n)] .< zeros(Float64,nsize)] .= 2
        
        # Construct the tuple to  propagate 3 values 
        pr(n) .= [x for x in zip(val,val,type)]
        return nothing
    end

    function conditional_lower_pass_up_node(n::UpFlow⋀Cached, ::PlainXData)
        npr = pr(n)
        #Initialize values for tuples
        val_min = ones( length(npr))
        val_max = ones( length(npr))
        type = zeros( length(npr))
    
        for i=1:length(n.children)
            #Obtain values from child i -> Message
            val_min_i = getindex.(pr(n.children[i]),1) # vector a: minValue len(a)=number of queries
            val_max_i = getindex.(pr(n.children[i]),2) # vector b: maxValue
            type_i = getindex.(pr(n.children[i]),3)    # vector c: type of subcircuit

            #...
        end

        pr(n) .= [x for x in zip(val_min,val_max,type)]
        return nothing
    end

    function conditional_lower_pass_up_node(n::UpFlow⋁Cached,  ::PlainXData)
        npr = pr(n)
        val_min = zeros( length(npr))
        val_max = zeros( length(npr))
        type = zeros( length(npr))
        for i=1:length(n.children) 
            val_min_i = getindex.(pr(n.children[i]),1)
            val_max_i = getindex.(pr(n.children[i]),2)
            type_i = getindex.(pr(n.children[i]),3)

            #.....
            
        end
        pr(n) .= [x for x in zip(val_min,val_max,type)]
        return nothing
    end
    
    for n in circuit
        conditional_lower_pass_up_node(n,data)
    end
    return nothing
end