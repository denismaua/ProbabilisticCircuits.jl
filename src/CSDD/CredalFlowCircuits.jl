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
        n.pr .= 1e-300
        #pr(n) .= 1e-300

        # for i=1:length(n.children) 
        #     cache .= 0
        #     #broadcast reduced memory allocation, though accessing prob_origin(n).log_thetas[i] still allocates lots of extra memory, 
        #     #it is proabably due to derefrencing the pointer
        #     broadcast!(+, cache, pr(n.children[i]), prob_origin(n).log_thetas_u[i])
        #     broadcast!(exp, cache, cache)
        #     broadcast!(+, pr(n), pr(n), cache)
        # end

        # broadcast!(log, pr(n), pr(n));
        #u = exp(prob_origin(n).log_thetas_u)
        #u = prob_origin(n).log_thetas_u
        u = Array{Float64}(undef, length(n.children), num_examples(data))
        c = Array{Float64}(undef, length(n.children), num_examples(data))
        
        # for i=1:length(n.children) 
        #     for j=1:num_examples(data)
        #         u[i,j] =  exp(prob_origin(n).log_thetas_u[i])
        #     end
        # end

        ### sotto versione broadcastata

        for j=1:num_examples(data)
            u[:,j] .=  exp.(prob_origin(n).log_thetas_u)
        end

        
        #println(u)
        # for i=1:length(n.children) 
        #     for j=1:num_examples(data)
             
        #      c[i,j] = exp(n.children[i].pr[j]) ## TODO log is monotone + broadcast
        #     end
        # end

        ### sotto versione broadcastata

        for j=1:num_examples(data) 
            c[:,j] .= exp.(pr.(n.children)[j]) ## TODO log is monotone + broadcast
        end


        # for i=1:length(n.children)
        #     println(n.children[i].pr)
        # end
        # println(c)
        
        optx = Array{Float64}(undef, length(n.children), num_examples(data))
        optx = copy(u)

        for j=1:num_examples(data)
            
            for i=1:length(n.children)
                if c[i,j] == sort(c[:,j])[1]
                    optx[i,j] = 0.0
                    somma = sum(optx[:,j])
                    optx[i,j] = 1.0-somma ## TODO compatta questo pezzo con find o findall 
                    break
                end
            end
        end

        # println(c)
        # println(u)
        # println(optx[:,1])
        # println(sum(optx[:,1]))
        
        for j=1:num_examples(data)
            pr(n)[j] = log(optx[:,j]'c[:,j])
        end
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
        l = Array{Float64}(undef, length(n.children), num_examples(data))
        c = Array{Float64}(undef, length(n.children), num_examples(data))
        
        # for i=1:length(n.children) 
        #     for j=1:num_examples(data)
        #         l[i,j] =  exp(prob_origin(n).log_thetas[i])
        #     end
        # end

        for j=1:num_examples(data)
            l[:,j] .=  exp.(prob_origin(n).log_thetas)
        end


        
        # for i=1:length(n.children) 
        #     for j=1:num_examples(data)
             
        #      c[i,j] = exp(n.children[i].pr[j])
        #     end
        # end

        for j=1:num_examples(data) 
            c[:,j] .= exp.(pr.(n.children)[j]) ## TODO log is monotone + broadcast
        end
        
        
        optx = Array{Float64}(undef, length(n.children), num_examples(data))
        optx = copy(l)


        for j=1:num_examples(data)
            
            for i=1:length(n.children)
                if c[i,j] == sort(c[:,j])[1]
                    optx[i,j] = 0.0
                    somma = sum(optx[:,j])
                    optx[i,j] = 1.0-somma ## TODO compatta questo pezzo con find o findall 
                    break
                end
            end
        end
        
        
        #println(sum(optx[:,1])) this is not exactly 1... approx stuff? or is there something wrong in the code?
       
        for j=1:num_examples(data)
            pr(n)[j] = log(optx[:,j]'c[:,j])
        end
        return nothing
    
    end

    # function credal_marginal_lower_pass_up_node(n::UpFlow⋁Cached, cache::Array{Float64}, ::PlainXData)
    #     pr(n) .= 1e-300
    #     for i=1:length(n.children)    
    #         cache .= 0
    #         # broadcast reduced memory allocation, though accessing prob_origin(n).log_thetas[i] still allocates lots of extra memory, 
    #         # it is proabably due to derefrencing the pointer
    #         broadcast!(+, cache, pr(n.children[i]), prob_origin(n).log_thetas_u[i])
    #         broadcast!(exp, cache, cache)
    #         broadcast!(+, pr(n), pr(n), cache)
    #     end
    #     broadcast!(log, pr(n), pr(n));
    #     return nothing
    # end

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
    marginal_pass_up(origin(circuit), data)
    marginal_pass_down(circuit)
end
