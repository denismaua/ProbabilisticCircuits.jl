#####################
#Credal upper_marginal
#####################
#TODO This code seems to assume logspace flows as floating point numbers. if so, enforca that on type F

##################### for local LP ###########################################
using JuMP
using Clp
#using LinearAlgebra

function minimi(coeff :: Array{Float64,1}, l_bounds :: Array{Float64,1}, u_bounds :: Array{Float64,1})
# cosi poi gli entro in broadcasting c, exp.(prob_origin(n).log_thetas), exp.(prob_origin(n).log_thetas_u)
    my_model = Model(Clp.Optimizer)
    set_optimizer_attribute(my_model, "LogLevel", 1)
     set_optimizer_attribute(my_model, "Algorithm", 4)


     @variable(my_model, l_bounds[i] <= x[i = 1:length(l_bounds)]<= u_bounds[i]) 
     @objective(my_model,Min,coeff' *x) # perché anche * ?
     @constraint(my_model, normalization, ones(Float64, length(l_bounds))'x == 1)
    
        
     optimize!(my_model)
        # println("Optimal Solutions:")
        # println("x = ", JuMP.value.(x))
        #  println("Optimal Value:")
        #  println("min =", JuMP.objective_value(my_model))
        #  #getobjectivevalue
     println("funzia")
     return JuMP.objective_value(my_model)     
end      



#############################################################


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
        
        # Construct the tuple to  propagate 4 values : conditional message, marginal upper message, marginal lower message, type
        pr(n) .= [x for x in zip(val,val,val,type)]
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

        pr(n) .= [x for x in zip(val_min,val_max,val_max,type)]
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
        pr(n) .= [x for x in zip(val_min,val_max,val_max,type)]
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
function conditional_lower_pass_up(circuit::UpFlowΔ{O,F}, data::XData{E}, mu::Array{Float64,1}) where {E <: eltype(F)} where {O,F}
    resize_flows(circuit, num_examples(data))
    
    conditional_lower_pass_up_node(n::UpFlowΔNode, ::PlainXData, mu::Array{Float64,1}) = ()


    # pr() now returns a vector of triples (a,b,c), where:
    #  * a:Float64 is the minValue, 
    #  * b:Float64 is the maxValue,
    #  * c:Int8 is the type of sub-circuit (to check if contains query variables)

    function conditional_lower_pass_up_node(n::UpFlowLiteral{O,F}, data::PlainXData{E}, mu::Array{Float64,1}) where {E <: eltype(F)} where {O,F}
        
        println("LITERAL NODE")

        npr = pr(n)
        #mu = 1.0 #  se metto 0.6 mi da ERROR perché ???   # dopo mu::Array{Float64,1}
        nsize = length(npr)
        val = zeros(nsize)           # primo el tuple
        marginal_lower = zeros(nsize) # secondo el tuple
        marginal_upper = zeros(nsize) # terzo el tuple
        # Type 0 observed, 1 query, 2 marginalize
        type = zeros( nsize)
        
        ################ MARGINAL MESSAGES ################################################
        ### qui preparo il messaggio marginal_lower 
        #####pass_up_node(n, data) questo non funge
        # now override missing values by 1
        #######npr_marginal_lower = pr(n)
        #######npr_marginal_lower[feature_matrix(data)[:,variable(n)] .< zero(eltype(F))] .= 1
       ######## npr_marginal_lower .= log.( npr_marginal_lower .+ 1e-300 )
        



        ###################################################################################

           
       
        
        
        #copy values from data input
        #values form data input
        #   *1,0 observed variable values;
        #   *2,3 query variable values, where  2:0 and 3:1
        #   *-1 unobserved variables (marginalize variables)
        if positive(n) ### qui non ci sono nodi top ricorda che i top diventano or !!! quindi n è o pos o neg

            marginal_lower = feature_matrix(data)[:,variable(n)] #marginal_lower
            marginal_upper = feature_matrix(data)[:,variable(n)] #marginal_lower
            val = feature_matrix(data)[:,variable(n)]   
        else

            marginal_lower =  ones(Float64,nsize) - feature_matrix(data)[:,variable(n)] ### TODO in verità se non è queried posso mettere un valore qualsiasi come per quelle da marginalizzare, vedi pseudocode
            marginal_upper =  ones(Float64,nsize) - feature_matrix(data)[:,variable(n)] ### TODO in verità se non è queried posso mettere un valore qualsiasi come per quelle da marginalizzare, vedi pseudocode
            val =  ones(Float64,nsize) - feature_matrix(data)[:,variable(n)] ### TODO in verità se non è queried posso mettere un valore qualsiasi come per quelle da marginalizzare, vedi pseudocode
        end ### sistemato il caso in cui la variabile sia una var osservat
        


            # override query variable values by 0 and 1, depending both on the queried status and the literal status
        if positive(n)
            #println("LITERAL POSITIVE NODE")


            val[feature_matrix(data)[:,variable(n)] .== (zeros(Float64,nsize) .+ 2.0)] .= (zeros(Float64,nsize) .- mu)[feature_matrix(data)[:,variable(n)] .== (zeros(Float64,nsize) .+ 2.0)]

            val[feature_matrix(data)[:,variable(n)] .== (zeros(Float64,nsize) .+ 3.0)] .= (ones(Float64,nsize) .- mu)[feature_matrix(data)[:,variable(n)] .== (zeros(Float64,nsize) .+ 3.0)]

            marginal_lower[feature_matrix(data)[:,variable(n)] .== (zeros(Float64,nsize) .+ 2.0)] .= 0.0
            marginal_upper[feature_matrix(data)[:,variable(n)] .== (zeros(Float64,nsize) .+ 2.0)] .= 0.0

            marginal_lower[feature_matrix(data)[:,variable(n)] .== (zeros(Float64,nsize) .+ 3.0)] .= 1.0
            marginal_upper[feature_matrix(data)[:,variable(n)] .== (zeros(Float64,nsize) .+ 3.0)] .= 1.0

            
        else

            #println("LITERAL NEGATIVE NODE")

            val[feature_matrix(data)[:,variable(n)] .== (zeros(Float64,nsize) .+ 2.0)] .= (ones(Float64,nsize) .- mu)[feature_matrix(data)[:,variable(n)] .== (zeros(Float64,nsize) .+ 2.0)]

            val[feature_matrix(data)[:,variable(n)] .== (zeros(Float64,nsize) .+ 3.0)] .= (zeros(Float64,nsize) .- mu)[feature_matrix(data)[:,variable(n)] .== (zeros(Float64,nsize) .+ 3.0)]

            marginal_lower[feature_matrix(data)[:,variable(n)] .== (zeros(Float64,nsize) .+ 2.0)] .= 1.0
            marginal_upper[feature_matrix(data)[:,variable(n)] .== (zeros(Float64,nsize) .+ 2.0)] .= 1.0

            marginal_lower[feature_matrix(data)[:,variable(n)] .== (zeros(Float64,nsize) .+ 3.0)] .= 0.0
            marginal_upper[feature_matrix(data)[:,variable(n)] .== (zeros(Float64,nsize) .+ 3.0)] .= 0.0
        end


         # type of queried variables (independent both from the query status and the literal status, correct)

         type[feature_matrix(data)[:,variable(n)] .== (zeros(Float64,nsize).+2.0)] .= 1

         type[feature_matrix(data)[:,variable(n)] .== (zeros(Float64,nsize).+3.0)] .= 1




         #  override marginalize variable  by 1, independently from queried status and literal status
         val[feature_matrix(data)[:,variable(n)] .< zeros(Float64,nsize)] .= 1
         marginal_lower[feature_matrix(data)[:,variable(n)] .< zeros(Float64,nsize)] .= 1
         marginal_upper[feature_matrix(data)[:,variable(n)] .< zeros(Float64,nsize)] .= 1



         # type of variables to marginalize
         type[feature_matrix(data)[:,variable(n)] .< zeros(Float64,nsize)] .= 2



         # TODO sistemare i marginal per la questione log

         #marginal_lower .= log.(marginal_lower .+ 1e-300 )
         #marginal_upper .= log.(marginal_upper .+ 1e-300 )
        
         # Construct the tuple to  propagate 4 values : conditional message, marginal lower message, marginal upper message, type
         pr(n) .= [x for x in zip(val,marginal_lower,marginal_upper,type)]
         println("tupla literal : ", pr(n))

         return nothing
    end

    function conditional_lower_pass_up_node(n::UpFlow⋀Cached, ::PlainXData, mu::Array{Float64,1})

        println("AND NODE")


    ############ MARGINAL STUFF ############
    # function credal_marginal_lower_pass_up_node(n::UpFlow⋀Cached, cache::Array{Float64}, ::PlainXData)
    #     pr(n) .= 0
    #     for i=1:length(n.children)
    #         # pr(n) .+= pr(n.children[i])
    #         broadcast!(+, pr(n), pr(n), pr(n.children[i]))
    #     end
    #     return nothing
    # end


    #######################################


        npr = pr(n)
        #Initialize values for tuples
        val_min = ones( length(npr))
        marginal_lower = zeros( length(npr))
        marginal_upper = ones( length(npr))
        type = zeros( length(npr))
        vect = ones( length(npr))

    
        for i=1:length(n.children) # che per noi è 2 perché siamo in un AND node
    
            #### Marginal messages: TODO work with their log
            broadcast!(*, marginal_lower, marginal_lower, getindex.(pr(n.children[i]),2))
            broadcast!(*, marginal_upper, marginal_upper, getindex.(pr(n.children[i]),3))
            

            ##### conditional message (val_min) is the product of pi_lower and marginal_lower/upper of the children, depending of the types of the latter
            # val_min for now has all entries set to 1
            # start by giving pi_lower values when the type of the children i (in which we are in the for cycle) is 1
            val_min[getindex.(pr(n.children[i]),4) .== ones( length(npr))] .= getindex.(pr(n.children[i]),1)[getindex.(pr(n.children[i]),4) .== ones( length(npr))]

            
            ##### type. as soon as I see a children with type 1, set type = 1. otherwise it stays 0. IN AND/OR NODES TYPE IS ONLY 1 OR 0 !
            type[getindex.(pr(n.children[i]),4) .== ones( length(npr))] .= 1

        
        end

       
        # multiply val_min by the correct marginal 
        val_min[(getindex.(pr(n.children[1]),4) .!== ones( length(npr))) .& (getindex.(pr(n.children[2]),1) .> zeros( length(npr))) ].*= getindex.(pr(n.children[1]),2)[(getindex.(pr(n.children[1]),4) .!== ones( length(npr))) .& (getindex.(pr(n.children[2]),1) .> zeros( length(npr))) ]
        val_min[(getindex.(pr(n.children[1]),4) .!== ones( length(npr))) .& (getindex.(pr(n.children[2]),1) .< zeros( length(npr))) ].*= getindex.(pr(n.children[1]),3)[(getindex.(pr(n.children[1]),4) .!== ones( length(npr))) .& (getindex.(pr(n.children[2]),1) .< zeros( length(npr))) ]

        val_min[(getindex.(pr(n.children[2]),4) .!== ones( length(npr))) .& (getindex.(pr(n.children[1]),1) .> zeros( length(npr))) ].*= getindex.(pr(n.children[2]),2)[(getindex.(pr(n.children[2]),4) .!== ones( length(npr))) .& (getindex.(pr(n.children[1]),1) .> zeros( length(npr))) ]
        val_min[(getindex.(pr(n.children[2]),4) .!== ones( length(npr))) .& (getindex.(pr(n.children[1]),1) .< zeros( length(npr))) ].*= getindex.(pr(n.children[2]),3)[(getindex.(pr(n.children[2]),4) .!== ones( length(npr))) .& (getindex.(pr(n.children[1]),1) .< zeros( length(npr))) ]


        pr(n) .= [x for x in zip(val_min,marginal_lower,marginal_upper,type)]
        return nothing
    end

    function conditional_lower_pass_up_node(n::UpFlow⋁Cached,  ::PlainXData, mu::Array{Float64,1})
        println("OR NODE")

        npr = pr(n)
        val_min = zeros( length(npr))
        marginal_lower = zeros( length(npr))
        marginal_upper = zeros( length(npr))
        type = zeros( length(npr))
        
        if n.children[1] isa UpFlowLiteral

            coeff = Array{Float64}(undef, length(npr), length(n.children)) # length(n.children) dev'essere sempre 2, assert?
            println("val_min_prima = ", val_min)


            for i=1:2

                # println("positivity of children ", i, positive(n.children[i]))
                # println("tripletta i-esimo children, per ogni instance, con i= ", i, " : " ,   pr(n.children[i]))
                # println("lower bounds : ", exp.(prob_origin(n).log_thetas)) ### questi sono gli stessi per ogni istanza
                # println("upper bounds : ", exp.(prob_origin(n).log_thetas_u))

                coeff[:,i] .= getindex.(pr(n.children[i]),1) 
                
            end

            for i=1:length(npr) ### TODO broadcast this
                val_min[i] = minimi(coeff[i,:], exp.(prob_origin(n).log_thetas), exp.(prob_origin(n).log_thetas_u))

            end 

             println("val_min_dopo = ", val_min)
             println("IT'S A TOP !")
        else 
        
             # for j=1:length(n.children)   
            #     val_min_j = getindex.(pr(n.children[j]),1)
            #     val_max_j = getindex.(pr(n.children[j]),2)
            #     type_j = getindex.(pr(n.children[j]),3) # è un vettore in cui la k-esima entry mi dice il tipo del i-esimo child wrt la k-esima istanza
            #     left_index = zeros( length(npr))
            #     right_index = zeros( length(npr))

            #     # deduco il type di n dal type dei suoi figli. ridondante rifarlo per ogni children 
            #     # visto che tutti i type_j sono uguali (sono i tipi degli elementi del nodo or) ma vabbè

            #     type[type_j .== ones(Int8,length(npr))] .= 1 

                        
            #     # dato il j-esimo children, mi memorizzo, per ogni istanza, se la queried variable è una left var o right var.
            #     # se la queried var è in n (ie, se la i-esima entry di type di n è uguale a 1), left_index e right_index devono sommare a 1 in quella entry.
            #     # altrimenti la queried var non è né left né right (ie, la i-esima entry di type di n è uguale a 0). 
            #     # In tal caso ( e solo in tal caso) left_index e right_index devono sommare a 0 in quella entry. condizione da usare dopo per settare npr !!! :) 
            #     #println("pr del primo grandchildren del j-esimo child")     
            #     #println(typeof(pr(n.children[j].children[1]))) 
            #     #println(pr(n.children[j].children[1]))
            #     #println(pr(n.children[j].children[1])[:,3]) cosi non gli piace ! è un array unidimensionale (ie un vettore) di tuples (triplette qui)

            #     #println(getindex(pr(n.children[j].children[1])[1],3)) cosi ok
                


            #     left_index[getindex.(pr(n.children[j].children[1]),3) .== 1] .= 1
            #     right_index[getindex.(pr(n.children[j].children[2]),3) .== 1] .= 1  #### TODO assertion for somma minore di uno
                
            #     ######## OK fino a qua apposto. ora posso creare i coeff per ogni nodo or che non sia un top:

            #     ### comincio a mettere il \pi   QUESTI FUNZIONAVA CON C MATRICE
            #     c[:,j][left_index .== 1.0] .= getindex.(pr(n.children[j].children[1])[left_index .== 1.0],1) ### qua mi calcola i pi_lower
            #     c[:,j][right_index .== 1.0] .= getindex.(pr(n.children[j].children[2])[right_index .== 1.0],1) 
                    
            #      ### ora aggiungo (in realtà moltiplico) il \sigma
            #     # c[:,j][c[:,j].< zeros(Float64,length(npr)) e left_index .== 1.0] .*= exp.(log_marginal_upper(subcircuito che parte da n.children[j].children[2], sub-batch che concerne solo le variabili di sto sub-cricuito))
            #     # c[:,j][c[:,j].< zeros(Float64,length(npr)) e right_index .== 1.0] .*= exp.(log_marginal_upper(subcircuito che parte da n.children[j].children[1], sub-batch che concerne solo le variabili di sto sub-cricuito))
            #     # c[:,j][c[:,j].> zeros(Float64,length(npr)) e left_index .== 1.0] .*= exp.(log_marginal_lower(subcircuito che parte da n.children[j].children[2], sub-batch che concerne solo le variabili di sto sub-cricuito))
            #     # c[:,j][c[:,j].> zeros(Float64,length(npr)) e right_index .== 1.0] .*= exp.(log_marginal_lower(subcircuito che parte da n.children[j].children[1], sub-batch che concerne solo le variabili di sto sub-cricuito))
                
            #    ### vettore = zeros( length(npr))
            #     ###println(vettore)
            #     ###println(c[:,j])
            #     ###c[:,j][(c[:,j].< zeros(Float64,length(npr))) .&& (left_index .== 1.0)] .= vettore[(c[:,j].< zeros(Float64,length(npr))) .&& (left_index .== 1.0)]
            # end
        




         ##### qui risolvo, in parallelo per ogni instance (query) LP e storo gli ottimi nei vari vettori 

         ## conditional 

            println("robetta da passare come coeff di tipo : ", typeof(getindex.(pr.(n.children),1)))
            println("tuple dei children : ", pr.(n.children))
            println("tuple della 1a inst  : ", getindex.(pr.(n.children),1))
            println("primi el delle tuple della 1a inst: ", getindex.(getindex.(pr.(n.children),1),1))

            c_co = Array{Float64}(undef, length(npr), length(n.children)) 
            c_marg_lo = Array{Float64}(undef, length(npr), length(n.children)) 
            c_marg_up = Array{Float64}(undef, length(npr), length(n.children)) 


            for i=1:length(npr) 
                c_co[i,:] = getindex.(getindex.(pr.(n.children),i),1) # vettore dei coefficenti del i-esimo problema COND  (i-esima instance)
                c_marg_lo[i,:] = getindex.(getindex.(pr.(n.children),i),2) # vettore dei coefficenti del i-esimo problema MARG_LOWER (i-esima instance)
                c_marg_up[i,:] = getindex.(getindex.(pr.(n.children),i),3) # vettore dei coefficenti del i-esimo problema MARG_UPPER  (i-esima instance)
            end
            
            println("coeff cond  : ", c_co)
            println("coeff marg_lo  : ", c_marg_lo)
            println("coeff marg_up  : ", c_marg_up)


            for i=1:length(npr) 
                val_min[i] = minimi(c_co[i,:], exp.(prob_origin(n).log_thetas), exp.(prob_origin(n).log_thetas) )
                marginal_lower[i] = minimi(c_marg_lo[i,:], exp.(prob_origin(n).log_thetas), exp.(prob_origin(n).log_thetas) )
                marginal_upper[i] = minimi(c_marg_up[i,:], exp.(prob_origin(n).log_thetas), exp.(prob_origin(n).log_thetas) )   
            end

        end

        
        
        pr(n) .= [x for x in zip(val_min,marginal_lower,marginal_upper,type)]
        return nothing
    end
    
    for n in circuit
        conditional_lower_pass_up_node(n,data,mu)
       #println( "tupla : ", pr(n)) #non va
    end
    return nothing
end






