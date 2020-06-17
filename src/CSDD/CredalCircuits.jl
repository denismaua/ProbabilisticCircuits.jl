#####################
# Credal circuits
#####################
abstract type CredalΔNode{O} <: DecoratorΔNode{O} end
abstract type CredalLeafNode{O} <: CredalΔNode{O} end
abstract type CredalInnerNode{O} <: CredalΔNode{O} end

mutable struct CredalLiteral{O} <: CredalLeafNode{O}
    origin::O
    data
    bit::Bool
    CredalLiteral(n) = new{node_type(n)}(n, nothing, false)
end

mutable struct Credal⋀{O} <: CredalInnerNode{O}
    origin::O
    children::Vector{<:CredalΔNode{<:O}}
    data
    bit::Bool
    Credal⋀(n, children) = begin
        new{node_type(n)}(n, convert(Vector{CredalΔNode{node_type(n)}},children), nothing, false)
    end
end

mutable struct Credal⋁{O} <: CredalInnerNode{O}
    origin::O
    children::Vector{<:CredalΔNode{<:O}}
    log_thetas::Vector{Float64}
    log_thetas_u::Vector{Float64}
    data
    bit::Bool
    Credal⋁(n, children) = new{node_type(n)}(n, convert(Vector{CredalΔNode{node_type(n)}},children), some_vector(Float64, length(children)), some_vector(Float64, length(children)), nothing, false)
end

const CredalΔ{O} = AbstractVector{<:CredalΔNode{<:O}}

Base.eltype(::Type{CredalΔ{O}}) where {O} = CredalΔNode{<:O}

#####################
# traits
#####################

import LogicCircuits.GateType # make available for extension
import LogicCircuits.node_type

@inline GateType(::Type{<:CredalLiteral}) = LiteralGate()
@inline GateType(::Type{<:Credal⋀}) = ⋀Gate()
@inline GateType(::Type{<:Credal⋁}) = ⋁Gate()

@inline node_type(::CredalΔNode) = CredalΔNode

#####################
# constructors and conversions
#####################

const CredalCache = Dict{ΔNode, CredalΔNode}

function CredalΔ2(circuit::Δ)::CredalΔ
    node2dag(CredalΔ2(circuit[end]))
end

function CredalΔ2(circuit::ΔNode)::CredalΔNode
    f_con(n) = error("Cannot construct a probabilistic circuit from constant leafs: first smooth and remove unsatisfiable branches.")
    f_lit(n) = CredalLiteral(n)
    f_a(n, cn) = Credal⋀(n, cn)
    f_o(n, cn) = Credal⋁(n, cn)
    foldup_aggregate(circuit, f_con, f_lit, f_a, f_o, CredalΔNode{node_type(circuit)})
end

function CredalΔ(circuit::Δ, cache::CredalCache = CredalCache())

    sizehint!(cache, length(circuit)*4÷3)

    pc_node(::LiteralGate, n::ΔNode) = CredalLiteral(n)
    pc_node(::ConstantGate, n::ΔNode) = error("Cannot construct a probabilistic circuit from constant leafs: first smooth and remove unsatisfiable branches.")

    pc_node(::⋀Gate, n::ΔNode) = begin
        children = map(c -> cache[c], n.children)
        Credal⋀(n, children)
    end

    pc_node(::⋁Gate, n::ΔNode) = begin
        children = map(c -> cache[c], n.children)
        Credal⋁(n, children)
    end

    map(circuit) do node
        pcn = pc_node(GateType(node), node)
        cache[node] = pcn
        pcn
    end
end

#####################
# methods
#####################

import LogicCircuits: literal, children # make available for extension

@inline literal(n::CredalLiteral)::Lit  = literal(n.origin)
@inline children(n::CredalInnerNode) = n.children

num_parameters(n::Credal⋁) = num_children(n)
num_parameters(c::CredalΔ) = sum(n -> num_parameters(n), ⋁_nodes(c))

"Return the first origin that is a probabilistic circuit node"
prob_origin(n::DecoratorΔNode)::CredalΔNode = origin(n, CredalΔNode)

"Return the first origin that is a probabilistic circuit"
prob_origin(c::DecoratorΔ)::CredalΔ = origin(c, CredalΔNode)

function estimate_credal_parameters2(pc::CredalΔ, data::XData{Bool}, s_idm::Float64; pseudocount::Float64)
    Logical.pass_up_down2(pc, data)
    w = (data isa PlainXData) ? nothing : weights(data)
    estimate_credal_parameters_cached2(pc, w, s_idm; pseudocount=pseudocount)
end

function estimate_credal_parameters_cached2(pc::CredalΔ, w, s_idm::Float64; pseudocount::Float64)
    flow(n) = Float64(sum(sum(n.data)))
    children_flows(n) = sum.(map(c -> c.data[1] .& n.data[1], children(n)))

    if issomething(w)
        flow_w(n) = sum(Float64.(n.data[1]) .* w)
        children_flows_w(n) = sum.(map(c -> Float64.(c.data[1] .& n.data[1]) .* w, children(n)))
        flow = flow_w
        children_flows = children_flows_w
    end

    estimate_credal_parameters_node2(n::CredalΔNode, s_idm::Float64) = ()
    function estimate_credal_parameters_node2(n::Credal⋁, s_idm::Float64)
        if num_children(n) == 1
            n.log_thetas_u .= 0.0
            n.log_thetas .= 0.0
        else
            smoothed_flow = flow(n) + pseudocount
            uniform_pseudocount = pseudocount / num_children(n)
            n.log_thetas .= log.((children_flows(n) .+ uniform_pseudocount)  ./ (smoothed_flow .+ s_idm))
            n.log_thetas_u .= log.(((children_flows(n) .+ uniform_pseudocount) .+ s_idm) ./ (smoothed_flow .+ s_idm))
            # @assert isapprox(sum(exp.(n.log_thetas)), 1.0, atol=1e-6) "Parameters do not sum to one locally"
            # normalize away any leftover error
            # n.log_thetas .- logsumexp(n.log_thetas)
        end
    end

    foreach(estimate_credal_parameters_node2, pc, s_idm)
end

function log_likelihood_per_instance2(pc::CredalΔ, data::XData{Bool})
    Logical.pass_up_down2(pc, data)
    log_likelihood_per_instance_cached(pc, data)
end

function log_likelihood_per_instance_cached(pc::CredalΔ, data::XData{Bool})
    log_likelihoods = zeros(num_examples(data))
    indices = some_vector(Bool, num_examples(data))::BitVector
    for n in pc
         if n isa Credal⋁ && num_children(n) != 1 # other nodes have no effect on likelihood
            foreach(n.children, n.log_thetas) do c, log_theta
                indices = n.data[1] .& c.data[1]
                view(log_likelihoods, indices::BitVector) .+=  log_theta # see MixedProductKernelBenchmark.jl
            end
         end
    end
    log_likelihoods
end

import LogicCircuits: conjoin_like, disjoin_like, literal_like, copy_node, normalize, replace_node # make available for extension

"Conjoin nodes in the same way as the example"
@inline function conjoin_like(example::CredalΔNode, arguments::Vector)
    if isempty(arguments)
        # @assert false "Credalabilistic circuit does not have anonymous true node"
        nothing
    elseif example isa Credal⋀ && children(example) == arguments
        example
    else
        n = conjoin_like(origin(example), origin.(arguments))
        Credal⋀(n, arguments)
    end
end

"Disjoin nodes in the same way as the example"
@inline function disjoin_like(example::CredalΔNode, arguments::Vector)
    if isempty(arguments)
        # @assert false "Probabilistic circuit does not have false node"
        nothing
    elseif example isa Credal⋁ && children(example) == arguments
        example
    else
        n = disjoin_like(origin(example), origin.(arguments))
        # normalize parameters
        thetas = zeros(Float64, length(arguments))
        flag = falses(length(arguments))
        for (i, c) in enumerate(arguments)
            ind = findfirst(x -> x == c, children(example))
            if issomething(ind)
                thetas[i] = exp(example.log_thetas[ind])
                flag[i] = true
            end
        end
        if all(flag)
            thetas = thetas / sum(thetas)
        end
        p = Credal⋁(n, arguments)
        p.log_thetas .= log.(thetas)
        p
    end
end

"Construct a new literal node like the given node's type"
@inline literal_like(::CredalΔNode, lit::Lit) = CredalLiteral(lit)

@inline copy_node(n::Credal⋁, cns) = begin
    orig = copy_node(origin(n), origin.(cns))
    p = Credal⋁(orig, cns)
    p.log_thetas .= copy(n.log_thetas)
    p
end

@inline copy_node(n::Credal⋀, cns) = begin
    orig = copy_node(origin(n), origin.(cns))
    Credal⋀(orig, cns)
end

import LogicCircuits.normalize

@inline normalize(n::Credal⋁, old_n::Credal⋁, kept::Union{Vector{Bool}, BitArray}) = begin
     thetas = exp.(old_n.log_thetas[kept])
     n.log_thetas .= log.(thetas / sum(thetas))
end

function estimate_credal_parameters(pc::CredalΔ, data::XBatches{Bool}, s_idm::Float64; pseudocount::Float64)
    estimate_credal_parameters(AggregateFlowΔ(pc, aggr_weight_type(data)), data, s_idm; pseudocount=pseudocount)
end

function estimate_credal_parameters(afc::AggregateFlowΔ, data::XBatches{Bool}, s_idm::Float64; pseudocount::Float64)
    @assert feature_type(data) == Bool "Can only learn probabilistic circuits on Bool data"
    @assert (afc[end].origin isa CredalΔNode) "AggregateFlowΔ must originate in a CredalΔ"
    collect_aggr_flows(afc, data)
    estimate_credal_parameters_cached(afc, s_idm; pseudocount=pseudocount)
    afc
end

function estimate_credal_parameters(fc::FlowΔ, data::XBatches{Bool}, s_idm::Float64; pseudocount::Float64)
    @assert feature_type(data) == Bool "Can only learn probabilistic circuits on Bool data"
    @assert (prob_origin(afc[end]) isa CredalΔNode) "FlowΔ must originate in a CredalΔ"
    collect_aggr_flows(fc, data)
    estimate_credal_parameters_cached(origin(fc), s_idm; pseudocount=pseudocount)
end

 # turns aggregate statistics into theta parameters
function estimate_credal_parameters_cached(afc::AggregateFlowΔ, s_idm::Float64; pseudocount::Float64)
    foreach(n -> estimate_credal_parameters_node(n, s_idm; pseudocount=pseudocount), afc)
end

estimate_credal_parameters_node(::AggregateFlowΔNode, s_idm::Float64; pseudocount::Float64) = () # do nothing
function estimate_credal_parameters_node(n::AggregateFlow⋁, s_idm::Float64; pseudocount)
    origin = n.origin::Credal⋁
    if num_children(n) == 1
        origin.log_thetas_u .= 0.0
        origin.log_thetas .= 0.0
    else
        smoothed_aggr_flow = (n.aggr_flow + pseudocount)
        uniform_pseudocount = pseudocount / num_children(n)
        origin.log_thetas .= log.( (n.aggr_flow_children .+ uniform_pseudocount) ./ (smoothed_aggr_flow .+ s_idm ))
        origin.log_thetas_u .= log.( ((n.aggr_flow_children .+ uniform_pseudocount) .+ s_idm) ./ (smoothed_aggr_flow .+ s_idm ))
        # @assert isapprox(sum(exp.(origin.log_thetas)), 1.0, atol=1e-6) "Parameters do not sum to one locally: $(exp.(origin.log_thetas)), estimated from $(n.aggr_flow) and $(n.aggr_flow_children). Did you actually compute the aggregate flows?"
        #normalize away any leftover error ()
        # origin.log_thetas .- logsumexp(origin.log_thetas)
    end
end

# compute log likelihood
function compute_log_likelihood(pc::CredalΔ, data::XBatches{Bool})
    compute_log_likelihood(AggregateFlowΔ(pc, aggr_weight_type(data)))
end

# compute log likelihood, reusing AggregateFlowΔ but ignoring its current aggregate values
function compute_log_likelihood(afc::AggregateFlowΔ, data::XBatches{Bool})
    @assert feature_type(data) == Bool "Can only test probabilistic circuits on Bool data"
    collect_aggr_flows(afc, data)
    ll = log_likelihood(afc)
    (afc, ll)
end

# return likelihoods given current aggregate flows.
function log_likelihood(afc::AggregateFlowΔ)
    sum(n -> log_likelihood(n), afc)
end

log_likelihood(::AggregateFlowΔNode) = 0.0
log_likelihood(n::AggregateFlow⋁) = sum(n.origin.log_thetas .* n.aggr_flow_children)

"""
Calculates log likelihood for a batch of fully observed samples.
(Also retures the generated FlowΔ)
"""
function log_likelihood_per_instance(pc::CredalΔ, batch::PlainXData{Bool})
    fc = FlowΔ(pc, num_examples(batch), Bool)
    (fc, log_likelihood_per_instance(fc, batch))
end

function log_proba_upper(pc::CredalΔ, batch::PlainXData{Bool})
    log_likelihood_per_instance(pc, batch)[2]
end

function log_proba_upper(pc::CredalΔ, batch::PlainXData{Int8})
    marginal_log_likelihood_per_instance(pc, batch)[2]
end

"""
Calculate log likelihood per instance for batches of samples.
"""
function log_likelihood_per_instance(pc::CredalΔ, batches::XBatches{Bool})::Vector{Float64}
    mapreduce(b -> log_likelihood_per_instance(pc, b)[2], vcat, batches)
end

"""
Calculate log likelihood for a batch of fully observed samples.
(This is for when you already have a FlowΔ)
"""
function log_likelihood_per_instance(fc::FlowΔ, batch::PlainXData{Bool})
    @assert (prob_origin(fc[end]) isa CredalΔNode) "FlowΔ must originate in a CredalΔ"
    pass_up_down(fc, batch)
    log_likelihoods = zeros(num_examples(batch))
    indices = some_vector(Bool, flow_length(fc))::BitVector
    for n in fc
         if n isa DownFlow⋁ && num_children(n) != 1 # other nodes have no effect on likelihood
            origin = prob_origin(n)::Credal⋁
            foreach(n.children, origin.log_thetas) do c, log_theta
                #  be careful here to allow for the Boolean multiplication to be done using & before switching to float arithmetic, or risk losing a lot of runtime!
                # log_likelihoods .+= prod_fast(downflow(n), pr_factors(c)) .* log_theta
                assign_prod(indices, downflow(n), pr_factors(c))
                view(log_likelihoods, indices::BitVector) .+=  log_theta # see MixedProductKernelBenchmark.jl
                # TODO put the lines above in Utils in order to ensure we have specialized types
            end
         end
    end
    log_likelihoods
end

"""
Calculate log likelihood for a batch of samples with partial evidence P(e).
(Also returns the generated FlowΔ)

To indicate a variable is not observed, pass -1 for that variable.
"""
function marginal_log_likelihood_per_instance(pc::CredalΔ, batch::PlainXData{Int8})
    opts = (flow_opts★..., el_type=Float64, compact⋀=false, compact⋁=false)
    fc = UpFlowΔ(pc, num_examples(batch), Float64, opts)
    (fc, marginal_log_likelihood_per_instance(fc, batch))
end

"""
Calculate log likelihood for a batch of samples with partial evidence P(e).
(If you already have a FlowΔ)

To indicate a variable is not observed, pass -1 for that variable.
"""
function marginal_log_likelihood_per_instance(fc::UpFlowΔ, batch::PlainXData{Int8})
    @assert (prob_origin(fc[end]) isa CredalΔNode) "FlowΔ must originate in a CredalΔ"
    marginal_pass_up(fc, batch)
    pr(fc[end])
end

function check_parameter_integrity(circuit::CredalΔ)
    for node in filter(n -> GateType(n) isa Credal⋁, circuit)
        @assert all(θ -> !isnan(θ), node.log_thetas) "There is a NaN in one of the log_thetas"
    end
    true
end

##################
# Sampling from a psdd
##################

"""
Sample from a PSDD without any evidence
"""
function sample(circuit::CredalΔ)::AbstractVector{Bool}
    inst = Dict{Var,Int64}()
    simulate(circuit[end], inst)
    len = length(keys(inst))
    ans = Vector{Bool}()
    for i = 1:len
        push!(ans, inst[i])
    end
    ans
end

# Uniformly sample based on the probability of the items
# and return the selected index
function sample(probs::AbstractVector{<:Number})::Int32
    z = sum(probs)
    q = rand() * z
    cur = 0.0
    for i = 1:length(probs)
        cur += probs[i]
        if q <= cur
            return i
        end
    end
    return length(probs)
end

function simulate(node::CredalLiteral, inst::Dict{Var,Int64})
    if positive(node)
        inst[variable(node.origin)] = 1
    else
        inst[variable(node.origin)] = 0
    end
end

function simulate(node::Credal⋁, inst::Dict{Var,Int64})
    idx = sample(exp.(node.log_thetas))
    simulate(node.children[idx], inst)
end
function simulate(node::Credal⋀, inst::Dict{Var,Int64})
    for child in node.children
        simulate(child, inst)
    end
end

"""
Sampling with Evidence from a psdd.
Internally would call marginal pass up on a newly generated flow circuit.
"""
function sample(circuit::CredalΔ, evidence::PlainXData{Int8})::AbstractVector{Bool}
    opts= (compact⋀=false, compact⋁=false)
    flow_circuit = UpFlowΔ(circuit, 1, Float64, opts)
    marginal_pass_up(flow_circuit, evidence)
    sample(flow_circuit)
end

"""
Sampling with Evidence from a psdd.
Assuming already marginal pass up has been done on the flow circuit.
"""
function sample(circuit::UpFlowΔ)::AbstractVector{Bool}
    inst = Dict{Var,Int64}()
    simulate2(circuit[end], inst)
    len = length(keys(inst))
    ans = Vector{Bool}()
    for i = 1:len
        push!(ans, inst[i])
    end
    ans
end

function simulate2(node::UpFlowLiteral, inst::Dict{Var,Int64})
    if positive(node)
        #TODO I don't think we need these 'grand_origin' parts below
        inst[variable(grand_origin(node))] = 1
    else
        inst[variable(grand_origin(node))] = 0
    end
end

function simulate2(node::UpFlow⋁, inst::Dict{Var,Int64})
    prs = [ pr(ch)[1] for ch in children(node) ]
    idx = sample(exp.(node.origin.log_thetas .+ prs))
    simulate2(children(node)[idx], inst)
end

function simulate2(node::UpFlow⋀, inst::Dict{Var,Int64})
    for child in children(node)
        simulate2(child, inst)
    end
end



##################
# Most Probable Explanation MPE of a psdd
#   aka MAP
##################

@inline function MAP(circuit::CredalΔ, evidence::PlainXData{Int8})::Matrix{Bool}
    MPE(circuit, evidence)
end

function MPE(circuit::CredalΔ, evidence::PlainXData{Int8})::Matrix{Bool}
    # Computing Marginal Likelihood for each node
    fc, lls = marginal_log_likelihood_per_instance(circuit, evidence)

    ans = Matrix{Bool}(zeros(size(evidence.x)))
    active_samples = Array{Bool}(ones( num_examples(evidence) ))

    mpe_simulate(fc[end], active_samples, ans)
    ans
end

"""
active_samples: bool vector indicating which samples are active for this node during mpe
result: Matrix (num_samples, num_variables) indicating the final result of mpe
"""
function mpe_simulate(node::UpFlowLiteral, active_samples::Vector{Bool}, result::Matrix{Bool})
    if positive(node)
        result[active_samples, variable(node)] .= 1
    else
        result[active_samples, variable(node)] .= 0
    end
end
function mpe_simulate(node::UpFlow⋁, active_samples::Vector{Bool}, result::Matrix{Bool})
    prs = zeros( length(node.children), size(active_samples)[1] )
    @simd  for i=1:length(node.children)
        prs[i,:] .= pr(node.children[i]) .+ (node.origin.log_thetas[i])
    end

    max_child_ids = [a[1] for a in argmax(prs, dims = 1) ]
    @simd for i=1:length(node.children)
        ids = Vector{Bool}( active_samples .* (max_child_ids .== i)[1,:] )  # Only active for this child if it was the max for that sample
        mpe_simulate(node.children[i], ids, result)
    end
end
function mpe_simulate(node::UpFlow⋀, active_samples::Vector{Bool}, result::Matrix{Bool})
    for child in node.children
        mpe_simulate(child, active_samples, result)
    end    
end
