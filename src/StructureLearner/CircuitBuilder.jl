using LightGraphs: topological_sort_by_dfs, outneighbors
using MetaGraphs: get_prop


"convert literal+/- to probability value 0/1"
@inline lit2value(l::Lit)::Int = (l > 0 ? 1 : 0)

"""
Learning from data a circuit with several structure learning algorithms
"""
function learn_credal_circuit(data::Union{XData, WXData},s_idm::Float64; 
        pseudocount = 1.0, algo = "chow-liu", algo_kwargs=(α=1.0, clt_root="graph_center"))::CredalΔ
    if algo == "chow-liu"
        clt = learn_chow_liu_tree(data; algo_kwargs...)
        pc = compile_cred_circuit_from_clt(clt)
        estimate_credal_parameters(pc, convert(XBatches,data),s_idm; pseudocount = pseudocount)
        pc 
    else
        error("Cannot learn a probabilistic circuit with algorithm $algo")
    end
end

"""
Learning from data a circuit with several structure learning algorithms
"""
function learn_probabilistic_circuit(data::Union{XData, WXData}; 
        pseudocount = 1.0, algo = "chow-liu", algo_kwargs=(α=1.0, clt_root="graph_center"))::ProbΔ
    if algo == "chow-liu"
        clt = learn_chow_liu_tree(data; algo_kwargs...)
        pc = compile_prob_circuit_from_clt(clt)
        estimate_parameters(pc, convert(XBatches,data); pseudocount = pseudocount)
        pc 
    else
        error("Cannot learn a probabilistic circuit with algorithm $algo")
    end
end

"Build decomposable probability circuits from Chow-Liu tree"
function compile_prob_circuit_from_clt(clt::CLT)::ProbΔ
    topo_order = Var.(reverse(topological_sort_by_dfs(clt::CLT))) #order to parse the node
    lin = Vector{ProbΔNode}()
    node_cache = Dict{Lit, LogicalΔNode}()
    prob_cache = ProbCache()
    parent = parent_vector(clt)

    prob_children(n)::Vector{<:ProbΔNode{<:node_type(n)}} =  
        copy_with_eltype(map(c -> prob_cache[c], n.children), ProbΔNode{<:node_type(n)})

    "default order of circuit node, from left to right: +/1 -/0"

    "compile leaf node into circuits"
    function compile_leaf(ln::Var)
        pos = LiteralNode( var2lit(ln))
        neg = LiteralNode(-var2lit(ln))
        node_cache[var2lit(ln)] = pos
        node_cache[-var2lit(ln)] = neg
        pos2 = ProbLiteral(pos)
        neg2 = ProbLiteral(neg)
        push!(lin, pos2)
        push!(lin, neg2)
        prob_cache[pos] = pos2
        prob_cache[neg] = neg2
    end

    "compile inner disjunction node"
    function compile_⋁inner(ln::Lit, children::Vector{Var})::Vector{⋁Node}
        logical_nodes = Vector{⋁Node}()
        v = lit2value(ln)

        for c in children
            #build logical ciruits
            temp = ⋁Node([node_cache[lit] for lit in [var2lit(c), - var2lit(c)]])
            push!(logical_nodes, temp)
            n = Prob⋁(temp, prob_children(temp))
            prob_cache[temp] = n
            n.log_thetas = zeros(Float64, 2)
            cpt = get_prop(clt, c, :cpt)
            weights = [cpt[(1, v)], cpt[(0, v)]]
            n.log_thetas = log.(weights)
            push!(lin, n)
        end

        return logical_nodes
    end

    "compile inner conjunction node into circuits, left node is indicator, rest nodes are disjunction children nodes"
    function compile_⋀inner(indicator::Lit, children::Vector{⋁Node})
        leaf = node_cache[indicator]
        temp = ⋀Node(vcat([leaf], children))
        node_cache[indicator] = temp
        n = Prob⋀(temp, prob_children(temp))
        prob_cache[temp] = n
        push!(lin, n)
    end

    "compile inner node, 1 inner variable to 2 leaf nodes, 2 * num_children disjunction nodes and 2 conjunction nodes"
    function compile_inner(ln::Var, children::Vector{Var})
        compile_leaf(ln)
        pos⋁ = compile_⋁inner(var2lit(ln), children)
        neg⋁ = compile_⋁inner(-var2lit(ln), children)
        compile_⋀inner(var2lit(ln), pos⋁)
        compile_⋀inner(-var2lit(ln), neg⋁)
    end

    "compile root, add another disjunction node"
    function compile_root(root::Var)
        temp = ⋁Node([node_cache[s] for s in [var2lit(root), -var2lit(root)]])
        n = Prob⋁(temp, prob_children(temp))
        prob_cache[temp] = n
        n.log_thetas = zeros(Float64, 2)
        cpt = get_prop(clt, root, :cpt)
        weights = [cpt[1], cpt[0]]
        n.log_thetas = log.(weights)
        push!(lin, n)
        return n
    end

    function compile_independent_roots(roots::Vector{ProbΔNode})
        temp = ⋀Node([c.origin for c in roots])
        n = Prob⋀(temp, prob_children(temp))
        prob_cache[temp] = n
        push!(lin, n)
        temp = ⋁Node([temp])
        n = Prob⋁{LogicalΔNode}(temp, prob_children(temp))
        prob_cache[temp] = n
        n.log_thetas = [0.0]
        push!(lin, n)
    end

    roots = Vector{ProbΔNode}()
    for id in topo_order
        children = Var.(outneighbors(clt, id))
        if isequal(children, [])
            compile_leaf(id)
        else
            compile_inner(id, children)
        end
        if 0 == parent[id]
            push!(roots, compile_root(id))
        end
    end

    if length(roots) > 1
        compile_independent_roots(roots)
    end

    return lin
end


"Build decomposable credal circuit from Chow-Liu tree"
function compile_cred_circuit_from_clt(clt::CLT)::CredalΔ
    topo_order = Var.(reverse(topological_sort_by_dfs(clt::CLT))) #order to parse the node
    lin = Vector{CredalΔNode}()
    node_cache = Dict{Lit, LogicalΔNode}()
    cred_cache = CredalCache()
    parent = parent_vector(clt)

    cred_children(n)::Vector{<:CredalΔNode{<:node_type(n)}} =  
        copy_with_eltype(map(c -> cred_cache[c], n.children), CredalΔNode{<:node_type(n)})

    "default order of circuit node, from left to right: +/1 -/0"

    "compile leaf node into circuits"
    function compile_leaf(ln::Var)
        pos = LiteralNode( var2lit(ln))
        neg = LiteralNode(-var2lit(ln))
        node_cache[var2lit(ln)] = pos
        node_cache[-var2lit(ln)] = neg
        pos2 = CredalLiteral(pos)
        neg2 = CredalLiteral(neg)
        push!(lin, pos2)
        push!(lin, neg2)
        cred_cache[pos] = pos2
        cred_cache[neg] = neg2
    end

    "compile inner disjunction node"
    function compile_⋁inner(ln::Lit, children::Vector{Var})::Vector{⋁Node}
        logical_nodes = Vector{⋁Node}()
        v = lit2value(ln)

        for c in children
            #build logical ciruits
            temp = ⋁Node([node_cache[lit] for lit in [var2lit(c), - var2lit(c)]])
            push!(logical_nodes, temp)
            n = Credal⋁(temp, cred_children(temp))
            cred_cache[temp] = n
            n.log_thetas = zeros(Float64, 2)
            cpt = get_prop(clt, c, :cpt)
            weights = [cpt[(1, v)], cpt[(0, v)]]
            n.log_thetas = log.(weights)
            push!(lin, n)
        end

        return logical_nodes
    end

    "compile inner conjunction node into circuits, left node is indicator, rest nodes are disjunction children nodes"
    function compile_⋀inner(indicator::Lit, children::Vector{⋁Node})
        leaf = node_cache[indicator]
        temp = ⋀Node(vcat([leaf], children))
        node_cache[indicator] = temp
        n = Credal⋀(temp, cred_children(temp))
        cred_cache[temp] = n
        push!(lin, n)
    end

    "compile inner node, 1 inner variable to 2 leaf nodes, 2 * num_children disjunction nodes and 2 conjunction nodes"
    function compile_inner(ln::Var, children::Vector{Var})
        compile_leaf(ln)
        pos⋁ = compile_⋁inner(var2lit(ln), children)
        neg⋁ = compile_⋁inner(-var2lit(ln), children)
        compile_⋀inner(var2lit(ln), pos⋁)
        compile_⋀inner(-var2lit(ln), neg⋁)
    end

    "compile root, add another disjunction node"
    function compile_root(root::Var)
        temp = ⋁Node([node_cache[s] for s in [var2lit(root), -var2lit(root)]])
        n = Credal⋁(temp, cred_children(temp))
        cred_cache[temp] = n
        n.log_thetas = zeros(Float64, 2)
        cpt = get_prop(clt, root, :cpt)
        weights = [cpt[1], cpt[0]]
        n.log_thetas = log.(weights)
        push!(lin, n)
        return n
    end

    function compile_independent_roots(roots::Vector{CredalΔNode})
        temp = ⋀Node([c.origin for c in roots])
        n = Credal⋀(temp, cred_children(temp))
        cred_cache[temp] = n
        push!(lin, n)
        temp = ⋁Node([temp])
        n = Credal⋁{LogicalΔNode}(temp, cred_children(temp))
        cred_cache[temp] = n
        n.log_thetas = [0.0]
        push!(lin, n)
    end

    roots = Vector{CredalΔNode}()
    for id in topo_order
        children = Var.(outneighbors(clt, id))
        if isequal(children, [])
            compile_leaf(id)
        else
            compile_inner(id, children)
        end
        if 0 == parent[id]
            push!(roots, compile_root(id))
        end
    end

    if length(roots) > 1
        compile_independent_roots(roots)
    end

    return lin
end
