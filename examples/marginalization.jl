# this assumes we have already installed LogicCircuits
using LogicCircuits
# Loads our local package
# We need to "activate" our local version first
import Pkg
# This assumes you are running the file from inside the Project's folder
Pkg.activate(".")
# If this is not the case, you need to give the full path address, e.g.
#Pkg.activate("/Users/denis/ProbabilisticCircuits.jl/")
# Alternatively, we can import our packages from github (but then local changes are not included)
#Pkg.add("https://github.com/denismaua/ProbabilisticCircuits.jl")
using ProbabilisticCircuits

# psdd = learn_probabilistic_circuit(WXData(data)); #using clt
# Loads a PSDD file 
pc = zoo_psdd("little_4var.psdd")
# Set some query. 1 assignts True, 0 assigns False, -1 missing
data = XData(Int8.([1 1 -1 -1]))
prob = exp.(log_proba(pc, data))
# println("Prob: $(prob)")


# loads a .vtree file
# vtree_lines = parse_vtree_file("examples/4vars.vtree");
# vtree = compile_vtree_format_lines(vtree_lines);
# println(vtree)

#loads formula and vtree from  the repo https://github.com/UCLA-StarAI/Circuit-Model-Zoo
# 
# cnf = zoo_cnf("easy/C17_mince.cnf")
# vtree = zoo_vtree("easy/C17_mince.min.vtree");
# mgr = SddMgr(TrimSddMgr, vtree)
# cnfΔ = node2dag(compile_cnf(mgr, cnf))


#loads a .sdd  file
# sdd = load_logical_circuit("examples/random.sdd")

# Learns CSDD from data based on learn_probabilistic_circuit
data = train(dataset(twenty_datasets("nltcs"); do_shuffle=false, batch_size=-1));
# learn_credal_circuit(data, s_idm) using CSDD/CredalCircuits.jl:estimate_credal_parameters(CredalΔ, data, s_idm)
csdd = learn_credal_circuit(WXData(data), 40.0); #using clt

# println(csdd)

# for node in csdd
#     if node isa Credal⋁  # typeof(node) == Credal⋁{UnstLogicalΔNode}
#         println(node)
#         println("lower: ",exp.(node.log_thetas))
#         println("upper: ",exp.(node.log_thetas_u))
#     end
# end

# Testing marginal upper and lower flows

obs = XData(Int8.([1 -1 -1 0 0 -1 -1 1 1 -1 -1 1 -1 -1 -1 -1 ; -1 -1 -1 0 0 -1 -1 1 1 -1 -1 1 -1 -1 -1 -1]))
lower_marg = exp.(log_marginal_lower(csdd, obs))
upper_marg = exp.(log_marginal_upper(csdd, obs))
println("Lower marginal: $(lower_marg)")
println("Upper marginal: $(upper_marg)")

# Testing complete evidence likelihood upper and lower flows

complete_obs = XData(Bool.([1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 1; 1 1 1 0 0 1 1 1 1 1 0 0 0 0 0 0]))
lower_prob = exp.(log_prob_lower(csdd, complete_obs))
upper_prob = exp.(log_prob_upper(csdd, complete_obs))
println("Lower prob: $(lower_prob)")
println("Upper prob: $(upper_prob)")

