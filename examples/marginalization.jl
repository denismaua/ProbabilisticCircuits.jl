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
# loads a .vtree file
# vtree_lines = parse_vtree_file("examples/4vars.vtree");
# vtree = compile_vtree_format_lines(vtree_lines);
# println(vtree)

# Learns CSDD from data based on learn_probabilistic_circuit
data = train(dataset(twenty_datasets("nltcs"); do_shuffle=false, batch_size=-1));
# learn_credal_circuit(data, s_idm) using CSDD/CredalCircuits.jl:estimate_credal_parameters(CredalΔ, data, s_idm)
csdd = learn_credal_circuit(WXData(data), 40.0); #using clt

# println(csdd)


for node in csdd
    if node isa Credal⋁  # typeof(node) == Credal⋁{UnstLogicalΔNode}
        println(node)
        println("lower: ",exp.(node.log_thetas))
        println("upper: ",exp.(node.log_thetas_u))
    end
end

# psdd = learn_probabilistic_circuit(WXData(data)); #using clt
# Loads a PSDD file 
pc = zoo_psdd("little_4var.psdd")
# Set some query. 1 assignts True, 0 assigns False, -1 missing
data = XData(Int8.([1 1 -1 -1]))
prob = exp.(log_proba(pc, data))
println("Prob: $(prob)")
