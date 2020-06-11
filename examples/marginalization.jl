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
# Loads a PSDD file 
pc = zoo_psdd("little_4var.psdd")
# Set some query. 1 assignts True, 0 assigns False, -1 missing
data = XData(Int8.([1 1 -1 -1]))
prob = exp.(log_proba(pc, data))
println("Prob: $(prob)")
