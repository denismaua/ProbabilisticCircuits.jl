# PROBABILISTIC CIRCUITS LIBRARY ROOT

module ProbabilisticCircuits

# USE EXTERNAL MODULES

using Reexport

include("Utils/Utils.jl")

@reexport using .Utils

# INCLUDE CHILD MODULES
include("Probabilistic/Probabilistic.jl")
include("Logistic/Logistic.jl")
include("CSDD/CSDD.jl")
include("IO/IO.jl")
include("StructureLearner/StructureLearner.jl")
include("Reasoning/Reasoning.jl")



# USE CHILD MODULES (in order to re-export some functions)
@reexport using .Probabilistic
@reexport using .Logistic
@reexport using .CSDD
@reexport using .IO
@reexport using .StructureLearner
@reexport using .Reasoning


end
