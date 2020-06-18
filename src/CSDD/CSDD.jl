module CSDD

using LogicCircuits
# using ..ProbabilisticCircuits
using ..Utils

export


#CredalSDDs
estimate_credal_parameters, estimate_credal_parameters_cached,
CredalΔNode, CredalΔ, CredalLeafNode, CredalInnerNode,
CredalLiteral, Credal⋀, Credal⋁, CredalCache,
# , variable, num_parameters, compute_log_likelihood,

log_proba_upper
# log_proba_lower
# ProbCircuits
# ProbΔNode, ProbΔ, ProbΔ, ProbLeafNode, ProbInnerNode,
# ProbLiteral, Prob⋀, Prob⋁, ProbCache, variable, num_parameters, compute_log_likelihood,
# log_proba,
# log_likelihood, estimate_parameters, log_likelihood_per_instance, marginal_log_likelihood_per_instance,
# initial_mixture_model, estimate_parameters_from_aggregates, compute_ensemble_log_likelihood,
# expectation_step, maximization_step, expectation_step_batch, train_mixture_with_structure, check_parameter_integrity,
# ll_per_instance_per_component, ll_per_instance_for_ensemble,estimate_parameters_cached,
# sample,
# MPE, MAP,prob_origin, copy_node, conjoin_like, disjoin_like, literal_like, normalize, replace_node,

# ProbFlowCircuits
# marginal_pass_up, marginal_pass_down, marginal_pass_up_down,

# Mixtures
# Mixture, AbstractFlatMixture, FlatMixture, FlatMixtureWithFlow,component_weights,FlatMixtureWithFlows,
# log_likelihood, log_likelihood_per_instance, log_likelihood_per_instance_component,
# init_mixture_with_flows, reset_mixture_aggregate_flows, aggregate_flows, estimate_parameters,
# AbstractMetaMixture, MetaMixture,AbstractFlatMixture,AbstractMixture, components, num_components,

# EM Learner
# train_mixture,

# Bagging
# bootstrap_samples_ids, learn_mixture_bagging, learn_mixture_bagging2,
# init_bagging_samples, train_bagging,

# # VtreeLearner
# MetisContext, metis_top_down, BlossomContext, blossom_bottom_up!,
# test_top_down, test_bottom_up!,learn_vtree_bottom_up,

# MutualInformation
# mutual_information, DisCache, conditional_entropy, sum_entropy_given_x,

# Clustering
# clustering,

# Queries
# pr_constraint, psdd_entropy, psdd_kl_divergence

include("CredalCircuits.jl")
include("CredalFlowCircuits.jl")
include("VtreeLearner.jl")
include("Queries.jl")

end
