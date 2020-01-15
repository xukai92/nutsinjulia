module AdvancedHMC

const DEBUG = Bool(parse(Int, get(ENV, "DEBUG_AHMC", "0")))

using Statistics: mean, var, middle
using LinearAlgebra: Symmetric, UpperTriangular, mul!, ldiv!, dot, I, diag, cholesky
using StatsFuns: logaddexp
using LazyArrays: BroadcastArray
using Random: GLOBAL_RNG, AbstractRNG
using ProgressMeter
using Parameters: @unpack, reconstruct
using ArgCheck: @argcheck

import StatsBase: sample

# Support of passing a vector of RNGs
Base.rand(rng::AbstractVector{<:AbstractRNG}) = rand.(rng)
Base.randn(rng::AbstractVector{<:AbstractRNG}) = randn.(rng)
function Base.randn(
    rng::AbstractVector{<:AbstractRNG},
    T,
    sz::Int...)
    return cat(randn.(rng, T, first(sz))...; dims=2)
end

# Notations
# ℓπ: log density of the target distribution
# ∂ℓπ∂θ: gradient of the log density of the target distribution
# θ: position variables / model parameters
# r: momentum variables

include("adaptation/Adaptation.jl")
export UnitEuclideanMetric, DiagEuclideanMetric, DenseEuclideanMetric
export NesterovDualAveraging, Preconditioner, NaiveHMCAdaptor, StanHMCAdaptor
using .Adaptation

using .Adaptation: AbstractScalarOrVec
import .Adaptation: adapt!, NesterovDualAveraging

include("hamiltonian.jl")
export Hamiltonian
include("integrator.jl")
export Leapfrog, JitteredLeapfrog, TemperedLeapfrog
include("trajectory.jl")
export StaticTrajectory, HMCDA, NUTS, EndPointTS, SliceTS, MultinomialTS, ClassicNoUTurn, GeneralisedNoUTurn, find_good_eps

include("diagnosis.jl")
include("sampler.jl")
export sample

### AD utilities

@enum AutoDiff ForwardDiffAD=1 ZygoteAD=2
const ADAVAILABLE = Dict{AutoDiff, Function}()
AutoDiff() = first(keys(ADAVAILABLE))
Hamiltonian(metric::AbstractMetric, ℓπ, ad::AutoDiff) = ADAVAILABLE[ad](metric, ℓπ)

function Hamiltonian(metric::AbstractMetric, ℓπ)
    available = collect(keys(ADAVAILABLE))
    ad2package(x) = string(x)[1:end-2]
    if length(available) == 0
        support_list_str = join(map(ad2package, instances(AutoDiff)), " or ")
        error("MethodError: no method matching Hamiltonian(metric::AbstractMetric, ℓπ) because no backend is loaded. Please load an AD package ($support_list_str) first.")
    elseif length(available) > 1
        available_list_str = join(map(ad2package, available), " and ")
        constructor_list_str = join(map(instance -> "Hamiltonian(metric, ℓπ, AutoDiff($(Int(instance)))) to use $instance", available), "\n  ")
        error("MethodError: Hamiltonian(metric::AbstractMetric, ℓπ) is ambiguous because multiple AD pakcages are available ($available_list_str). Please use AD explictly. Candidates:\n  $constructor_list_str")
    else
        return Hamiltonian(metric, ℓπ, first(available))
    end
end

export AutoDiff

using Requires
include("init.jl")

end # module
