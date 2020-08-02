using Test, Random, BSON, AdvancedHMC, Distributions, ForwardDiff

function simulate_and_compare(hmc_variant)

Random.seed!(1)

# Set parameter dimensionality and initial parameter value
dim = 10; θ₀ = rand(dim)

# Define the target distribution
ℓπ(θ) = logpdf(MvNormal(zeros(dim), ones(dim)), θ)

# Set the number of samples to draw and iterations for warmup
n_samples, n_adapts = 2_000, 1_000

# Define a Hamiltonian system
metric = DiagEuclideanMetric(dim)
hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)

# Define a leapfrog solver, with initial step size chosen heuristically
ϵ₀ = find_good_stepsize(hamiltonian, θ₀)
integrator = Leapfrog(ϵ₀)

kernel =
if hmc_variant == "hmc_mh"
    HMCKernel(FullRefreshment(), Trajectory(integrator, FixedNSteps(10)), MetropolisTS)
elseif hmc_variant == "hmc_multi"
    HMCKernel(FullRefreshment(), Trajectory(integrator, FixedNSteps(10)), MultinomialTS)
elseif hmc_variant == "nuts_slice"
    HMCKernel(FullRefreshment(), Trajectory(integrator, NoUTurn()), SliceTS)
elseif hmc_variant == "nuts_multi"
    HMCKernel(FullRefreshment(), Trajectory(integrator, NoUTurn()), MultinomialTS)
end

adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, ϵ₀))

# Run the sampler to draw samples from the specified Gaussian, where
#   - `samples` will store the samples
#   - `stats` will store diagnostic statistics for each sample
samples, stats = sample(hamiltonian, kernel, θ₀, n_samples, adaptor, n_adapts; progress=true)

old = BSON.load("$(@__DIR__)/regression/ef6de39/$hmc_variant.bson")

@test samples == old[:samples]
@test stats == old[:stats]

end # function

@testset "Regression" begin
    simulate_and_compare("hmc_mh")
    simulate_and_compare("hmc_multi")
    simulate_and_compare("nuts_slice")
    simulate_and_compare("nuts_multi")
end
