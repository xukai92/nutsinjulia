#####################
#   MODEL
#####################

struct Joint{Tℓprior, Tℓll} <: Function
    ℓprior      :: Tℓprior
    ℓlikelihood :: Tℓll
end

function (joint::Joint)(θ)
    return joint.ℓprior(θ) .+ joint.ℓlikelihood(θ)
end


struct TemperedJoint{Tℓprior, Tℓll, T<:Real} <: Function
    ℓprior      :: Tℓprior
    ℓlikelihood :: Tℓll
    β           :: Real
end

function (tj::TemperedJoint)(θ)
    return tj.ℓprior(θ) .+ (tj.ℓlikelihood(θ) .* tj.β)
end


function MCMCTempering.make_tempered_model(
    model::DifferentiableDensityModel,
    β::Real
)
    ℓπ_β = TemperedJoint(model.ℓπ.ℓprior, model.ℓπ.ℓlikelihood, β)
    ∂ℓπ∂θ_β = TemperedJoint(model.∂ℓπ∂θ.ℓprior, model.∂ℓπ∂θ.ℓlikelihood, β)
    model_β = DifferentiableDensityModel(ℓπ_β, ∂ℓπ∂θ_β)
    return model_β
end



#####################
#   SWAPPING
#####################

function MCMCTempering.make_tempered_loglikelihood(
    model::DifferentiableDensityModel,
    β::Real
)
    function logπ(z)
        return model.ℓπ.ℓlikelihood(z) * β
    end
    return logπ
end


function MCMCTempering.get_params(trans::Transition)
    return trans.z.θ
end
