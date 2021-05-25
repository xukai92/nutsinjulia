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


struct TemperedJoint{Tℓprior, Tℓll, T<:AbstractFloat} <: Function
    ℓprior      :: Tℓprior
    ℓlikelihood :: Tℓll
    β           :: T
end

function (tj::TemperedJoint)(θ)
    return tj.ℓprior(θ) .+ (tj.ℓlikelihood(θ) .* tj.β)
end


function MCMCTempering.make_tempered_model(model::DifferentiableDensityModel, β::T) where {T<:AbstractFloat}
    ℓπ_β = TemperedJoint(model.ℓπ.ℓprior, model.ℓπ.ℓlikelihood, β)
    ∂ℓπ∂θ_β = TemperedJoint(model.∂ℓπ∂θ.ℓprior, model.∂ℓπ∂θ.ℓlikelihood, β)
    model_β = DifferentiableDensityModel(ℓπ_β, ∂ℓπ∂θ_β)
    return model_β
end



#####################
#   SWAPPING
#####################

function MCMCTempering.make_tempered_logπ(model::DifferentiableDensityModel, β::T) where {T<:AbstractFloat}
    function logπ(z)
        return model.ℓπ(z) * β
    end
    return logπ
end


function MCMCTempering.get_θ(trans::Transition)
    return trans.z.θ
end
