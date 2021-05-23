#####################
#   MODEL
#####################

function MCMCTempering.make_tempered_model(model::DifferentiableDensityModel, β::T) where {T<:AbstractFloat}
    ℓπ_β(θ) = model.ℓπ(θ) * β
    ∂ℓπ∂θ_β(θ) = model.∂ℓπ∂θ(θ) * β
    model = DifferentiableDensityModel(ℓπ_β, ∂ℓπ∂θ_β)
    return model
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


function MCMCTempering.get_θ(state::HMCState)
    return state.z.θ
end
