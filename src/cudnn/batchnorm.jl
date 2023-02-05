using cuDNN: cudnnNormalizationForward!, cudnnNormalizationBackward, 
             CUDNN_NORM_PER_CHANNEL, CUDNN_TENSOR_NCHW


mutable struct BNCache
    mean
    ivar
end

BNCache() = BNCache(nothing, nothing)

@inline _wsize(x::AbstractArray{<:Any,N}) where N = ntuple(i -> i == N-1 ? size(x, N-1) : 1, N)

function batchnorm(g::DenseCuArray{T}, b::DenseCuArray{T}, x::Union{DenseCuArray{T,4},DenseCuArray{T,5}},
    running_mean, running_var, momentum; kws...) where T<:CUDNNFloat
    batchnorm!(similar(x), g, b, x, running_mean, running_var, momentum; kws...)
end

function batchnorm!(y::DenseCuArray{T}, scale::DenseCuArray{T}, bias::DenseCuArray{T}, x::DenseCuArray{T},
        running_mean, running_var, momentum;
        cache = nothing,
        alpha = T(1), beta = T(0),
        eps = T(1e-5),
        training = true,
        affine = true,
        track_stats = true,
        
        workspace = nothing,
        reserveSpace = nothing,
        ) where T

    dims = _wsize(x)
    mode = CUDNN_NORM_PER_CHANNEL
    format = CUDNN_TENSOR_NCHW


    if running_mean === nothing || running_var === nothing
        running_mean !== running_var && throw(ArgumentError("both or neither of running_mean and running_var must be nothing"))
        if track_stats || !training
            running_mean = fill!(similar(x, dims), 0)
            running_var = fill!(similar(x, dims), 1)
        end
    end

    # default training  => momentum > 0, use_estimates=false, gradients calculated
    # default inference => momentum = 0, use_estimates=true,  gradients not calculated

    kws = (; mode, format, alpha, beta, epsilon=eps, workspace, reserveSpace)
    if training && cache !== nothing
        savedMean = fill!(similar(x, dims), 0)
        savedInvVariance = fill!(similar(x, dims), 1)
    else
        savedMean = nothing
        savedInvVariance = nothing
    end

    cudnnNormalizationForward!(y, x, running_mean, running_var, bias, scale; 
                            training, exponentialAverageFactor=momentum, 
                            savedMean, savedInvVariance,
                            kws...)

    if training && cache !== nothing
        cache.mean = savedMean
        cache.ivar = savedInvVariance
    end

    return y
end
    # if training
    #     if !use_estimates && momentum == 0
    #         cudnnNormalizationForward!(y, x, nothing, nothing, bias, scale; training=true, exponentialAverageFactor=0, kw...)
    #     elseif !use_estimates && momentum > 0
    #         cudnnNormalizationForward!(y, x, mean_estimate, var_estimate, bias, scale; training=true, exponentialAverageFactor=momentum, kw...)
    #     elseif use_estimates && momentum == 0
    #         ((x .- mean_estimate) ./ sqrt.(epsilon .+ var_estimate)) .* scale .+ bias
    #     elseif use_estimates && momentum > 0
    #         update_estimates!(x, mean_estimate, var_estimate, momentum)
    #         ((x .- mean_estimate) ./ sqrt.(epsilon .+ var_estimate)) .* scale .+ bias
    #     end
    # else
    #     if !use_estimates && momentum == 0
    #         cudnnNormalizationForward(x, nothing, nothing, bias, scale; training=true, exponentialAverageFactor=0, kw...)
    #     elseif !use_estimates && momentum > 0
    #         cudnnNormalizationForward(x, mean_estimate, var_estimate, bias, scale; training=true, exponentialAverageFactor=momentum, kw...)
    #     elseif use_estimates && momentum == 0
    #         cudnnNormalizationForward(x, mean_estimate, var_estimate, bias, scale; training=false, kw...)
    #     elseif use_estimates && momentum > 0
    #         update_estimates!(x, mean_estimate, var_estimate, momentum)
    #         cudnnNormalizationForward(x, mean_estimate, var_estimate, bias, scale; training=false, kw...)
    #     end
    # end
# end


# function update_estimates!(x, mean_estimate, var_estimate, update)
#     (x, mean_estimate, var_estimate, update) = value.((x, mean_estimate, var_estimate, update))
#     dims = findall(size(mean_estimate) .== 1)
#     xmean = mean(x; dims)
#     xvar  = var(x; dims, mean=xmean, corrected=false)
#     update = eltype(x)(update)
#     mean_estimate .= xmean * update + mean_estimate * (1-update)
#     var_estimate  .= xvar  * update + var_estimate  * (1-update)
# end



# function batchnorm(
#   x::GPUVal, mean_estimate::GPUVal, var_estimate::GPUVal, bias::GPUVal, scale::GPUVal;
#   use_estimates = !Knet.training(),
#   update =training ? 0.1 : 0.0,
#   epsilon = 1e-5,
#   mode = nothing,
#   format = nothing,
#   savedMean = nothing,
#   savedVar = nothing,
#   workspace = nothing,
#   reserveSpace = nothing,
#   dx = Ref{Any}(nothing),
#   dscale = Ref{Any}(nothing),
#   dbias = Ref{Any}(nothing),
#   o...)
#   @assert size(mean_estimate) == size(var_estimate) == size(bias) == size(scale)
#   n = ndims(x)
#   if size(mean_estimate) == ntuple(i->(i===n-1 ? size(x,i) : 1), n)
#       mode === nothing ? mode = CUDNN_NORM_PER_CHANNEL : @assert mode === CUDNN_NORM_PER_CHANNEL
#       format === nothing ? format = CUDNN_TENSOR_NCHW : @assert format === CUDNN_TENSOR_NCHW
#   elseif size(mean_estimate) == ntuple(i->(i===1 ? size(x,i) : 1), n)
#       mode === nothing ? mode = CUDNN_NORM_PER_CHANNEL : @assert mode === CUDNN_NORM_PER_CHANNEL
#       format === nothing ? format = CUDNN_TENSOR_NHWC : @assert format === CUDNN_TENSOR_NHWC
#   elseif size(mean_estimate) == ntuple(i->(i===n ? 1 : size(x,i)), n)
#       mode === nothing ? mode = CUDNN_NORM_PER_ACTIVATION : @assert mode === CUDNN_NORM_PER_ACTIVATION
#       format === nothing ? format = CUDNN_TENSOR_NCHW : @assert format === CUDNN_TENSOR_NCHW
#   else
#       error("Unsupported batchnorm size x=$(size(x)) m=$(size(m))")
#   end
#   # default training  => update > 0, use_estimates=false, gradients calculated
#   # default inference => update = 0, use_estimates=true,  gradients not calculated
#   # Other combinations must be manually implemented
#   kw = (; mode, format, epsilon, savedMean, savedInvVariance=savedVar, workspace, reserveSpace, dx, dscale, dbias)
#   if training && !use_estimates && update == 0
#       cudnnNormalizationForward(x, nothing, nothing, bias, scale; training=true, exponentialAverageFactor=0, kw...)
#   elseif training && !use_estimates && update > 0
#       (mean_estimate, var_estimate) = value.((mean_estimate, var_estimate))
#       cudnnNormalizationForward(x, mean_estimate, var_estimate, bias, scale; training=true, exponentialAverageFactor=update, kw...)
#   elseif training && use_estimates && update == 0
#       ((x .- mean_estimate) ./ sqrt.(epsilon .+ var_estimate)) .* scale .+ bias
#   elseif training && use_estimates && update > 0
#       update_estimates!(x, mean_estimate, var_estimate, update)
#       ((x .- mean_estimate) ./ sqrt.(epsilon .+ var_estimate)) .* scale .+ bias
#   elseif !training && !use_estimates && update == 0
#       cudnnNormalizationForward(x, nothing, nothing, bias, scale; training=true, exponentialAverageFactor=0, kw...)
#   elseif !training && !use_estimates && update > 0
#       (mean_estimate, var_estimate) = value.((mean_estimate, var_estimate))
#       cudnnNormalizationForward(x, mean_estimate, var_estimate, bias, scale; training=true, exponentialAverageFactor=update, kw...)
#   elseif !training && use_estimates && update == 0
#       (mean_estimate, var_estimate) = value.((mean_estimate, var_estimate))
#       cudnnNormalizationForward(x, mean_estimate, var_estimate, bias, scale; training=false, kw...)
#   elseif !training && use_estimates && update > 0
#       (mean_estimate, var_estimate) = value.((mean_estimate, var_estimate))
#       update_estimates!(x, mean_estimate, var_estimate, update)
#       cudnnNormalizationForward(x, mean_estimate, var_estimate, bias, scale; training=false, kw...)
#   end
# end

# function update_estimates!(x, mean_estimate, var_estimate, update)
#   (x, mean_estimate, var_estimate, update) = value.((x, mean_estimate, var_estimate, update))
#   dims = findall(size(mean_estimate) .== 1)
#   xmean = mean(x; dims)
#   xvar  = var(x; dims, mean=xmean, corrected=false)
#   update = eltype(x)(update)
#   mean_estimate .= xmean * update + mean_estimate * (1-update)
#   var_estimate  .= xvar  * update + var_estimate  * (1-update)
# end













##########################################
##### FROM KNEt
# function batchnorm(
#     x, mean_estimate, var_estimate, bias, scale;
#     epsilon = 1e-5,
#     update =training ? 0.1 : 0.0,
#     use_estimates = !Knet.training(),
#     o...
# )
#     update,epsilon = eltype(x).((update,epsilon))
#     if update > 0 || !use_estimates
#         dims = findall(size(mean_estimate) .== 1)
#         xmean = mean(x; dims)
#         xvar  = var(x; dims, mean=xmean, corrected=false)
#     end
#     if update > 0
#         (m, v, xm, xv) = value.((mean_estimate, var_estimate, xmean, xvar))
#         m .= xm * update + m * (1-update)
#         v .= xv * update + v * (1-update)
#     end        
#     if use_estimates
#         y = ((x .- mean_estimate) ./ sqrt.(epsilon .+ var_estimate)) .* scale .+ bias
#     else
#         y = ((x .- xmean) ./ sqrt.(epsilon .+ xvar)) .* scale .+ bias
#     end
#     return y
# end
############