# On the CPU, the check all(isfinite, max_) pays for itself,
# as the broadcast without Inf checks is 50% faster.
# But on the GPU, the computation is free, but all(isfinite, max_) costs an
# extra synchronisation, almost 2x slower when x = CUDA.rand(Float32, 100, 1000)

function NNlib.softmax!(out::CuArray{T}, x::CuArray; dims = 1) where {T}
    max_ = NNlib.fast_maximum(x; dims)
    # if all(isfinite, max_)
    #     @fastmath out .= exp.(x .- max_)
    # else
        @fastmath @. out = ifelse(isequal(max_,Inf32), ifelse(isequal(x,Inf32), 1f0, 0f0), exp(x - max_))
    # end
    tmp = dims isa Colon ? sum(out) : sum!(max_, out)
    out ./= tmp
end

function NNlib.logsoftmax!(out::CuArray{T}, x::CuArray; dims = 1) where {T}
    max_ = NNlib.fast_maximum(x; dims)
    # if all(isfinite, max_)
    #     out .= x .- max_
    # else
        @. out = ifelse(isequal(max_,Inf32), ifelse(isequal(x,Inf32), 0f0, -Inf32), x - max_)
    # end
    @fastmath log_ = log.(sum(exp, out; dims))
    out .-= log_
end
