using NNlibCUDA, NNlib, CUDA

@inline _wsize(x::AbstractArray{<:Any,N}) where N = ntuple(i -> i == N-1 ? size(x, N-1) : 1, N)

# Test batchnorm
x = randn(Float32, 2, 2, 3, 5) |> cu
affine_sz = _wsize(x)
g = fill!(similar(x, affine_sz), 1)
b = fill!(similar(x, affine_sz), 0)
running_var = fill!(similar(x, affine_sz), 1)
running_mean = fill!(similar(x, affine_sz), 0)

NNlibCUDA.batchnorm(g, b, x, running_mean, running_var, 0.1)
