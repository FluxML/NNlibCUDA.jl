NNlib._batched_mul!(::Type{DT}, C, A, B, α::Number, β::Number) where {DT<:CuArray{T}} where {T<:Float16} =
    NNlib._batched_try_gemm!(DT, C, A, B, α, β)

# Batched matrix multiplication
# 1st argument is produced by NNlib.storage_type(A)
NNlib._batched_gemm!(::Type{<:CuArray}, transA::Char, transB::Char, α::Number, A, B, β::Number, C) =
     CUBLAS.gemm_strided_batched!(transA, transB, α, A, B, β, C)

Base.unsafe_convert(::Type{CuPtr{T}}, A::NNlib.BatchedAdjOrTrans{T}) where {T} =
    Base.unsafe_convert(CuPtr{T}, parent(A))
