function print_array_strs(x)
    str = sprint((io, x)->show(io, MIME"text/plain"(), x), x)
    return @view split(str, '\n')[2:end]
end

@testset "BatchedAdjOrTrans" begin
    x = randn(Float32, 3,4,2)
    y = cu(x)

    bax = batched_adjoint(x)
    btx = batched_transpose(x)
    bay = batched_adjoint(y)
    bty = batched_transpose(y)

    rbax = reshape(bax, :)
    rbtx = reshape(btx, :)
    rbay = reshape(bay, :)
    rbty = reshape(bty, :)

    rbax2 = reshape(bax, (12, 2))
    rbtx2 = reshape(btx, (12, 2))
    rbay2 = reshape(bay, (12, 2))
    rbty2 = reshape(bty, (12, 2))

    @test sprint(show, bax) == sprint(show, bay)
    @test sprint(show, btx) == sprint(show, bty)

    @test print_array_strs(bax) == print_array_strs(bay)
    @test print_array_strs(btx) == print_array_strs(bty)

    @test Array(bax) == Array(bay)
    @test collect(bax) == collect(bay)
    @test Array(btx) == Array(bty)
    @test collect(btx) == collect(bty)

    @test sprint(show, rbax) == sprint(show, rbay)
    @test sprint(show, rbtx) == sprint(show, rbty)

    @test print_array_strs(rbax) == print_array_strs(rbay)
    @test print_array_strs(rbtx) == print_array_strs(rbty)

    @test Array(rbax) == Array(rbay)
    @test collect(rbax) == collect(rbay)
    @test Array(rbtx) == Array(rbty)
    @test collect(rbtx) == collect(rbty)

    @test sprint(show, rbax2) == sprint(show, rbay2)
    @test sprint(show, rbtx2) == sprint(show, rbty2)

    @test print_array_strs(rbax2) == print_array_strs(rbay2)
    @test print_array_strs(rbtx2) == print_array_strs(rbty2)

    @test Array(rbax2) == Array(rbay2)
    @test collect(rbax2) == collect(rbay2)
    @test Array(rbtx2) == Array(rbty2)
    @test collect(rbtx2) == collect(rbty2)

end
