using LinearAlgebra, Random, Printf
using Plots



function lowdin_dense_demo()
    Random.seed!(2)
    N = 100
    n = 5
    A = rand(N,N)
    A =(A + A')/2
    A += diagm(1:N)*.2
    App = A[1:n, 1:n]
    Apq = A[1:n, n+1:N]
    Aqp = A[n+1:N, 1:n]
    Aqq = A[n+1:N, n+1:N]
    
    # Initial eigenvalues
    vals_full = eigvals(A)
    ei = eigvals(App)
    exact = vals_full[1]
    @printf(" Exact: %12.8f %12.8fi\n", real(exact), imag(exact))
    # display(ei[1])
    function Aeff(E)
        return App + Apq * inv(E*I - Aqq) * Aqp
    end

    println("\nMETHOD 1 \n")
    for i in 1:20
        if i == 1
            ei = eigvals(Aeff(exact))
            idx = argmin(real(ei)) 
            @printf(" Iter: %2i %12.8f %12.8fi\n", i, real(ei[idx]), imag(ei[idx]))
        else
            idx = argmin(real(ei))
            ei = eigvals(Aeff(ei[idx]))
            idx = argmin(real(ei)) 
            @printf(" Iter: %2i %12.8f %12.8fi\n", i, real(ei[idx]), imag(ei[idx]))
        end

    end

    println("\nMETHOD 2 \n")


    ep, vp = eigen(App)
    idx = argmin(real(ep))
    vp = vp[:,idx]

    for i in 1:20

        vold = vp
        if i == 1
            Ai = Aeff(exact)
        else
            Ai = Aeff(ep[idx])
        end
        ep,vp = eigen(Ai)
        ovlps = real(inv(vp)*vold)
        # ovlps = real(inv(vp)*vold)
        idx = argmax(abs.(ovlps))
        vp = vp[:,idx]
        ovlp = ovlps[idx] 
        # println(ovlps)
        # @printf("iter: %2i idx %i ovlp %12.8f\n", i, idx, ovlp)
        @printf(" Iter: %2i ovlp: %12.8f eig: %12.8f %12.8fi\n", i, ovlp, real(ep[idx]), imag(ep[idx]))
    end

    return 
end

lowdin_dense_demo()
