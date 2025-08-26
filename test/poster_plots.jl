using OpenSCI
using PauliOperators
using LinearAlgebra
using DifferentialEquations
using Plots, Measures
using Printf
using OrderedCollections
using StatProfilerHTML

function compute_ρt_exp(t, w, v, λ, ρ0::Vector, dim, mat_ops)
        
    # w = inv(F.vectors)
    # v = F.vectors
    # λ = F.values
    ρt = v * Diagonal(exp.(λ*t)) * w * ρ0
    ρt = reshape(ρt, (dim, dim))
    exp_eig = tr(mat_ops*ρt)
    popu = []
    for i in 1:dim
        push!(popu, ρt[i, i])
    end
    return exp_eig, popu
end

function matrix_to_dyad(mat_p)
    n, m = size(mat_p)
    N = Int(log2(n))
    dyad_pauli = DyadSum(N)
    for i in 1:m
        for j in 1:n
            temp = mat_p[i, j]
            
            if abs(temp) ≠ 0
                dyad_pauli += temp*Dyad(N, i-1, j-1)
            end
        end
    end  
    return dyad_pauli  
end

function pinv_sparsedyads(dyad_dict::SparseDyadVectors{N,T})::SparseDyadVectors{N,T} where {N,T}
    dyad_keys = collect(keys(dyad_dict))                            
    values_matrix = transpose(hcat(values(dyad_dict)...))           

    pinv_matrix = pinv(values_matrix)                              

    pinv_sdv = OrderedDict{DyadBasis{N}, Vector{T}}()
    for i in eachindex(dyad_keys)
        pinv_sdv[dyad_keys[i]] = vec(pinv_matrix[:, i])           
    end

    return pinv_sdv
end

# function expectation_sparse(vi, wi, d_ops, v0, ei, t, R)
#     expval = 0.0
#     for m in 1:R
#         ovi = sum(conj(d_ops[d]) * vi[d][m] for d in keys(vi) if haskey(d_ops, d); init=0.0)
#         # println("ovi:", ovi)
#         wir = sum((wi[d][m]) * v0[d][1] for d in keys(wi) if haskey(v0, d); init=0.0) # Should it be conj(wi)?
#         # println("wvi", wir)
#         expval += exp(ei[m]*t) * ovi * wir
#     end
#     return expval
# end

function expectation_sparse(vi, wi, d_ops, v0, ei, t, R)
    expval = 0.0

    for m in 1:R
        # Compute ovi = ⟨O|v_m⟩
        ovi = 0.0
        for d in keys(vi)
            if haskey(d_ops, d)
                ovi += conj(d_ops[d]) * vi[d][m]
                # display(ovi)
            end
        end

        # Compute wir = ⟨w_m|ρ(0)⟩
        wir = 0.0
        for d in keys(wi)
            if haskey(v0, d)
                wir += wi[d][m] * v0[d][1]   # no conj(wi) for biorthogonal left eigenvectors
            end
        end
        # display(wir)
        # Accumulate contribution for this mode
        expval += exp(ei[m] * t) * ovi * wir
    end

    return expval
end

function population_densities(N, vi, wi, v0, ei, t, R)
    pops = zeros(2^N)
    for i in 0:2^N-1
        obs = DyadSum(Dyad(N, i, i))
        pops[i+1] = abs(expectation_sparse(vi, wi, obs, v0, ei, t, R))
    end
    return pops
end


function run_eig(;N = 2)
    # N = 2
    dim = 2^N

    # Initializing the Lindbladian
    L = Lindbladian(N)
    add_hamiltonian!(L, OpenSCI.heisenberg_1D(N, 1.1, 1.2, 1.3))
    add_channel_dephasing!(L, 0.1)
    add_channel_depolarizing!(L, .1)
    println("Lindbladian: ")
    display(L)

    Lmat = Matrix(L)
    # println("Matrix Form of L: ")
    display(size(Lmat))
    println("Diagonalization started")

    state = DyadSum(Dyad(N, 0, 0))
    vec_state_i = vec(Matrix(state))


    F = eigen(Lmat)
    # Sort Eigenvalues by real part
    println("Diagonalization Ended")

    perm = sortperm(F.values, by=real)
    F.values .= F.values[perm]
    F.vectors .= F.vectors[:, perm]

    # label_exact = reduce(hcat, [["|$(i)><$(i)|"] for i in 0:2^N-1])

    # t = [i for i in sol.t]
    # plt = plot(title = "Evolution under depolarizing noise",
    #          xlabel = "Time",
    #         ylabel = "Population",
    #         # label=label_exact,
    #         ylim = (-0.1, 1),
    #         dpi = 300)

    # for (state, pops) in populations_ode
    #     plot!(t, pops, label = string(state.bra.v),linewidth = 2,
    #     )
    # end

    # savefig("test/ode_$N.png")

    # p1 = scatter(
    #     real.(F.values), imag.(F.values),
    #     title = "Eigenspectrum of Lindbladian: depolarizing channel",
    #     xlabel = "Re(λ)",
    #     ylabel = "Im(λ)",
    #     legend = false,
    #     marker = (:circle, 4),
    #     dpi = 300,
    # )
    
    # savefig(p1, "test/eig_$(N)-new.png")  

    nkeep = 10
    v0 = SparseDyadVectors(state, R = nkeep)
    p_dyad, ei = selected_ci(L, v0, ϵdiscard=1e-4, max_iter_outer = 10)
    vi = p_dyad
    wi = pinv_sparsedyads(vi)
    dim_1, R_1 = size(p_dyad)
    # display((todense(wi))' * todense(wi))
    # display(vi)
    # display("Values")
    # pop_sci = expectation_sparse(vi, wi,DyadSum(Dyad(N, 0, 0)), v0, ei, 0, R_1)
    # display(pop_sci)
    # return


    # return

    time_step = [i/10 for i in 0:50]
    pop_t_ex = []
    pop_t_ex_sci = []
    v = F.vectors
    w = inv(v)
    λ = F.values

    ops = PauliSum(N)

    # for i in 1:N
    #     ops += Pauli(N, Z=[i])
    # end
    # ops = Pauli(N, Z = [3])
    ops += Pauli(N, Z = [2,3])
    mat_ops = Matrix(ops)
    d_ops = matrix_to_dyad(mat_ops)

    for T in time_step
        # println("\nTime: ", T)
    
        # println("\n ==========================EXP==================================")
        # Exact Formalism
        exp_eig, pop_ex = compute_ρt_exp(T, w, v, λ, vec_state_i, dim, mat_ops)

        pop_sci = population_densities(N, vi, wi, v0, ei, T, R_1)
        # out_n = expectation_sparse(vi, wi, d_ops, v0, ei, T, R_1)

        # println("Exp Value using Eigen Values")
        # display(exp_eig)
    
        push!(pop_t_ex, (pop_ex))
        push!(pop_t_ex_sci, (pop_sci))

    end
    
    label_exact = reduce(hcat, [["|$(i)><$(i)|"] for i in 0:2^N-1])

    Y1 = abs.(hcat(pop_t_ex...)')  # Exact results

    p1 = plot(
        time_step, Y1,
        title = "Exact Evolution",
        xlabel = "Time",
        ylabel = "Population",
        # ylim = [-0.1, 1.1],
        ylim = [-0.05, 0.12],
        xlim = [4, 5],
        linewidth = 2,
        # label=label_exact,
        dpi = 300,
        # legend = :topright, 
        legend = false,
        top_margin=5mm,bottom_margin = 5mm, right_margin=5mm, left_margin=5mm,
        # marker = (:circle, 2)
    )
    
    # SCI
    Y2 = abs.(hcat(pop_t_ex_sci...)')
    p2 = plot(
        time_step, Y2,
        title = "SCI Evolution",
        xlabel = "Time",
        ylabel = "Population",
        ylim = [-0.05, 0.12],
        xlim = [4, 5],
        linewidth = 2,
        # label=label_exact,
        dpi = 300,
        # legend = :topright,
        legend = false, 
        top_margin=5mm,bottom_margin = 5mm, right_margin=5mm, left_margin=5mm,
        # marker = (:circle, 2)

    )
    
    # Side-by-side layout
    p = plot(p1, p2, layout=(1,2), size=(1000,400))

    savefig(p, "test/poster_inset_N$N.png")

    # p = plot(
    #     time_step, pop_t_ex,
    #     title = "Two Spin Correlation <Z2 Z3>",
    #     label = "Exact Evolution",
    #     linewidth = 2,
    #     dpi = 300,
    #     xlabel = "Time",
    #     ylabel = "Expectation Value", color = :coral,
    #     top_margin = 5mm#, bottom_margin = 5mm, left_margin = 5mm, right_margin = 5mm
    # )

    # # Add SCI curve
    # plot!(time_step, pop_t_ex_sci,
    #     label = "SCI Evolution",
    #     linewidth = 2, color = :teal,
    # )


    # savefig("test/poster_pop_compare_$(N)-exp.png")
    
end



function run_ode(;N = 2)
    # N = 2
    dim = 2^N

    # Initializing the Lindbladian
    L = Lindbladian(N)
    add_hamiltonian!(L, OpenSCI.heisenberg_1D(N, 1.1, 1.2, 1.3))
    add_channel_dephasing!(L, 0.1)
    add_channel_depolarizing!(L, .1)
    println("Lindbladian: ")
    display(L)

    Lmat = Matrix(L)
    # println("Matrix Form of L: ")
    display(size(Lmat))
    println("ODE started")

    state = DyadSum(Dyad(N, 0, 0))
    vec_state_i = vec(Matrix(state))


    # F = eigen(Lmat)
    # # Sort Eigenvalues by real part
    # perm = sortperm(F.values, by=real)
    # F.values .= F.values[perm]
    # F.vectors .= F.vectors[:, perm]

    T = 10
    f(du, u, p, t) = du .= Lmat*u
    tspan = (0.0, T)
    prob = ODEProblem(f, vec(Matrix(state)), tspan) 
    sol = solve(prob, reltol = 1e-8, abstol = 1e-8, saveat = 0.01)


    populations_ode = Dict{Dyad{N}, Vector{Float64}}([])

    for i in 0:2^N-1
        ii = Dyad(N,i,i)
        ii_idx = index(ii)
        populations_ode[ii] = [abs(v[ii_idx]) for v in sol.u]
    end
    println("ODE ended")

    # label_exact = reduce(hcat, [["|$(i)><$(i)|"] for i in 0:2^N-1])

    t = [i for i in sol.t]
    plt = plot(title = "Evolution under depolarizing noise",
             xlabel = "Time",
            ylabel = "Population",
            # label=label_exact,
            ylim = (-0.1, 1),
            dpi = 300)

    for (state, pops) in populations_ode
        plot!(t, pops, label = string(state.bra.v),linewidth = 2,
        )
    end

    savefig("test/ode_$N.png")

    # p1 = scatter(
    #     real.(F.values), imag.(F.values),
    #     title = "Eigenspectrum of Lindbladian: depolarizing channel",
    #     xlabel = "Re(λ)",
    #     ylabel = "Im(λ)",
    #     legend = false,
    #     marker = (:circle, 4),
    #     dpi = 300,
    # )
    
    # savefig(p1, "test/eig_$(N)-new.png")  

    
    # time_step = [i/10 for i in 0:100]
    # ops = Pauli(N, Z = [1, 2])
    # mat_ops = Matrix(ops)
    # pop_t_ex = []

    # for T in time_step
    #     println("\nTime: ", T)
    
    #     println("\n ==========================EXP==================================")
    #     # Exact Formalism
    #     exp_eig, pop_ex = compute_ρt_exp(T, F, vec_state_i, mat_ops, dim)
    #     println("Exp Value using Eigen Values")
    #     display(exp_eig)
    
    #     push!(pop_t_ex, pop_ex)
    # end
    
    # label_exact = reduce(hcat, [["|$(i)><$(i)|"] for i in 0:2^N-1])

    # Y1 = abs.(hcat(pop_t_ex...)')  # Exact results
    
    # p1 = plot(
    #     time_step, Y1,
    #     title = "Evolution under depolarizing noise",
    #     xlabel = "Time",
    #     ylabel = "Population",
    #     linewidth = 2,
    #     label=label_exact,
    #     ylim = (-0.1, 1),
    #     dpi = 300
    #     # legend = false
    # )
    
    # savefig("test/pop_$(N)-new.png")
end

function run()
    times_func1 = Float64[]
    times_func2 = Float64[]
    n_runs = [i for i in 1:6]
    
    for j in n_runs
        push!(times_func1, @elapsed run_eig(N = j))
        push!(times_func2, @elapsed run_ode(N = j))
    end
    
    plot(n_runs, times_func1, label="Eigenspectrum", marker=:circle, dpi = 300, linewidth = 2)
    plot!(n_runs, times_func2, label="ODE Solver", marker=:diamond, linewidth = 2)
    xlabel!("Run #")
    ylabel!("Time (s)")
    title!("Execution time comparison")
    savefig("test/time_compare_new.png")
end

function run_sci(; N = 2)

    dim = 2^N

    # Initializing the Lindbladian
    L = Lindbladian(N)
    add_hamiltonian!(L, OpenSCI.heisenberg_1D(N, 1.5, 1.2, 1.3))
    add_channel_dephasing!(L, 0.1)
    add_channel_depolarizing!(L, .1)
    # println("Lindbladian: ")
    # display(L)

    Lmat = Matrix(L)
    display(size(Lmat))
    state = DyadSum(Dyad(N, 0, 0))
    vec_state_i = vec(Matrix(state))

    state = DyadSum(Dyad(N, 0, 0))
    nkeep = 2
    v0 = SparseDyadVectors(state, R = nkeep)

    iterations = [i for i in 1:5]
    p_size = []
    # for sci_iter in iterations
        # println("\n SCI_ITER MAX: ", sci_iter)
    p_dyad, eig_sci = selected_ci(L, v0, ϵdiscard=1e-3, max_iter_outer = 10)
        # display(eig_sci)
        # push!(p_size, first(size(p_dyad)))
        # push!(p_size, abs(eig_sci[end]))

        # println("\n P_DYAD: ", p_size)
    # end

    # plot(iterations, p_size, 
    #     dpi = 300,
    #     title = "Eigenvalue Convergence vs. SCI Iteration",
    #     xlabel = "Iterations",
    #     ylabel = "Absolute Error",
    #     legend = false,
    #     marker = :circle,
    #     linewidth = 2,
    #     color = :teal, 
    #     )

    # savefig("test/poster_sci_eig.png")
    return first(size(Lmat)), first(size(p_dyad)), eig_sci[end]
end


function run_sci_comp()
    system = 2:7
    lind_sizes = Float64[]  # Lindbladian matrix sizes
    p_sizes    = Float64[]  # P-space sizes
    eig_val = Float64[]
    for i in system
        lmat, psize, val = run_sci(N = i)
        push!(lind_sizes, lmat)  # assuming lmat is square
        push!(p_sizes, psize)   
        push!(eig_val, abs(val))

    end
    # lind_sizes = [16, 64, 256, 1024, 4096, 16384]  # Lindbladian matrix sizes
    # p_sizes    = [6, 32, 48, 492, 1457, 4560]  # P-space sizes


    # Make plot
    plot(system, eig_val, label=false, marker=:o, lw=2, dpi = 300, title = "Eigenvalue for steady state with restricted SCI iteration")

    # plot(system, lind_sizes, label="Lindbladian Matrix Size", marker=:o, lw=2, dpi = 300)
    # plot!(system, p_sizes, label="P-space Size", marker=:s, lw=2, color = :teal)
    xlabel!("System Size")
    ylabel!("Absolute error")
    title!("Lindbladian vs P-space Size")
    savefig("test/poster_lmat_eig.png")
end


function run_eig_compare()
    N = 6 
    dim = 2^N

    # Initializing the Lindbladian
    L = Lindbladian(N)
    add_hamiltonian!(L, OpenSCI.heisenberg_1D(N, 1.1, 1.2, 1.3))
    add_channel_dephasing!(L, 0.1)
    add_channel_depolarizing!(L, .1)

    Lmat = Matrix(L)

    state = DyadSum(Dyad(N, 0, 0))
    vec_state_i = vec(Matrix(state))

    # Full diagonalization
    F = eigen(Lmat)
    perm = sortperm(F.values, by=real)
    F.values .= F.values[perm]
    F.vectors .= F.vectors[:, perm]

    # SCI approximation
    nkeep = 2
    v0 = SparseDyadVectors(state, R = nkeep)
    p_dyad, eig_sci = selected_ci(L, v0, ϵdiscard=1e-4, max_iter_outer = 10)

    Lmat_sci = build_subspace_L(L, p_dyad)

    F_sci = eigen(Lmat_sci)
    # perm = sortperm(F_sci.values, by=real)
    # F_sci.values .= F_sci.values[perm]
    # F_sci.vectors .= F_sci.vectors[:, perm]

    # Combined plot
    scatter(real.(F.values), imag.(F.values),
    title = "Eigenvalue Spectrum – Highlighting SCI Subspace",
        label = "Full Diagonalization",
        xlabel = "Re(λ)", ylabel = "Im(λ)",
        marker = (:diamond, 4, :orange, 0.5),
        # color = "#d95f02", 
        legend = :topright,
        # aspect_ratio = :equal,
        dpi = 300,
        # size = (600, 500)
    )

    scatter!(real.(F_sci.values), imag.(F_sci.values),
        label = "SCI Approximation",
        marker = (:circle, 2, :teal),
        # color =  "#1b9e77" 
    )

    savefig("test/poster_eig_$N-combined.png")
end

function run_lowdin()
    data = [
        1 0.4017455182 0.1672152452;
        2 0.2298332312 0.1092963124;
        3 0.0453666183 0.004991892731;
        4 3.21E-15 3.21E-15;
        5 9.19E-15 9.19E-15;
        6 4.41E-15 4.41E-15
    ]

#     data = [
#     1 0.6916106946 1.34e2;
#     2 0.6893771038 56.84009605;
#     3 0.5112050447 17.53286758;
#     4 0.1804307443 0.3112513497;
#     5 0.0610933655 0.00186569831;
#     6 4.29e-5 6.90e-5;
#     7 1.17e-5 9.92e-6;
#     8 1.17e-5 9.92e-6
# ]

# data = [

# 1	0.7129068688	0.3638895546;
# 2	0.4532160926	1.419752048;
# 3	0.2327792301	0.7224850221;
# 4	0.07650215734	0.05051238364;
# 5	0.01595547546	0.006943095685;
# 6	2.88E-14	2.88E-14 ]

    index = data[:, 1]
    col1 = data[:, 2]
    col2 = data[:, 3]

    # --- 2. Plot ---
    plot(index, col1, label="SCI error", marker=:circle, linewidth=2, dpi =300, color = :teal, yscale=:log10)
    plot!(index, col2, label="Lowdin Correction Error", marker=:square, linewidth=2)
    xlabel!("SCI Iteration")
    ylabel!("Absolute Error")
    title!("Absolute Error vs SCI Iteration for N = 4")

    # --- 3. Save figure ---
    savefig("test/poter_lowdin_plot_N4_SS.png")
end

# run_lowdin()
run_eig(N=6)
# run_eig_compare()
# run_sci(N=5)
