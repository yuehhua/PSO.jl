__precompile__(true)

module PSO
    function make_constraints(::Type{Val{nothing}}, args, kwargs, verbose)
        verbose && println("No constraints given.")
        return x -> [0.0]
    end

    function make_constraints(eqs::Vector, args, kwargs, verbose)
        verbose && println("Converting ieqcons to a single constraint function.")
        return x -> [f(x, args...; kwargs...) for f in eqs]
    end

    function make_constraints(eqs::Function, args, kwargs, verbose)
        verbose && println("Single constraint function given in f_ieqcons.")
        return x -> eqs(x, args...; kwargs...)
    end

    make_constraints(eqs, args, kwargs, verbose) = make_constraints(Val{eqs}, args, kwargs, verbose)

    function update_position!(x, p, fx, fp, fs)
        i_update = (fx .< fp) .& fs
        p[i_update, :] = copy(x[i_update, :])
        fp[i_update, :] = fx[i_update, :]
    end

    function pso(func::Function, lb::Vector, ub::Vector, constraints, args, kwargs,
                 swarmsize, ω, ϕp, ϕg, maxiter, minstep, minfunc, verbose)
        assert(length(ub) == length(lb))
        assert(all(ub .> lb))

        obj = x -> func(x, args...; kwargs...)
        cons = make_constraints(constraints, args, kwargs, verbose)
        is_feasible = x -> all(cons(x) .>= 0)

        # Initialize the particle swarm
        vhigh = abs.(ub .- lb)
        vlow = -vhigh
        S = swarmsize
        D = length(lb)  # the number of dimensions each particle has

        x = lb' .+ rand(S, D) .* (ub .- lb)'  # particle positions
        v = vlow' .+ rand(S, D) .* (vhigh .- vlow)'  # particle velocities
        p = zeros(S, D)  # best particle positions

        fx = [obj(x[i, :]) for i = 1:S]  # current particle function values
        fs = [is_feasible(x[i, :]) for i = 1:S]  # feasibility of each particle
        fp = ones(S) * Inf  # best particle function values

        g = copy(x[1, :])  # best swarm position
        fg = Inf  # best swarm position starting value

        # Store particle's best position (if constraints are satisfied)
        update_position!(x, p, fx, fp, fs)

        # Update swarm's best position
        i_min = indmin(fp)
        if fp[i_min] < fg
            g = copy(p[i_min, :])
            fg = fp[i_min]
        end

        # Iterate until termination criterion met
        it = 1
        while it <= maxiter
            rp = rand(S, D)
            rg = rand(S, D)

            # Update the particles' velocities and positions
            v = ω*v .+ ϕp*rp.*(p .- x) .+ ϕg*rg.*(g' .- x)
            x += v
            # Correct for bound violations
            maskl = x .< lb'
            masku = x .> ub'
            x = x.*(.~(maskl .| masku)) .+ lb'.*maskl .+ ub'.*masku

            # Update objectives and constraints
            for i = 1:S
                fx[i] = obj(x[i, :])
                fs[i] = is_feasible(x[i, :])
            end

            # Store particle's best position (if constraints are satisfied)
            update_position!(x, p, fx, fp, fs)

            # Compare swarm's best position with global best position
            i_min = indmin(fp)
            if fp[i_min] < fg
                verbose && println("New best for swarm at iteration $(it): $(p[i_min, :]) $(fp[i_min])")

                p_min = copy(p[i_min, :])
                stepsize = √(sum((g .- p_min).^2))

                if abs.(fg .- fp[i_min]) <= minfunc
                    verbose && println("Stopping search: Swarm best objective change less than $(minfunc)")
                    return (g, fg, p, fp)
                end
                if stepsize <= minstep
                    verbose && println("Stopping search: Swarm best position change less than $(minstep)")
                    return (g, fg, p, fp)
                end

                g = copy(p_min)
                fg = fp[i_min]
            end

            verbose && println("Best after iteration $(it): $(g) $(fg)")
            it += 1
        end

        println("Stopping search: maximum iterations reached --> $(maxiter)")
        is_feasible(g) || print("However, the optimization couldn't find a feasible design. Sorry")
        return (g, fg, p, fp)
    end

    function pso(func, lb, ub; constraints=nothing, args=(), kwargs=Dict(), swarmsize=100,
                 omega=0.5, phip=0.5, phig=0.5, maxiter=100, minstep=1e-8, minfunc=1e-8,
                 verbose=false, particle_output=false)
        g, fg, p, fp = pso(func, lb, ub, constraints, args, kwargs,
            swarmsize, omega, phip, phig, maxiter, minstep, minfunc, verbose)
        return particle_output? (g, fg, p, fp) : (g, fg)
    end

    export pso;

end # module
