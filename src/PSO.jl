__precompile__(true)

module PSO

    function pso(func::Function, lb::Vector, ub::Vector; ieqcons::Vector=[],
                 f_ieqcons=nothing, args=(), kwargs=Dict(),
                 swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100,
                 minstep=1e-8, minfunc=1e-8, debug=false, processes=1, particle_output=false)
        assert(length(ub) == length(lb))
        assert(all(ub .> lb))

        obj = x -> func(x, args...; kwargs...)

        # Check for constraint function(s)
        local cons
        if f_ieqcons == nothing
            if length(ieqcons) == 0
                debug && println("No constraints given.")
                cons = x -> [0.0]
            else
                debug && println("Converting ieqcons to a single constraint function.")
                cons = x -> [f(x, args...; kwargs...) for f in ieqcons]
            end
        else
            debug && println("Single constraint function given in f_ieqcons.")
            cons = x -> f_ieqcons(x, args...; kwargs...)
        end
        is_feasible = x -> all(cons(x) .>= 0)

        # Initialize the multiprocessing module if necessary
        # if processes > 1:
        #     import multiprocessing
        #     mp_pool = multiprocessing.Pool(processes)

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
        i_update = (fx .< fp) .& fs
        p[i_update, :] = copy(x[i_update, :])
        fp[i_update, :] = fx[i_update, :]

        # Update swarm's best position
        i_min = indmin(fp)
        if fp[i_min] < fg
            fg = fp[i_min]
            g = copy(p[i_min, :])
        end

        # Iterate until termination criterion met
        it = 1
        while it <= maxiter
            rp = rand(S, D)
            rg = rand(S, D)

            # Update the particles velocities
            v = omega*v .+ phip*rp.*(p .- x) .+ phig*rg.*(g' .- x)
            # Update the particles' positions
            x += v
            # Correct for bound violations
            maskl = x .< lb'
            masku = x .> ub'
            x = x.*(.~(maskl .| masku)) .+ lb'.*maskl .+ ub'.*masku

            # Update objectives and constraints
            if processes > 1
                # fx = np.array(mp_pool.map(obj, x))
                # fs = np.array(mp_pool.map(is_feasible, x))
            else
                for i = 1:S
                    fx[i] = obj(x[i, :])
                    fs[i] = is_feasible(x[i, :])
                end
            end

            # Store particle's best position (if constraints are satisfied)
            i_update = (fx .< fp) .& fs
            p[i_update, :] = copy(x[i_update, :])
            fp[i_update, :] = fx[i_update, :]

            # Compare swarm's best position with global best position
            i_min = indmin(fp)
            if fp[i_min] < fg
                debug && println("New best for swarm at iteration $(it): $(p[i_min, :]) $(fp[i_min])")

                p_min = copy(p[i_min, :])
                stepsize = √(sum((g .- p_min).^2))

                if abs.(fg .- fp[i_min]) <= minfunc
                    println("Stopping search: Swarm best objective change less than $(minfunc)")
                    return particle_output? (p_min, fp[i_min], p, fp) : (p_min, fp[i_min])
                end
                if stepsize <= minstep
                    println("Stopping search: Swarm best position change less than $(minstep)")
                    return particle_output? (p_min, fp[i_min], p, fp) : (p_min, fp[i_min])
                end

                g = copy(p_min)
                fg = fp[i_min]
            end

            debug && println("Best after iteration $(it): $(g) $(fg)")
            it += 1
        end

        println("Stopping search: maximum iterations reached --> $(maxiter)")

        if !is_feasible(g)
            print("However, the optimization couldn't find a feasible design. Sorry")
        end

        return particle_output? (g, fg, p, fp) : (g, fg)
    end

    export pso;

end # module
