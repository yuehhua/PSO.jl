__precompile__(true)

module PSO

    # mutable struct Particle
    #     dimension::Int64
    #     position::Vector
    #     velocity::Vector
    #     best_pos::Vector
    #     best_err::Float64
    #     current_err::Float64
    #
    #     function Particle(x0)
    #         dim = length(x0)
    #         return new(dim, 2 * rand(dim) - 1.0, [x0...], [], -1.0, -1.0)
    #     end
    # end
    #
    # function evaluate(particle::Particle, costFunc)
    #     particle.current_err = costFunc(particle.position)
    #     if particle.current_err < particle.best_err || particle.best_err == -1.0
    #         particle.best_pos = particle.position
    #         particle.best_err = particle.current_err
    #     end
    # end
    #
    # function update_velocity!(particle::Particle, grp_best_pos, w=0.5, c1=1.0, c2=2.0)
    #     for i in 1:particle.dimension
    #         vel_cognitive = c1 * rand() * (particle.best_pos[i] - particle.position[i])
    #         vel_social = c2 * rand() * (grp_best_pos[i] - particle.position[i])
    #         particle.velocity[i] = w * particle.velocity[i] + vel_cognitive + vel_social
    #     end
    # end
    #
    # function update_position!(particle::Particle, bounds)
    #     for i in 1:particle.dimension
    #         particle.position[i] = particle.position[i] + particle.velocity[i]
    #
    #         # adjust maximum position if necessary
    #         if particle.position[i] > bounds[i][2]
    #             particle.position[i] = bounds[i][2]
    #         end
    #
    #         # adjust minimum position if neseccary
    #         if particle.position[i] < bounds[i][1]
    #             particle.position[i] = bounds[i][1]
    #         end
    #     end
    # end
    #
    #
    # function pso(costFunc, initial, bounds, num_particles, maxiter)
    #     dimension = length(initial)
    #     grp_best_err = -1.0
    #     grp_best_pos = []
    #     swarm = Particle[Particle(initial) for i in 1:dimension]
    #     for i in 1:maxiter
    #         # cycle through particles in swarm and evaluate fitness
    #         for j in 1:dimension
    #             evaluate(swarm[j], costFunc)
    #
    #             # determine if current particle is the best (globally)
    #             if swarm[j].current_err < grp_best_err || grp_best_err == -1
    #                 grp_best_pos = swarm[j].position
    #                 grp_best_err = swarm[j].current_err
    #             end
    #         end
    #
    #         # cycle through swarm and update velocity and position
    #         for j in 1:dimension
    #             update_velocity!(swarm[j], grp_best_pos)
    #             update_position!(swarm[j], bounds)
    #         end
    #     end
    #
    #     println("FINAL:\n$(grp_best_pos)\n$(grp_best_err)")
    # end

    function pso(func::Function, lb::Vector, ub::Vector; ieqcons::Vector=[],
                 f_ieqcons=nothing, args=(), kwargs=Dict(),
                 swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100,
                 minstep=1e-8, minfunc=1e-8, debug=false, processes=1, particle_output=false)
        assert(length(ub) == length(lb))
        assert(all(ub .> lb))

        vhigh = abs.(ub .- lb)
        vlow = -vhigh

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
        S = swarmsize
        D = length(lb)  # the number of dimensions each particle has
        x = rand(S, D)  # particle positions
        v = zeros(S, D)  # particle velocities
        p = zeros(S, D)  # best particle positions
        fx = zeros(S)  # current particle function values
        fs = falses(S)  # feasibility of each particle
        fp = ones(S) * Inf  # best particle function values
        g = []  # best swarm position
        fg = Inf  # best swarm position starting value

        # Initialize the particle's position
        x = lb' .+ x .* (ub .- lb)'

        # Calculate objective and constraints for each particle
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

        # Update swarm's best position
        i_min = indmin(fp)
        if fp[i_min] < fg
            fg = fp[i_min]
            g = copy(p[i_min, :])
        else
            # At the start, there may not be any feasible starting point, so just
            # give it a temporary "best" point since it's likely to change
            g = copy(x[0, :])
        end

        # Initialize the particle's velocity
        v = vlow' .+ rand(S, D) .* (vhigh .- vlow)'

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
