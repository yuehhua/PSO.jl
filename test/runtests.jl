using PSO
using Base.Test
using FactCheck

facts("Example minimization of 4th-order banana function") do
    function myfunc(x)
        x1, x2 = x[1], x[2]
        return x1^4 - 2*x2*x1^2 + x2^2 + x1^2 - 2*x1 + 5
    end

    lb = [-3.0, -1.0]
    ub = [2.0, 6.0]

    context("no constraints") do
        xopt1, fopt1 = pso(myfunc, lb, ub)
        x, y = xopt1
        @fact x --> roughly(1.0; atol=1e-3)
        @fact y --> roughly(1.0; atol=1e-3)
        @fact fopt1 --> roughly(4.0, atol=1e-8)

        println("The optimum is at:")
        println("    $(xopt1)")
        println("Optimal function value:")
        println("    myfunc: $(fopt1)")
    end

    function mycon(x)
        x1, x2 = x[1], x[2]
        return [-(x1 + 0.25)^2 + 0.75*x2]
    end

    context("with constraint") do
        # xopt2, fopt2 = pso(myfunc, initial, bounds, num_particles, maxiter)
        xopt2, fopt2 = pso(myfunc, lb, ub, constraints=mycon, swarmsize=1000)
        x, y = xopt2
        @fact x --> roughly(0.5; atol=1e-2)
        @fact y --> roughly(0.75; atol=1e-1)
        @fact fopt2 --> roughly(4.5, atol=1e-3)

        println("The optimum is at:")
        println("    $(xopt2)")
        println("Optimal function value:")
        println("    myfunc: $(fopt2)")
        println("    mycon : $(mycon(xopt2))")
    end
end



facts("Engineering example: minimization of twobar truss weight, subject to") do
    println("  Yield Stress <= 100 kpsi")
    println("  Yield Stress <= Buckling Stress")
    println("  Deflection   <= 0.25 inches")

    function weight(x, B, ρ, E, P)
        H, d, t = x  # all in inches
        return ρ*2*π*d*t*√((B/2)^2 + H^2)
    end

    function stress(x, B, ρ, E, P)
        H, d, t = x  # all in inches
        return (P * √((B/2)^2 + H^2)) / (2*t*π*d*H)
    end

    function buckling_stress(x, B, ρ, E, P)
        H, d, t = x  # all in inches
        return (π^2 * E * (d^2 + t^2)) / (8 * ((B/2)^2 + H^2))
    end

    function deflection(x, B, ρ, E, P)
        H, d, t = x  # all in inches
        return (P * (√((B/2)^2 + H^2))^3) / (2*t*π*d*H^2 * E)
    end

    function mycons(x, B, ρ, E, P)
        strs = stress(x, B, ρ, E, P)
        buck = buckling_stress(x, B, ρ, E, P)
        defl = deflection(x, B, ρ, E, P)
        return [100 - strs, buck - strs, 0.25 - defl]
    end

    B = 60  # inches
    rho = 0.3  # lb/in^3
    E = 30000  # kpsi
    P = 66  # lb (force)
    args = (B, rho, E, P)
    lb = [10.0, 1.0, 0.01]
    ub = [30.0, 3.0, 0.25]
    context("") do
        xopt4, fopt4 = pso(weight, lb, ub, constraints=mycons, args=args, swarmsize=1000)
        x, y, z = xopt4
        @fact x --> roughly(29.0; atol=1.0)
        @fact y --> roughly(2.5; atol=1.0)
        @fact fopt4 --> roughly(11.88, atol=1e-2)

        println("The optimum is at:")
        println("    $(xopt4)")
        println("Optimal function values:")
        println("    weight         : $(fopt4)")
        println("Constraint functions:")
        println("    stress         : $(stress(xopt4, args...))")
        println("    buckling stress: $(buckling_stress(xopt4, args...))")
        println("    deflection     : $(deflection(xopt4, args...))")
    end
end
