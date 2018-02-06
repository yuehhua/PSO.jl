using PSO
using Base.Test
num_particles = 15
maxiter = 100

############################################################################

println(repeat("*", 65))
println("Example minimization of 4th-order banana function (no constraints)")

function myfunc(x)
    x1, x2 = x[1], x[2]
    return x1^4 - 2*x2*x1^2 + x2^2 + x1^2 - 2*x1 + 5
end

initial = (0.5, 0.5)
bounds = [(-3.0, 2.0), (-1.0, 6.0)]
lb = [-3.0, -1.0]
ub = [2.0, 6.0]
# xopt1, fopt1 = pso(myfunc, initial, bounds, num_particles, maxiter)
@time xopt1, fopt1 = pso(myfunc, lb, ub)
println("The optimum is at:")
println("    $(xopt1)")
println("Optimal function value:")
println("    myfunc: $(fopt1)")


############################################################################

println(repeat("*", 65))
println("Example minimization of 4th-order banana function (with constraint)")

function mycon(x)
    x1, x2 = x[1], x[2]
    return [-(x1 + 0.25)^2 + 0.75*x2]
end

# xopt2, fopt2 = pso(myfunc, initial, bounds, num_particles, maxiter)
@time xopt2, fopt2 = pso(myfunc, lb, ub, f_ieqcons=mycon)

println("The optimum is at:")
println("    $(xopt2)")
println("Optimal function value:")
println("    myfunc: $(fopt2)")
println("    mycon : $(mycon(xopt2))")


############################################################################

println(repeat("*", 65))
println("Engineering example: minimization of twobar truss weight, subject to")
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
@time xopt4, fopt4 = pso(weight, lb, ub, f_ieqcons=mycons, args=args)


println("The optimum is at:")
println("    $(xopt4)")
println("Optimal function values:")
println("    weight         : $(fopt4)")
println("Constraint functions:")
println("    stress         : $(stress(xopt4, args...))")
println("    buckling stress: $(buckling_stress(xopt4, args...))")
println("    deflection     : $(deflection(xopt4, args...))")
