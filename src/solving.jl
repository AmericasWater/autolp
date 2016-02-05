include("makematrix.jl")

# Suggested method from the ones below: maparmijomax

# From http://web.mit.edu/15.053/www/AMP-Chapter-13.pdf
#   with some help from Menke's Inversion Theory notes

# Frank-Wolfe applies linear approximations to a nonlinear problem
# with linear constraints
# tolerance is the maximum by which x-values change between
# iterations; when small, we hae converged.
# NOTE: segmentmax is not defined yet (not used for MAP method below)
function frankwolfemax(objective, A, b, ub, x0, tolerance)
    gradfunc = ForwardDiff.gradient(objective)
    errors = Inf * ones(length(x0))
    while (sum(abs(errors)) > tolerance)
        grad = gradfunc(x0)
        y0 = linprog(-grad, A, '<', b, x0 * 0, ub) # XXX: assumes linprog minimizes
        x1 = segmentmax(objective, x0, y0, A, b, ub)

        # Are we close enough to an answer?
        errors = x1 - x0
        x0 = x1
    end
end

# MAP (Method of Approximation Programming) extends Frank-Wolfe with
# nonlinear constraints and uses small steps to ensure that we stay
# within bounds and do not need the segment maximization method.
function mapmax(objective, constraints, lb, ub, x0, tolerance)
    gradfunc = ForwardDiff.gradient(objective)
    constraintgradfuncs = []
    for constraint in constraints
        constraintgradfuncs = [constraintgradfuncs; ForwardDiff.gradient(constraint)]
    end

    errors = Inf * ones(length(x0))
    while (sum(abs(errors)) > tolerance)
        grad = gradfunc(x0)

        A = []
        b = []
        for ii in 1:length(constraints)
            A = [A; constraintgradfuncs[ii](x0)]
            b = [b; constraints[ii](x0)] # Not the final values for 'b' yet!
        end

        b = A * x0 - b

        x1 = linprog(-grad, A, '<', b, x0 - tolerance, x0 + tolerance) # XXX: assumes linprog minimizes

        # Are we close enough to an answer?
        errors = x1 - x0
        x0 = x1
    end
end

# Armijo's Rule: When fails, a gradient-based method should use a
# smaller step size.
function armijorule(objective, gradx0, alpha, x0, x1)
    println("Armijo")
    unitvec = -gradx0 / sqrt(sum(gradx0.^2))
    println(objective(x1))
    println(objective(x0))
    println(10e-4 * alpha * unitvec' * gradx0)
    result = objective(x1) .<= objective(x0) + 10e-4 * alpha * unitvec' * gradx0
    result[1]
end

# My modified version of MAP, which reduces the step-size by Armijo's rule
function maparmijomax(objective, constraints, lb, ub, x0, alpha, tolerance)
    gradfunc = ForwardDiff.gradient(objective)
    constraintgradfuncs = []
    for constraint in constraints
        constraintgradfuncs = [constraintgradfuncs; ForwardDiff.gradient(constraint)]
    end

    errors = Inf * ones(length(x0))
    while (sum(abs(errors)) > tolerance)
        println(x0)
        grad = gradfunc(x0)
        println(grad)

        A = x0' # Drop later
        b = [x0[1]] # Drop later
        for ii in 1:length(constraints)
            println(constraintgradfuncs[ii](x0))
            A = [A; constraintgradfuncs[ii](x0)']
            b = [b; constraints[ii](x0)] # Not the final values for 'b' yet!
        end

        if length(A) > 0
            A = A[2:end, :]
            b = b[2:end]
            println(size(A))
            println(size(x0))
            b = A * x0 - b
        end

        passes = false
        x1 = Union{}
        while ~passes
            println(alpha)
            sol = linprog(grad, A, '<', b, x0 - alpha, x0 + alpha)
            x1 = sol.sol
            println(x1)
            passes = all(map(constraint -> constraint(x1) < 0, constraints))
            println(passes)
            passes = passes && armijorule(objective, grad, alpha, x0, x1)
            println(passes)
            if ~passes
                alpha = alpha / 2
            end
        end

        # Are we close enough to an answer?
        errors = x1 - x0
        x0 = x1
    end

    x0
end

# Optimize a univariate objective, by constructing a close linear
# approximation
# NOTE: Not using Hessian cross-terms, so probably over-computing
function univariatematrix(objective, lb, ub, x0, tolerance)
    hess, allresults = hessian(unaryobjective, x0, AllResults)
    pointslope = ForwardDiff.gradient(allresults)
    pointvalue = value(allresults)

    safespan = sqrt(2 * tolerance / hess[1, 2])
    if ub < x0
        trueslope = objective(x0 + safespan) / safespan
        leftfs, leftmins, leftmaxs = univariatematrix(objective, x0 + safespan, ub,
                                          x0 + safespan, tolerance)
    else
        leftfs = leftmins = leftmaxs = []
    end

    if ub > x0
        trueslope = objective(x0 - safespan) / safespan
        rightfs, rightmins, rightmaxs = univariatematrix(objective, lb, x0 - safespan,
                                                         x0 - safespan, tolerance)
    else
        rightfs = rightmins = rightmaxs = []
    end

    fs = [trueslope -trueslope fs]
    mins = [0 0 mins]
    maxs = [safespan safespan maxs]

    linprog(fs, [], '<', [], mins, maxs)
end

# Test
function testfunction(mm, xx)
    sin(20 * mm[1] * xx) + mm[1] * mm[2]
end

obsxx = Vector{Float64}(linspace(0, 1, 40))
obsyy = testfunction([1.21, 1.54], obsxx) + rand(Normal(0, 1), 40)

# Calculate the error to minimize it
function testobjective(mm)
    sum(testfunction(mm, obsxx) - obsyy)^2
end

@time sol2 = maparmijomax(testobjective, [mm -> mm[1] + mm[2] - 3.], [0, 0], [2, 2], [1., 1.], .5, 1e-6)
println(sol2)
@time sol1 = maparmijomax(testobjective, [], [0, 0], [2, 2], [1., 1.], .5, 1e-6)
println(sol1)


