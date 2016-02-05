using ForwardDiff
using Distributions
using MathProgBase

# Linear programming problem:
#   min f' x
#   s.t. A x <= b, Aeq x = beq, LB <= x <= UB

"""
Translate an arbitrary function `objective` into a LP minimization vector near `x0`.
"""
function objectivevector(objective, x0)
    asfloats = float([x0...])
    -ForwardDiff.gradient(xx -> objective(xx...))(asfloats)
end

"""
Translate arbitrary constraint functions into LP constraints.
  - constraints is a list of functions all of which take the same arguments
    the functions should return < 0 when satistifed.
  - x0 is a set of arguments near which the constraints should be evaluated
  - Assumes local linearity
"""
function constraintmatrix(constraints, x0)
    asfloats = float([x0...])

    A = Float64[]
    b = Float64[]
    for constraint in constraints
        A = [A; ForwardDiff.gradient(x -> constraint(x...), asfloats)]
        b = [b; constraint(x0...)] # Not the final values for 'b' yet!
    end

    A = reshape(A, (div(length(A), length(x0)), length(x0)))'
    b = A * x0 - b

    return A, b
end

if ARGS == ["--test"]
    # Use the second example from http://www.purplemath.com/modules/linprog3.htm
    objective(x, y) = 8x + 12y
    constraints = [(x, y) -> 10x + 20y - 140,
                   (x, y) -> 6x + 8y - 72]
    x0 = [0, 0]

    f = objectivevector(objective, x0)
    A, b = constraintmatrix(constraints, x0)

    println("Objective vector:")
    println(f)
    println("Constraint matrix:")
    println(A)
    println(b)

    sol = linprog(f, A, '<', b, [0, 0], [Inf, Inf])

    println("Solution:")
    println(sol.sol)
end
