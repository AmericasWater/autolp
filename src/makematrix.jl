using ForwardDiff
using Distributions
using MathProgBase
using NamedArrays

# Linear programming problem:
#   min f' x
#   s.t. A x <= b, Aeq x = beq, LB <= x <= UB

vardefs = [];

function definevars(defs::Vector{Tuple{ASCIIString, Int64}})
    global vardefs

    vardefs = defs;
end

function getvars()
    allvars = ASCIIString[]
    for (name, num) in vardefs
        allvars = [allvars; ASCIIString["$name$ii" for ii in 1:num]]
    end

    allvars
end

"""
Translate an arbitrary function `objective` into a LP minimization vector near `x0`.
"""
function objectivevector(objective, x0; args=[])
    asfloats = float([x0...])
    result = ForwardDiff.gradient(xx -> objective(splitargs(xx, args)...))(asfloats)
    if args == [] && vardefs != []
        named = NamedArray(result)
        setnames!(named, getvars(), 1)
        named
    else
        result
    end
end

"""
Translate arbitrary constraint functions into LP constraints.
  - constraints is a list of functions all of which take the same arguments
    the functions should return < 0 when satistifed.
  - x0 is a set of arguments near which the constraints should be evaluated
  - Assumes local linearity
"""
function constraintmatrix(constraints, x0; names=[], args=[])
    asfloats = float([x0...])

    A = Float64[]
    b = Float64[]
    for constraint in constraints
        A = [A; ForwardDiff.gradient(xx -> constraint(splitargs(xx, args)...), asfloats)]
        b = [b; constraint(splitargs(x0, args)...)] # Not the final values for 'b' yet!
    end

    A = reshape(A, (length(x0), div(length(A), length(x0))))'
    b = A * x0 - b
    # Clean up -0.0 in A
    A[A .== 0] = 0

    if args == [] && vardefs != []
        nA = NamedArray(A)
        setnames!(nA, getvars(), 2)
        if names != []
            setnames!(nA, names, 1)
        end
        setdimnames!(nA, "Variable", 2)
        setdimnames!(nA, "Constraint", 1)

        nb = NamedArray(b)
        setnames!(nb, names, 1)

        return nA, nb
    else
        return A, b
    end
end

"""
Return xx as either itself, for application to a function with as many
arguments, or as a vector of vectors, for a function of vectors.
"""
function splitargs(xx, args)
    if length(args) == 0
        if vardefs != []
            args = map(nn -> nn[2], vardefs)
        else
            return xx
        end
    end

    ## At this point, have a args variable
    xxs = Vector{Number}[]
    for ii in 1:length(args)
        start = sum(args[1:ii-1])+1
        index = sum(args[1:ii])
        push!(xxs, xx[start:index])
    end

    xxs
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
