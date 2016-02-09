include("makematrix.jl")

## 5 counties in a row with links between them
definevars([("C", 5), ("E", 4)]);

# How much resource each county requires
requirements = [1, 2, 3, 4, 5];
# The cost per unit of producing it within each county
productioncosts = [1, 3, 5, 7, 9];
# The cost of transporting a unit of resource to the next county
transportcost = 2;

# Minimize the production + transportation costs
objective(production, transport) =
    sum(production .* productioncosts) + sum(transport * transportcost)

# Make a network constraint for county ii
function makeconstraint(ii)
    # The constraint function
    function constraint(production, transport)
        transport = [0; transport; 0] # force no transport on boundary
        # Require that production - exports + imports >= requirements
        requirements[ii] - (production[ii] - transport[ii + 1] + transport[ii])
    end
end

# Set up the constraints
constraints = map(makeconstraint, 1:5);
constnames = ASCIIString["E_balance_$ii" for ii in 1:5];

# Define a single point solution
x0 = zeros(9);

# Create the matrices for the linear programming problem
f = objectivevector(objective, x0)
A, b = constraintmatrix(constraints, x0, names=constnames)

# Solve it!
sol = linprog(Vector{Float64}(f), Array{Float64}(A), '<', Vector{Float64}(b), zeros(9), ones(9) * Inf)
sol.sol
