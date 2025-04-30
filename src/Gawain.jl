module Gawain

# stdlib dependencies
using Logging

# external dependencies
using Accessors

# our own dependencies
using GALAHAD
using NLPModels
using SolverCore

const Callback_docstring = "
The callback is called at each iteration.
The expected signature of the callback is `callback(model, solver, stats)`, and its output is ignored.
Changing any of the input arguments will affect the subsequent iterations.
  In particular, setting `stats.status` to anything else than `:unknown` will stop the algorithm, and setting it `:user` will signal that user-requested termination was requested.
All relevant information should be available in `model`, `solver` and `stats`.
Notably, you can access, and modify, the following:
- `solver.x`: current iterate;
- `solver.gx`: current gradient;
- `stats`: structure holding the output of the algorithm (`GenericExecutionStats`), which contains, among other things:
  - `stats.dual_feas`: norm of the residual, for instance, the norm of the gradient for unconstrained problems;
  - `stats.iter`: current iteration counter;
  - `stats.objective`: current objective function value;
  - `stats.status`: current status of the algorithm. Should be `:unknown` unless the algorithm attained a stopping criterion. Changing this to anything will stop the algorithm, but you should use `:user` to properly indicate the intention.
  - `stats.elapsed_time`: elapsed time in seconds.
"

include("trb.jl")

end
