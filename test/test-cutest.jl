using CUTEst
using SolverCore

model = CUTEstModel("3PK")

# straightfoward test
stats = trb(model)
@test stats.status == :first_order
@test stats.iter == 16

# test with solver object
solver = TRBSolver(model)
stats = GenericExecutionStats(model)
solve!(solver, model, stats)
@test stats.status == :first_order
@test stats.iter == 16

# test resolve
SolverCore.reset!(solver)
SolverCore.reset!(stats)
solve!(solver, model, stats)
@test stats.status == :first_order
@test stats.iter == 16

# test callback
SolverCore.reset!(solver)
SolverCore.reset!(stats)
solve!(
  solver,
  model,
  stats,
  callback = (model, solver, stats) ->
    set_status!(stats, stats.iter == 3 ? :user : :unknown),
)
@test stats.status == :user
@test stats.iter == 3

finalize(model)
