using CUTEst

model = CUTEstModel("3PK")

# straightfoward test
stats = trb(model)
@test stats.status == :first_order
@test stats.iter == 5

# test with solver object
solver = TRBSolver(model)
stats = GenericExecutionStats(model)
solve!(solver, stats)
@test stats.status == :first_order
@test stats.iter == 5

# test resolve
reset!(solver)
reset!(stats)
solve!(solver, stats)
@test stats.status == :first_order
@test stats.iter == 5

# test callback
reset!(solver)
reset!(stats)
solve!(
  solver,
  stats,
  callback = (model, solver, stats) ->
    set_status!(stats, stats.iter == 3 ? :user : :unknown),
)
@test stats.status == :user
@test stats.iter == 3
