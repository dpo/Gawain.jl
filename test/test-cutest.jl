using CUTEst

model = CUTEstModel("3PK")

# straightfoward test
stats = trb(model)
@test stats.status == :first_order

# test with solver object
solver = TRBSolver(model)
stats = GenericExecutionStats(model)
solve!(solver, stats)
@test stats.status == :first_order

# test resolve
reset!(solver)
reset!(stats)
solve!(solver, stats)
@test stats.status == :first_order
