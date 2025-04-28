using CUTEst

model = CUTEstModel("3PK")

# straightfoward test
status, x = trb(model)
@test status == TRB_STATUS(0)

# test with solver object
solver = TRBSolver(model)
status, x = solve!(solver)
@test status == TRB_STATUS(0)

# test resolve
reset!(solver)
status, x = solve!(solver)
@test status == TRB_STATUS(0)
