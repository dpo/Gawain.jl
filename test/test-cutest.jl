using CUTEst

model = CUTEstModel("3PK")
solver = TRBSolver(model)
status, x = trb(solver)
@test status == TRB_STATUS(0)
