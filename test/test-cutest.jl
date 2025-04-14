using CUTEst

model = CUTEstModel("3PK")
status, x = trb(model)
@test status == 0
