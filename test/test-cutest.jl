using CUTEst

model = CUTEstModel("3PK")
status, x = trb(model)
@test status == TRB_STATUS(0)
