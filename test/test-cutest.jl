using CUTEst

model = CUTEstModel("3PK")
status, x1 = trb(model)
@test status == TRB_STATUS(0)

reset!(model)
status, x2 = trb_hprod(model)
@test status = TRB_STATUS(0)
@test all(x1 .== x2)
