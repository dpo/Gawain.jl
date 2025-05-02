using CUTEst

macro wrappedallocs(expr)
  argnames = [gensym() for a in expr.args]
  quote
    function g($(argnames...))
      @allocated $(Expr(expr.head, argnames...))
    end
    $(Expr(:call, :g, [esc(a) for a in expr.args]...))
  end
end

if Sys.isunix()
  model = CUTEstModel("3PK")
  solver = TRBSolver(model)
  stats = GenericExecutionStats(model, solver_specific = Dict{Symbol,TRB_STATUS}())
  solve!(solver, stats)
  reset!(solver)
  # reset!(model)
  al = @wrappedallocs solve!(solver, stats)
  @test al == 0
  finalize(model)
end
