export TRB_STATUS
export TRBSolver
export reset!, solve!, trb

@enum TRB_IMPORT_STATUS begin
  TRB_IMPORT_SUCCESS = 1
  TRB_IMPORT_ALLOC_ERROR = -1
  TRB_IMPORT_DEALLOC_ERROR = -2
  TRB_IMPORT_DIMENSION_ERROR = 3
end

trb_import_error(status::TRB_IMPORT_STATUS) = status != TRB_IMPORT_SUCCESS
trb_import_error(status::Integer) = trb_import_error(TRB_IMPORT_STATUS(status))

@enum TRB_STATUS begin
  TRB_SUCCESS = 0
  TRB_ALLOC_ERROR = -1
  TRB_DEALLOC_ERROR = -2
  TRB_DIMENSION_ERROR = -3
  TRB_OBJECTIVE_UNBOUNDED = -7
  TRB_SYMBOLIC_ANALYSIS_FAILED = -9
  TRB_FACTORIZATION_FAILED = -10
  TRB_SYSTEM_SOLVE_FAILED = -11
  TRB_ILL_CONDITIONED = -16
  TRB_MAXIMUM_ITER = -18
  TRB_MAXIMUM_TIME = -19
  TRB_USER_TERMINATION = -82
  TRB_EVALUATE_OBJECTIVE = 2
  TRB_EVALUATE_GRADIENT = 3
  TRB_EVALUATE_HESSIAN = 4
  TRB_EVALUATE_HPROD_UPDATE = 5  # perform u = u + Hv
  TRB_APPLY_PRECONDITIONER = 6
  TRB_EVALUATE_SPARSE_HPROD = 7
end

trb_error(status::TRB_STATUS) = status < TRB_SUCCESS
trb_error(status::Integer) = trb_error(TRB_STATUS(status))

function get_status(trb_status::TRB_STATUS)
  if trb_status == TRB_SUCCESS
    return :first_order
  elseif trb_status == TRB_OBJECTIVE_UNBOUNDED
    return :unbounded
  elseif trb_status == TRB_MAXIMUM_ITER
    return :max_iter
  elseif trb_status == TRB_MAXIMUM_TIME
    return :max_time
  elseif trb_status == TRB_USER_TERMINATION
    return :user
  elseif trb_status > TRB_SUCCESS  # still performing iterations
    return :unknown
  else
    return :exception
  end
end

"""
    TRBSolver(model)

A structure that contains all required storage to run solver TRB.
A TRBSolver instance can be used to perform efficient re-solves of a problem (e.g., with a different initial guess or different parameter).

### Arguments

- `model::AbstractNLPModel`: an `NLPModel` representing an unconstrained or bound-constrained problem.
"""
mutable struct TRBSolver{T,S,Vi} <:
               AbstractOptimizationSolver where {T,S,Vi<:AbstractVector}
  f::Base.RefValue{T}
  x::S
  gx::S
  hrows::Vi
  hcols::Vi
  hvals::S
  u::S
  v::S
  data::Base.RefValue{Ptr{Cvoid}}
  control::Base.RefValue{trb_control_type{T,Int}}
  status::Base.RefValue{Int}
  inform::Base.RefValue{trb_inform_type{T,Int}}
  inform_status::Base.RefValue{Int}
  eval_status::Base.RefValue{Int}
end

function TRBSolver(model::AbstractNLPModel{T,S}) where {T,S}
  bound_constrained(model) || error("trb does not handle constraints other than bounds")
  f = Ref{T}(zero(T))
  x = similar(model.meta.x0)
  gx = similar(x)
  nnzh = get_nnzh(model)
  hrows, hcols = hess_structure(model)
  hvals = similar(x, nnzh)
  u = similar(x)
  v = similar(x)
  data = Ref{Ptr{Cvoid}}()
  control = Ref{trb_control_type{T,Int}}()
  status = Ref{Int}(0)
  inform = Ref{trb_inform_type{T,Int}}()
  inform_status = Ref{Int}()
  eval_status = Ref{Int}()
  trb_initialize(T, Int, data, control, status)
  @reset control[].f_indexing = true  # Fortran 1-based indexing
  # TODO: figure out the best way for a user to adjust options
  # the following lines allocate
  @reset control[].print_level = 0
  @reset control[].error = 6
  @reset control[].out = 6
  solver = TRBSolver{T,S,typeof(hrows)}(
    f,
    x,
    gx,
    hrows,
    hcols,
    hvals,
    u,
    v,
    data,
    control,
    status,
    inform,
    inform_status,
    eval_status,
  )
  cleanup(s) = trb_terminate(T, Int, s.data, s.control, s.inform)
  finalizer(cleanup, solver)
  solver
end

function SolverCore.reset!(solver::TRBSolver{T,S,Vi}) where {T,S,Vi}
  # reset!(solver.model)
  # FIXME: The following line resets solver.control and deactivates f_indexing
  # trb_initialize(T, Int, solver.data, solver.control, solver.status)
end

"""
    trb(model; kwargs...)

Solve the unconstrained or bound-constrained problem `model` with GALAHAD solver TRB.

### Arguments

- `model::AbstractNLPModel`: an `NLPModel` representing an unconstrained or bound-constrained problem.

### Keyword arguments

- `callback`: a function called at each iteration allowing the user to access intermediate solver information. See below for more details.
- `x0::AbstractVector`: an initial guess of the same type as `model.meta.x0`. Default: `model.meta.x0`.
- `prec`: a function or callable of `(x, u, v)` that overwrites `u` with a preconditioner applied to `v` at the current iterate `x`.
          `prec(x, u, v)` should return `0` on success. Default: u = v, i.e., no preconditioner.
- `print_level::Int`: verbosity level (see the TRB documentation). Default: 1.
- `maxit::Int`: maximum number of iterations. Default: max(50, n), where n is the number of variables.

### Callback

$(Callback_docstring)

If re-solves are of interest, it is more efficient to first instantiate a solver object and call `solve!()` repeatedly:

    solver = TRBSolver(model1)
    stats = GenericExecutionStats(model, solver_specific = Dict{Symbol, TRB_STATUS}())
    solve!(solver, model2, stats; kwargs...)

where the `kwargs...` are the same as above.
In this scenario, `model1` and `model2` need not be the same model, but they must have the same number of variables, and the same Hessian sparsity pattern.
"""
function trb(model::AbstractNLPModel; kwargs...)
  solver = TRBSolver(model)
  stats = GenericExecutionStats(model, solver_specific = Dict{Symbol,TRB_STATUS}())
  solve!(solver, model, stats; kwargs...)
end

function SolverCore.solve!(
  solver::TRBSolver{T,S,Vi},
  model::AbstractNLPModel{T,S},
  stats::GenericExecutionStats;
  callback = (args...) -> nothing,
  x0::AbstractVector{Float64} = model.meta.x0,
  prec::Fprec = (x, u, v) -> (u .= v; return 0),
) where {T,S,Vi,Fprec}

  start_time = time()
  set_status!(stats, :unknown)
  n = get_nvar(model)
  length(x0) == n || error("initial guess has inconsistent size")
  length(solver.x) == n || error("model dimension incompatible with solver")
  x = solver.x .= x0
  reset!(stats)

  nnzh = get_nnzh(model)
  length(solver.hrows) == nnzh ||
    error("number of nonzeros in Hessian incompatible with solver")

  trb_import(
    T,
    Int,
    solver.control,
    solver.data,
    solver.status,
    n,
    model.meta.lvar,
    model.meta.uvar,
    "coordinate",
    nnzh,
    solver.hrows,
    solver.hcols,
    C_NULL,
  )

  trb_import_status = TRB_IMPORT_STATUS(solver.status[])
  if trb_import_error(trb_import_status)
    @error "trb_import exits with status = $trb_import_status"
    set_solver_specific!(stats, :trb_import_status, trb_import_status)
    set_time!(stats, time() - start_time)
    set_solution!(stats, x)
    return stats
  end

  local trb_status::TRB_STATUS
  set_iter!(stats, 0)

  callback(model, solver, stats)
  finished = stats.status != :unknown

  while !finished
    trb_solve_reverse_with_mat(
      T,
      Int,
      solver.data,
      solver.status,
      solver.eval_status,
      n,
      x,
      solver.f[],
      solver.gx,
      nnzh,
      solver.hvals,
      solver.u,
      solver.v,
    )
    trb_status = TRB_STATUS(solver.status[])
    if trb_status == TRB_SUCCESS # successful termination
      finished = true
    elseif trb_error(trb_status)
      @error "trb_solve_reverse_with_mat returns with status = $trb_status"
      finished = true
    elseif trb_status == TRB_EVALUATE_OBJECTIVE
      solver.f[] = obj(model, x)
      solver.eval_status[] = 0
    elseif trb_status == TRB_EVALUATE_GRADIENT
      grad!(model, x, solver.gx)
      solver.eval_status[] = 0
    elseif trb_status == TRB_EVALUATE_HESSIAN
      hess_coord!(model, x, solver.hvals)
      solver.eval_status[] = 0
    elseif trb_status == TRB_APPLY_PRECONDITIONER
      solver.eval_status[] = prec(x, solver.u, solver.v)
    else
      @error "the value $trb_status of status should not occur"
      finished = true
    end

    trb_information(T, Int, solver.data, solver.inform, solver.inform_status)
    set_time!(stats, time() - start_time)
    set_solution!(stats, x)
    set_objective!(stats, solver.f[])
    set_dual_residual!(stats, solver.inform[].norm_pg)
    set_iter!(stats, solver.inform[].iter)
    set_status!(stats, get_status(trb_status))
    set_solver_specific!(stats, :trb_status, trb_status)

    callback(model, solver, stats)
    finished |= stats.status != :unknown
  end

  trb_information(T, Int, solver.data, solver.inform, solver.inform_status)
  set_time!(stats, time() - start_time)
  set_solution!(stats, x)
  set_objective!(stats, solver.f[])
  set_primal_residual!(stats, zero(T))
  set_dual_residual!(stats, solver.inform[].norm_pg)
  set_iter!(stats, solver.inform[].iter)
  if has_bounds(model)
    stats.multipliers_L .= max.(0, solver.gx)
    stats.multipliers_U .= .-min.(0, solver.gx)
    stats.bounds_multipliers_reliable = true
  end
  stats.status == :user || set_status!(stats, get_status(trb_status))
  set_solver_specific!(stats, :trb_status, trb_status)

  stats
end
