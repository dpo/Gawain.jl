# Bare-bones interface to TRB.
# Only Float64 is currently supported.
#
# TODO: Parametrize solver.
# TODO: Define solver object for efficient re-solves.
# TODO: Pass control structure as argument.

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

mutable struct TRBSolver{M, T, S, Fobj, Fgrad, Fhess, Vi} <: AbstractOptimizationSolver where {M <: AbstractNLPModel, T, S, Fobj, Fgrad, Fhess, Vi <: AbstractVector}
  model::M
  obj_local::Fobj
  grad_local::Fgrad
  hess_local::Fhess
  f::Ref{T}
  g::S
  hrows::Vi
  hcols::Vi
  hvals::S
  u::S
  v::S
  data::Ref{Ptr{Cvoid}}
  control::Ref{trb_control_type{T,Int}}
  status::Ref{Int}
  inform::Ref{trb_inform_type{T,Int}}
end

function TRBSolver(model::AbstractNLPModel{T,S}) where {T, S}
  bound_constrained(model) || error("trb does not handle constraints other than bounds")
  obj_local = (x, f) -> (f[] = obj(model, x); return 0)
  grad_local = (x, g) -> (grad!(model, x, g); return 0)
  hess_local = (x, hvals) -> (hess_coord!(model, x, hvals); return 0)
  f = Ref{T}(zero(T))
  g = similar(model.meta.x0)
  nnzh = get_nnzh(model)
  hrows, hcols = hess_structure(model)
  hvals = similar(g, nnzh)
  u = similar(g)
  v = similar(g)
  data = Ref{Ptr{Cvoid}}()
  control = Ref{trb_control_type{T,Int}}()
  status = Ref{Int}(0)
  inform = Ref{trb_inform_type{T,Int}}()
  trb_initialize(T, Int, data, control, status)
  solver = TRBSolver{typeof(model), T, S, typeof(obj_local), typeof(grad_local), typeof(hess_local), typeof(hrows)}(model, obj_local, grad_local, hess_local, f, g, hrows, hcols, hvals, u, v, data, control, status, inform)
  cleanup(s) = trb_terminate(T, Int, s.data, s.control, s.inform)
  finalizer(cleanup, solver)
  solver
end

function SolverCore.reset!(solver::TRBSolver{M, T, S, Fobj, Fgrad, Fhess, Vi}) where {M, T, S, Fobj, Fgrad, Fhess, Vi}
  reset!(solver.model)
  trb_initialize(T, Int, solver.data, solver.control, solver.status)
end

"""
    trb(nlp; kwargs...)

Solve the bound-constrained problem `nlp` with GALAHAD solver TRB.

### Arguments

- `nlp::AbstractNLPModel`: an `NLPModel` representing an unconstrained or bound-constrained problem.

### Keyword arguments

- `x0::AbstractVector`: an initial guess of the same type as `nlp.meta.x0`. Default: `nlp.meta.x0`.
- `prec`: a function or callable of `(x, u, v)` that overwrites `u` with a preconditioner applied to `v` at the current iterate `x`.
          `prec(x, u, v)` should return `0` on success. Default: u = v, i.e., no preconditioner.
- `print_level::Int`: verbosity level (see the TRB documentation). Default: 1.
- `maxit::Int`: maximum number of iterations. Default: max(50, n), where n is the number of variables.
"""
function trb(model::AbstractNLPModel; kwargs...)
  solver = TRBSolver(model)
  solve!(solver; kwargs...)
end

function SolverCore.solve!(
  solver::TRBSolver{M, T, S, Fobj, Fgrad, Fhess, Vi};
  x0::AbstractVector{Float64} = solver.model.meta.x0,
  prec::Fprec = (x, u, v) -> (u .= v; return 0),
  print_level::Int = 1,
  maxit::Int = max(50, solver.model.meta.nvar),
) where {M, T, S, Fobj, Fgrad, Fhess, Vi, Fprec}

  model = solver.model
  n = get_nvar(model)
  length(x0) == n || error("initial guess has inconsistent size")
  x = copy(x0)

  @reset solver.control[].f_indexing = true  # Fortran 1-based indexing
  @reset solver.control[].print_level = print_level
  @reset solver.control[].maxit = maxit

  nnzh = get_nnzh(model)

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
    trb_terminate(T, Int, solver.data, solver.control, solver.inform)
    # TODO: the function should become type stable when we return proper execution stats
    return trb_import_status, x
  end

  eval_status = Ref{Int}()
  finished = false
  while !finished
    trb_solve_reverse_with_mat(
      T,
      Int,
      solver.data,
      solver.status,
      eval_status,
      n,
      x,
      solver.f[],
      solver.g,
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
      eval_status[] = solver.obj_local(x, solver.f)
    elseif trb_status == TRB_EVALUATE_GRADIENT
      eval_status[] = solver.grad_local(x, solver.g)
    elseif trb_status == TRB_EVALUATE_HESSIAN
      eval_status[] = solver.hess_local(x, solver.hvals)
    elseif trb_status == TRB_APPLY_PRECONDITIONER
      eval_status[] = prec(x, solver.u, solver.v)
    else
      @error "the value $trb_status of status should not occur"
      finished = true
    end
  end

  # TODO: convert inform to execution stats
  # trb_information(Float64, Int, data, inform, status)
  TRB_STATUS(solver.status[]), x
end
