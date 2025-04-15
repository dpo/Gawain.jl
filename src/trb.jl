# Bare-bones interface to TRB.
# Only Float64 is currently supported.
#
# TODO: Parametrize solver.
# TODO: Define solver object for efficient re-solves.
# TODO: Pass control structure as argument.

export TRB_STATUS
export trb

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
  TRB_EVALUATE_HPROD_ADD = 5  # perform u = u + Hv
  TRB_APPLY_PRECONDITIONER = 6
  TRB_APPLY_SPARSE_HPROD = 7
end

trb_error(status::TRB_STATUS) = Int(status) < 0
trb_error(status::Integer) = trb_error(TRB_STATUS(status))

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
function trb(
  nlp::AbstractNLPModel{Float64,Vector{Float64}};
  x0::AbstractVector{Float64} = nlp.meta.x0,
  prec::F = (x, u, v) -> (u .= v; return 0),
  print_level::Int = 1,
  maxit::Int = max(50, nlp.meta.nvar),
) where {F}

  bound_constrained(nlp) || error("trb does not handle constraints other than bounds")
  n = get_nvar(nlp)
  length(x0) == n || error("initial guess has inconsistent size")

  data = Ref{Ptr{Cvoid}}()
  control = Ref{trb_control_type{Float64,Int64}}()
  status = Ref{Int64}(0)
  inform = Ref{trb_inform_type{Float64,Int64}}()
  trb_initialize(Float64, Int64, data, control, status)

  @reset control[].f_indexing = true  # Fortran 1-based indexing
  @reset control[].print_level = print_level
  @reset control[].maxit = maxit

  nnzh = get_nnzh(nlp)
  hrows, hcols = hess_structure(nlp)
  x = copy(x0)

  trb_import(
    Float64,
    Int64,
    control,
    data,
    status,
    n,
    nlp.meta.lvar,
    nlp.meta.uvar,
    "coordinate",
    nnzh,
    hrows,
    hcols,
    C_NULL,
  )

  if status[] != 1
    @error "trb_import exits with status = $(status[])"
    trb_terminate(Float64, Int64, data, control, inform)
    return TRB_STATUS(status[]), x
  end

  obj_local = (x, f) -> (f[] = obj(nlp, x); return 0)
  grad_local = (x, g) -> (grad!(nlp, x, g); return 0)
  hess_local = (x, hvals) -> (hess_coord!(nlp, x, hvals); return 0)

  eval_status = Ref{Int64}()
  f = Ref{Float64}(0.0)
  g = similar(nlp.meta.x0)
  hvals = similar(g, nnzh)
  u = similar(g)
  v = similar(g)

  finished = false
  while !finished
    trb_solve_reverse_with_mat(
      Float64,
      Int64,
      data,
      status,
      eval_status,
      n,
      x,
      f[],
      g,
      nnzh,
      hvals,
      u,
      v,
    )
    trb_status = TRB_STATUS(status[])
    if trb_status == TRB_SUCCESS # successful termination
      finished = true
    elseif trb_error(trb_status)
      @error "trb_solve_reverse_with_mat returns with status = $trb_status"
      finished = true
    elseif trb_status == TRB_EVALUATE_OBJECTIVE
      eval_status[] = obj_local(x, f)
    elseif trb_status == TRB_EVALUATE_GRADIENT
      eval_status[] = grad_local(x, g)
    elseif trb_status == TRB_EVALUATE_HESSIAN
      eval_status[] = hess_local(x, hvals)
    elseif trb_status == TRB_APPLY_PRECONDITIONER
      eval_status[] = prec(x, u, v)
    else
      @error "the value $trb_status of status should not occur"
      finished = true
    end
  end

  # TODO: convert inform to execution stats
  # trb_information(Float64, Int64, data, inform, status)
  trb_terminate(Float64, Int64, data, control, inform)
  TRB_STATUS(status[]), x
end
