# Bare-bones interface to TRB.
# Only Float64 is currently supported.
#
# TODO: Parametrize solver.
# TODO: Define solver object for efficient re-solves.
# TODO: Pass control structure as argument.

export TRB_STATUS
export trb, trb_hprod

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

  trb_import_status = TRB_IMPORT_STATUS(status[])
  if trb_import_error(trb_import_status)
    @error "trb_import exits with status = $trb_import_status"
    trb_terminate(Float64, Int64, data, control, inform)
    # TODO: the function should become type stable when we return proper execution stats
    return trb_import_status, x
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
    if trb_status == TRB_SUCCESS
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

function trb_hprod(
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
    "absent",
    0,
    C_NULL,
    C_NULL,
    C_NULL,
  )

  trb_import_status = TRB_IMPORT_STATUS(status[])
  if trb_import_error(trb_import_status)
    @error "trb_import exits with status = $trb_import_status"
    trb_terminate(Float64, Int64, data, control, inform)
    # TODO: the function should become type stable when we return proper execution stats
    return trb_import_status, x
  end

  v_full = similar(nlp.meta.x0)
  Hv_full = similar(v_full)
  nnz_u = Ref{Int64}()
  nnz_v = Ref{Int64}()

  obj_local = (x, f) -> (f[] = obj(nlp, x); return 0)
  grad_local = (x, g) -> (grad!(nlp, x, g); return 0)
  hprod_update_local = (x, u, v) -> begin
    @debug "computing hprod update"
    hprod!(nlp, x, v, Hv_full)
    u .+= Hv_full
    return 0
  end
  sparse_hprod_local =
    (x, nnz_u, index_nz_u, u, nnz_v, index_nz_v, v) -> begin
      @debug "computing sparse hprod" nnz_v[] index_nz_v v
      # TODO: make this more efficient and include in NLPModels. CUTEstModels can do this natively.
      # turn v into a full-size vector
      v_full .= 0
      @views v_full[index_nz_v[1:nnz_v[]]] .= v[index_nz_v[1:nnz_v[]]]
      sv = Vector(sparsevec(index_nz_v[1:nnz_v[]], v[index_nz_v[1:nnz_v[]]], length(x)))
      @debug "" v_full' sv'
      @assert all(v_full .== sv)
      # perform hprod, store result into full-sized Hv_full
      hprod!(nlp, x, v_full, Hv_full)
      @debug "" Hv_full
      # turn Hv_full into a sparse vector
      k = 0
      for i ∈ eachindex(Hv_full)
        val = Hv_full[i]
        val == 0 && continue
        k = k + 1
        u[i] = val
        index_nz_u[k] = i
      end
      nnz_u[] = k
      @debug "" nnz_u[] index_nz_u u
      @assert all(
        Hv_full .==
        Vector(sparsevec(index_nz_u[1:nnz_u[]], u[index_nz_u[1:nnz_u[]]], length(x))),
      )
      return 0
    end

  eval_status = Ref{Int64}()
  f = Ref{Float64}(0.0)
  g = similar(nlp.meta.x0)
  u = similar(g)
  index_nz_u = Vector{Int64}(UndefInitializer(), n)
  v = similar(g)
  index_nz_v = Vector{Int64}(UndefInitializer(), n)

  finished = false
  while !finished
    trb_solve_reverse_without_mat(
      Float64,
      Int64,
      data,
      status,
      eval_status,
      n,
      x,
      f[],
      g,
      u,
      v,
      index_nz_v,
      nnz_v,
      index_nz_u,
      nnz_u[],
    )
    trb_status = TRB_STATUS(status[])
    if trb_status == TRB_SUCCESS
      finished = true
    elseif trb_error(trb_status)
      @error "trb_solve_reverse_with_mat returns with status = $trb_status"
      finished = true
    elseif trb_status == TRB_EVALUATE_OBJECTIVE
      eval_status[] = obj_local(x, f)
    elseif trb_status == TRB_EVALUATE_GRADIENT
      eval_status[] = grad_local(x, g)
    elseif trb_status == TRB_EVALUATE_HPROD_UPDATE
      eval_status[] = hprod_update_local(x, u, v)
    elseif trb_status == TRB_APPLY_PRECONDITIONER
      eval_status[] = prec(x, u, v)
    elseif trb_status == TRB_EVALUATE_SPARSE_HPROD
      eval_status[] = sparse_hprod_local(x, nnz_u, index_nz_u, u, nnz_v, index_nz_v, v)
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
