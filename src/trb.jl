# Bare-bones interface to TRB.
# Only Float64 is currently supported.
#
# TODO: Parametrize solver.
# TODO: Define solver object for efficient re-solves.
# TODO: Pass control structure as argument.

export trb

function trb(
  nlp::AbstractNLPModel{Float64,Vector{Float64}};
  x0 = nlp.meta.x0,
  prec = (x, u, v, _) -> (u .= v; return 0),
  print_level = 1,
  maxit = max(50, nlp.meta.nvar),
)

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
    return status[], x
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
    if status[] == 0 # successful termination
      finished = true
    elseif status[] < 0
      @error "trb_solve_reverse_with_mat returns with status = $(status[])"
      finished = true
    elseif status[] == 2
      eval_status[] = obj_local(x, f)
    elseif status[] == 3
      eval_status[] = grad_local(x, g)
    elseif status[] == 4
      eval_status[] = hess_local(x, hvals)
    elseif status[] == 6
      eval_status[] = prec(x, u, v)
    else
      @error "the value $(status[]) of status should not occur"
      finished = true
    end
  end

  # TODO: convert inform to execution stats
  # trb_information(Float64, Int64, data, inform, status)
  trb_terminate(Float64, Int64, data, control, inform)
  status[], x
end
