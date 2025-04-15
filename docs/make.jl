using Gawain
using Documenter

DocMeta.setdocmeta!(Gawain, :DocTestSetup, :(using Gawain); recursive = true)

const page_rename = Dict("developer.md" => "Developer docs") # Without the numbers
const numbered_pages = [
  file for file in readdir(joinpath(@__DIR__, "src")) if
  file != "index.md" && splitext(file)[2] == ".md"
]

makedocs(;
  modules = [Gawain],
  authors = "Dominique Orban <dominique.orban@gmail.com>",
  repo = "https://github.com/dpo/Gawain.jl/blob/{commit}{path}#{line}",
  sitename = "Gawain.jl",
  format = Documenter.HTML(; canonical = "https://dpo.github.io/Gawain.jl"),
  pages = ["index.md"; numbered_pages],
)

deploydocs(; repo = "github.com/dpo/Gawain.jl", devbranch = "main", push_preview = true)
