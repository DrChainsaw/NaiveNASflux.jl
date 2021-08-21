using Documenter, Literate, NaiveNASflux

const nndir = joinpath(dirname(pathof(NaiveNASflux)), "..")

function literate_example(sourcefile; rootdir=nndir, sourcedir = "test/examples", destdir="docs/src/examples")
    fullpath = Literate.markdown(joinpath(rootdir, sourcedir, sourcefile), joinpath(rootdir, destdir); flavor=Literate.DocumenterFlavor(), mdstrings=true, codefence="````julia" => "````")
    dirs = splitpath(fullpath)
    srcind = findfirst(==("src"), dirs)
    joinpath(dirs[srcind+1:end]...)
end

quicktutorial = literate_example("quicktutorial.jl")
xorpruning = literate_example("xorpruning.jl")

makedocs(   sitename="NaiveNASflux",
            root = joinpath(nndir, "docs"), 
            format = Documenter.HTML(
                prettyurls = get(ENV, "CI", nothing) == "true"
            ),
            pages = [
                "index.md",
                quicktutorial,
                xorpruning,
                "API Reference" => [
                    "reference/createvertex.md",
                    "reference/layerwrappers.md",
                    "reference/misc.md"
                ]
            ],
            modules = [NaiveNASflux],
        )

function touchfile(filename, rootdir=nndir, destdir="test/examples")
    filepath = joinpath(rootdir, destdir, filename)
    isfile(filepath) && return
    write(filepath, """
    md\"\"\"
    # Markdown header
    \"\"\"
    """)
end

if get(ENV, "CI", nothing) == "true"
    deploydocs(
        repo = "github.com/DrChainsaw/NaiveNASflux.jl.git",
        push_preview=true
    )
end