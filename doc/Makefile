LOCAL_ROOT = ../../..
include ${LOCAL_ROOT}/include/Makefile.standard

Specification: Specification.dvi

TEXS := Specification.tex intro.tex primary.tex secondary.tex summary.tex
FIGS := layout.ps

# date is there to stop make from caring that bibtex returns an error
# code due to mutiply defined entries
Specification.dvi: $(TEXS) $(FIGS)
	latex Specification
	latex Specification

# convert xfig file format (.fig) to Postscript (.ps)
%.ps: %.fig
	fig2dev -L ps $< > $@
