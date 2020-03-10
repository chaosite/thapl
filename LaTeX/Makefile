outputs = simple.pdf hanoi.pdf convolution.pdf intersection.pdf garbage_collector.pdf circuit_sim.pdf hannukah.pdf poster.pdf

PDF_VIEWER=evince

all: $(outputs)
.PHONY: all view

$(outputs): %.pdf: %.tex ../Examples/%.thapl
	xelatex -shell-esc $*
	($(PDF_VIEWER) --presentation $*.pdf 2> /dev/null || true) &

view: $(outputs)
	$(PDF_VIEWER) --presentation $^
