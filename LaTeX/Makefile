outputs = inheritance.pdf arrow.pdf arrow_pre.pdf simple.pdf hanoi.pdf projectile.pdf convolution.pdf intersection.pdf garbage_collector.pdf circuit_sim.pdf hannukah.pdf poster.pdf

all: $(outputs)
.PHONY: all

$(outputs): %.pdf: %.tex ../Examples/%.thapl
	xelatex -shell-esc $*
	evince --presentation $*.pdf
