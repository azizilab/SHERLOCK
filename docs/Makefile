# Minimal makefile for Sphinx documentation.
# Run `make html` from the docs/ directory. Set SPHINXBUILD to point at a
# specific interpreter, e.g. `make html SPHINXBUILD="../env/bin/python -m sphinx"`.

SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help html clean

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)"

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)"

clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)"
