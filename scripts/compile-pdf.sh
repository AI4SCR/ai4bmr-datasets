#!/bin/bash
pandoc paper.md --citeproc --bibliography=paper.bib -s -o paper.pdf -V geometry:margin=1.5in
pandoc paper-arxiv.md --citeproc --bibliography=paper.bib -s -o paper-arxiv.pdf --template=arxiv.tex -V geometry:margin=1.5in
#pandoc paper-arxiv.md --bibliography=paper.bib -s -o paper-arxiv.tex --template=default.tex -V geometry:margin=1.5in --citeproc
