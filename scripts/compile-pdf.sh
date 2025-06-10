#!/bin/bash
pandoc paper.md --citeproc --bibliography=paper.bib -s -o paper.pdf -V geometry:margin=1.5in