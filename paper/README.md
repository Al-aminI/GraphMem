# GraphMem Research Paper

**Title**: GraphMem: Self-Evolving Graph-Based Memory for Production AI Agents

**Author**: Al-Amin Ibrahim

## Compiling the Paper

### Prerequisites

Install a LaTeX distribution:
- **macOS**: `brew install --cask mactex`
- **Ubuntu**: `sudo apt install texlive-full`
- **Windows**: Install MiKTeX or TeX Live

### Build

```bash
cd paper
make
```

This will produce `main.pdf`.

### Clean

```bash
make clean
```

## Using Overleaf

1. Create a new project on [Overleaf](https://www.overleaf.com/)
2. Upload `main.tex` and `references.bib`
3. Compile with pdfLaTeX

## Paper Structure

```
paper/
├── main.tex           # Main LaTeX source
├── references.bib     # BibTeX references
├── Makefile          # Build automation
└── README.md         # This file
```

## Submission Targets

This paper is formatted for submission to:

- **NeurIPS** (Systems for ML track)
- **ICML** (Machine Learning)
- **ACL/EMNLP** (NLP venues)
- **AAAI** (AI applications)

For venue-specific formatting, modify the document class and style files as needed.

## Key Contributions

1. **Hybrid Graph-Vector Architecture**: Combines knowledge graphs with embeddings
2. **Self-Evolution Mechanisms**: Importance scoring, decay, consolidation
3. **Context Engineering**: 99% token reduction through targeted retrieval
4. **Comprehensive Evaluation**: 4.2× speedup, 35% accuracy improvement

## Citation

```bibtex
@software{graphmem2024,
  author = {Ibrahim, Al-Amin},
  title = {GraphMem: Self-Evolving Graph-Based Memory for Production AI Agents},
  year = {2024},
  url = {https://github.com/Al-aminI/GraphMem}
}
```

