# Efficient Private Transformer Inference

This directory contains the IEEE Access manuscript migrated from the author's
Overleaf project on 2026-07-17.

## Build

The project is self-contained and includes the IEEE Access class, the IEEEtran
class and bibliography style, required font files, and `spotcolor.sty`.

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error access.tex
```

Clean generated auxiliary files with:

```bash
latexmk -C access.tex
```

The main source is `access.tex`. The section files are kept at the same level
because they are included directly with `\input{...}`.
The committed `access.pdf` is the locally verified IEEE Access preview.

## Manuscript status

The Introduction, Background and Motivation, Related Work, and Method sections
are data-independent drafts. Experimental results, author affiliation cleanup,
funding information, correspondence information, and the final AI-assistance
disclosure remain marked with `\todo{...}` placeholders.

## Experiment environment

- Cluster: three NVIDIA A10 24 GB servers, Ubuntu 22.04, 8 CPU cores and 32 GB RAM per host
- Network: 15 Gbps private network among P0, P1, and P2
- Framework: hpmpc/PIGEON, `PROTOCOL=5` (Trio semi-honest 3PC)
- GPU: `sm_86`, CUDA 12.x, AVX-512 host support
- Recommended build parameters: `DATTYPE=512 PROCESS_NUM=8 SPLITROLES=1 FRACTIONAL=20 USE_CUDA_GEMM=2`
- Target: private BERT-base Softmax, LayerNorm, and FFN inference
