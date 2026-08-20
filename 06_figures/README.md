# 06 — Figures

| script | does |
|---|---|
| `make_fig01.py` | the two flow configurations, with colourbars and a scale bar |
| `make_paper_figures.py` | error spectra, decomposition, latent and field corrections |
| `plot_dmd_baseline.py` | linear-propagator comparison, four panels |
| `sync_figures.py` | regenerates everything, copies into a LaTeX bundle, verifies that every `\includegraphics` resolves |

Run `sync_figures.py --regen` rather than the individual scripts when refreshing a
manuscript. It exists because the figures previously lived in three directories
and drifted apart silently: a script would be corrected, regenerated into one
directory, and the document compiled against another. Every "the fix didn't take"
during preparation was that, not a plotting bug.

Two things in these scripts are deliberate and will look wrong otherwise. The
spectrum figures clip the y-axis at 1e-6, because the small constant added to keep
the DC null finite on a log scale would otherwise stretch the axis over fifteen
empty decades. And the field panel of the DMD figure reads `fieldval_*.npy`, not
the DMD JSON — the JSON's field entry holds the sinusoid baseline under a name
that suggests otherwise.
