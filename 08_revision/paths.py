"""Data locations used by the scripts in this folder.

Set the environment variables below to point at your own copies; the defaults are
the layout used when the paper was prepared.

  PDROM_DATA_DIR      snapshots_Re*.npy, metadata_*, rollouts, cae_/lstm_ files,
                      seed_sweep_v3_Re500/ (outputs of steps 01-05)
  PDROM_REVISION_DIR  revision snapshots and results (snapshots_Re400cont.npy, ...)
  PDROM_ENSEMBLE_DIR  revision ensembles written by 02_train_rom/seed_sweep_v4.py
                      (S1_fixedCAE/, S3_pod/, ...; see README.md)
  PDROM_CASES_DIR     solved cases of 01_generate_data/circular_cylinder/revision
  PDROM_ARCHIVE_DIR   solved production cases (meshes, for nearwall_resolution_v2.py)
  PDROM_FINE_DIR      solved refined-mesh Re=800/1000 cases (forces, for make_fig_forces.py)
  PDROM_OUT_DIR       where results are written (default: the revision/ensemble
                      directories, i.e. next to their inputs)
"""
import os

def _p(key, default):
    return os.path.expanduser(os.environ.get(key, default))

DATA = _p("PDROM_DATA_DIR", "~/cylinder_lstm_data")
REV = _p("PDROM_REVISION_DIR", os.path.join(DATA, "revision"))
ENS = _p("PDROM_ENSEMBLE_DIR", os.path.join(REV, "B"))
CASES = _p("PDROM_CASES_DIR", "~/OpenFOAM/run/Cylinder2D_study/revision")
ARCHIVE = _p("PDROM_ARCHIVE_DIR", "~/cfd_archive")
FINE = _p("PDROM_FINE_DIR", os.path.join(ARCHIVE, "CYL2D_fine"))
OUT_REV = _p("PDROM_OUT_DIR", REV)
OUT_ENS = _p("PDROM_OUT_DIR", ENS)
