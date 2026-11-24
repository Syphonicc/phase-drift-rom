#!/bin/bash
# v2 (2026-11-24): case names as stored in this repository (v1 used Re100_main, Re400_cont, Re1000_main,
# Re800_fine, Re1000_fine, Re400_long25D); the refined-mesh cases also receive blockMeshDict.fine.
# Re400_walls10D and forces_only/forces_prod were set up separately (see README.md).
# Revision CFD runs (as used for the paper; the case directories next to this script are its output). Same solver settings/BCs as the production Re=400 case
# (copied from the external disk), differing only in nu, mesh, or time span.
#   Re100     R1   : circular cylinder Re=100, main mesh, 0 -> 250
#   Re400_continued     R4.2 : Re=400 main mesh continued from t=90 -> 290 (long truth)
#   Re1000    R3   : Re=1000 main-mesh baseline for grid convergence, 0 -> 90
#   Re800_finemesh     R3   : Re=800 fine mesh (sqrt2 refinement), 0 -> 90
#   Re1000_finemesh    R3   : Re=1000 fine mesh, 0 -> 90
#   Re400_outlet25D  R4.7 : Re=400, outlet at 25D downstream (same spacing), 0 -> 90

source /usr/lib/openfoam/openfoam2512/etc/bashrc || true
set -e
# SRC : a solved production Re=400 case (../Re400, run to t=90; its polyMesh and its 90/ directory are used)
# FINE: a case holding the refined mesh (../Re400_finemesh after blockMesh with blockMeshDict.fine)
SRC="${PDROM_PRODUCTION_CASE:?set PDROM_PRODUCTION_CASE to the solved production Re=400 case}"
FINE="${PDROM_FINE_CASE:?set PDROM_FINE_CASE to the refined-mesh Re=400 case}"
R=$(cd "$(dirname "$0")" && pwd)

mk() {  # mk name meshsrc nu endTime nproc
    local c=$R/$1
    mkdir -p $c/system $c/constant
    cp $SRC/system/{fvSchemes,fvSolution,mirrorMeshDict,blockMeshDict.main} $c/system/
    cp $SRC/constant/{turbulenceProperties,transportProperties} $c/constant/
    sed -i "s/^nu .*/nu              $3;/" $c/constant/transportProperties
    # controlDict: production settings, fieldAverage removed, gzip output
    python3 - "$SRC/system/controlDict" "$c/system/controlDict" "$4" <<'EOF'
import re,sys
s=open(sys.argv[1]).read()
s=re.sub(r'\n\s*fieldAverage1\s*\{.*?\n    \}\n','\n',s,flags=re.S)
s=re.sub(r'startFrom\s+\w+;','startFrom       latestTime;',s)
s=re.sub(r'endTime\s+[\d.]+;','endTime         %s;'%sys.argv[3],s)
s=re.sub(r'(writeFormat\s+ascii;)',r'\1\nwriteCompression on;',s)
open(sys.argv[2],'w').write(s)
EOF
    cat > $c/system/decomposeParDict <<EOF
FoamFile { version 2.0; format ascii; class dictionary; object decomposeParDict; }
numberOfSubdomains $5;
method scotch;
EOF
    if [ "$2" = copy ]; then cp -r $SRC/constant/polyMesh $c/constant/
    elif [ "$2" = fine ]; then cp -r $FINE/constant/polyMesh $c/constant/; cp $FINE/system/blockMeshDict.fine $c/system/
    fi
    mkdir -p $c/0; cp $SRC/0/U $SRC/0/p $c/0/
}

mk Re100    copy 1.2e-3 250 2
mk Re1000   copy 1.2e-4 90  2
mk Re800_finemesh    fine 1.5e-4 90  3
mk Re1000_finemesh   fine 1.2e-4 90  3
mk Re400_outlet25D none 3.0e-4 90  3
mk Re400_continued    copy 3.0e-4 290 2
# continuation: start from the production t=90 state
rm -rf $R/Re400_continued/0; cp -r $SRC/90 $R/Re400_continued/90
rm -f $R/Re400_continued/90/{UMean,pMean}

# long domain: outlet 18.6667R (9.33D) -> 50R (25D); downstream cells 69 -> 318
# keeps the production streamwise spacing (8.6667R/69 = 40R/318 = 0.1256R)
c=$R/Re400_outlet25D
sed -i -e 's/^xOutlet .*/xOutlet   #eval{ 50.0*$R };/' \
       -e 's/(69 15 \$nz)/(318 15 $nz)/' -e 's/(69 61 \$nz)/(318 61 $nz)/' \
       $c/system/blockMeshDict.main
(cd $c && blockMesh -dict system/blockMeshDict.main > log.blockMesh 2>&1 \
       && mirrorMesh -overwrite > log.mirrorMesh 2>&1 \
       && checkMesh > log.checkMesh 2>&1)

for d in Re100 Re1000 Re800_finemesh Re1000_finemesh Re400_outlet25D Re400_continued; do
    (cd $R/$d && decomposePar -force > log.decomposePar 2>&1)
done
echo setup done
