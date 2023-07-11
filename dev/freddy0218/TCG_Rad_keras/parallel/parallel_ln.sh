#!/bin/bash

for expname in lwswthhdiau lwswthhdiav lwswthhdiaw #lwswthhdia #lwswthw lwswthhdia
do 
	sbatch --mail-type ALL --mail-user iathin.tam@unil.ch --chdir /work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/TCG_Rad_keras/parallel/ --job-name RADpytorch --output /work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/TCG_Rad_keras/parallel/logs/con-%j.out --error /work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/TCG_Rad_keras/parallel/logs/err-%j.err --partition cpu --nodes 1 --ntasks 1 --cpus-per-task 1 --mem 64G --time 04:30:00 --wrap "module purge; module load gcc; source ~/.bashrc; python /work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/TCG_Rad_keras/parallel/run_linear.py $expname 23"
done	
