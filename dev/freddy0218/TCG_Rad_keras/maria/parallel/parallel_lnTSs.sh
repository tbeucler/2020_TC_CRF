#!/bin/bash

for expname in lwsw_drop
do
	for splitnum in 0 1 2
	do	
		for dropout in 0.10 0.15 0.20 0.25 0.30 0.35 0.01 0.05 0.40 0.45 0.50
		do
			sbatch --mail-type ALL --mail-user iathin.tam@unil.ch --chdir /work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/TCG_Rad_keras/maria/parallel/ --job-name RADpytorch_maria --output /work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/TCG_Rad_keras/maria/parallel/logs/con-%j.out --error /work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/TCG_Rad_keras/maria/parallel/logs/err-%j.err --partition cpu --nodes 1 --ntasks 1 --cpus-per-task 1 --mem 64G --time 04:30:00 --wrap "module purge; module load gcc; source ~/.bashrc; conda activate fred_workenv; python3 /work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/TCG_Rad_keras/maria/parallel/run_ts_mcdrop.py $expname $splitnum 23 $dropout"
		done
	done
done

