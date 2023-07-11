#!/bin/bash

for expname in lwsw
do
	for splitnum in 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 #16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
	do	
			sbatch --mail-type ALL --mail-user iathin.tam@unil.ch --chdir /work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/TCG_Rad_keras/parallel/ --job-name RADpytorch --output /work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/TCG_Rad_keras/parallel/logs/con-%j.out --error /work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/TCG_Rad_keras/parallel/logs/err-%j.err --partition cpu --nodes 1 --ntasks 1 --cpus-per-task 1 --mem 64G --time 04:30:00 --wrap "module purge; module load gcc; source ~/.bashrc; conda activate fred_workenv; python3 /work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/TCG_Rad_keras/parallel/run_ts.py $expname $splitnum 23"
		done
done

