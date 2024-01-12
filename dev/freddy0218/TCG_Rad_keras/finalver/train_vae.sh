#!/bin/bash

for expname in vae
do
	for splitnum in 0 1 2 #3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 #0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 17 18 19 20 #16
	do	
		for losscoeff in 0.55 0.5 0.45 0.4  #0.001 0.005 0.01 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 #0 0.001 0.005 #0.01 0.05 0.10 0.15 0.20 0.25 0.30 0.35 #0.01 0.05 0.40 0.45 0.50
		do
			sbatch --mail-type ALL --mail-user iathin.tam@unil.ch --chdir /work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/TCG_Rad_keras/finalver/ --job-name RADpytorch --output /work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/TCG_Rad_keras/finalver/logs/con-%j.out --error /work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/TCG_Rad_keras/finalver/logs/err-%j.err --partition cpu --nodes 1 --ntasks 1 --cpus-per-task 1 --mem 16G --time 04:30:00 --wrap "module purge; module load gcc; source ~/.bashrc; conda activate fred_workenv; python3 /work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/TCG_Rad_keras/finalver/run_vae.py $expname $splitnum 23 $losscoeff"
		done
	done
done
