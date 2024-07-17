#!/bin/bash -l

factor_lr=2
steps_lr=2

config=$1
id_name=$2
dataset_dir=$3
results_dir=$4
output_dir=$5


if [ ! -d "$results_dir" ]; then
    mkdir "$results_dir"
fi

if [ ! -d "$results_dir/$id_name" ]; then
    mkdir "$results_dir/$id_name"
fi

if [ ! -d "$output_dir" ]; then
    mkdir "$output_dir"
fi

if [ ! -d "$output_dir/$id_name" ]; then
    mkdir "$output_dir/$id_name"
fi

if [ -e run_job* ]; then
    rm run_job*
fi


for k in 1 2 3 4; do

    if [ -d "$results_dir/$id_name/$k" ]; then
        rm -r "$results_dir/$id_name/$k"
    fi
    mkdir "$results_dir/$id_name/$k"

    if [ -d "$output_dir/$id_name/$k" ]; then
        rm -r "$output_dir/$id_name/$k"
    fi
    mkdir "$output_dir/$id_name/$k"

    start_lr=0.0001
    lr=$start_lr

    for ((i=0;i<$steps_lr+1;i+=1)); do 

        echo $lr

        touch run_job.sh

        echo '#!/bin/bash -l' >> run_job.sh
        echo '#SBATCH --gres=gpu:a100:1' >> run_job.sh
        echo '#SBATCH --time=10:00:00' >> run_job.sh
        echo '#SBATCH --export=NONE' >> run_job.sh
        echo 'unset SLURM_EXPORT_ENV' >> run_job.sh
        echo 'module load python' >> run_job.sh
        echo 'conda activate be' >> run_job.sh


        sed "2a\#SBATCH --output=$output_dir/$id_name/$k/bin_enc_$i.out" run_job.sh > ./run_job_bin_enc.sh
        sed -i "2a\#SBATCH --job-name=$id_name\_$k_$i\_bin_enc" run_job_bin_enc.sh

        sed "2a\#SBATCH --output=$output_dir/$id_name/$k/bin_enc_wide_$i.out" run_job.sh > ./run_job_bin_enc_wide.sh
        sed -i "2a\#SBATCH --job-name=$id_name\_$k_$i\_bin_enc_wide" run_job_bin_enc_wide.sh
        
        sed "2a\#SBATCH --output=$output_dir/$id_name/$k/no_pen_$i.out" run_job.sh > ./run_job_no_pen.sh
        sed -i "2a\#SBATCH --job-name=$id_name\_$k_$i\_no_pen" run_job_no_pen.sh

        sed "2a\#SBATCH --output=$output_dir/$id_name/$k/no_pen_dropout_$i.out" run_job.sh > ./run_job_no_pen_dropout.sh
        sed -i "2a\#SBATCH --job-name=$id_name\_$k_$i\_no_pen_dropout" run_job_no_pen_dropout.sh
        
        sed "2a\#SBATCH --output=$output_dir/$id_name/$k/lin_pen_$i.out" run_job.sh > ./run_job_lin_pen.sh
        sed -i "2a\#SBATCH --job-name=$id_name\_$k_$i\_lin_pen" run_job_lin_pen.sh
        
        sed "2a\#SBATCH --output=$output_dir/$id_name/$k/lin_pen_dropout_$i.out" run_job.sh > ./run_job_lin_pen_dropout.sh
        sed -i "2a\#SBATCH --job-name=$id_name\_$k_$i\_lin_pen_dropout" run_job_lin_pen_dropout.sh

        sed "2a\#SBATCH --output=$output_dir/$id_name/$k/nonlin_pen_$i.out" run_job.sh > ./run_job_nonlin_pen.sh
        sed -i "2a\#SBATCH --job-name=$id_name\_$k_$i\_nonlin_pen" run_job_nonlin_pen.sh


        echo "python scripts/train.py --config  $config  --model-name bin_enc  --dropout-penultimate False --lr $lr  --encoding-metrics True --store-penultimate True --results-dir $results_dir/$id_name/$k --dataset-dir $dataset_dir --sample $i" >> run_job_bin_enc.sh
        echo "python scripts/train.py --config  $config  --model-name bin_enc  --dropout-penultimate False --penultimate-nodes 512 --lr $lr  --encoding-metrics True --store-penultimate False --results-dir $results_dir/$id_name/$k --dataset-dir $dataset_dir --sample $i" >> run_job_bin_enc_wide.sh
        echo "python scripts/train.py --config  $config  --model-name no_pen  --dropout-penultimate False --lr $lr --encoding-metrics True --store-penultimate False --results-dir $results_dir/$id_name/$k --dataset-dir $dataset_dir --sample $i" >> run_job_no_pen.sh
        echo "python scripts/train.py --config  $config  --model-name no_pen  --dropout-penultimate True --lr $lr --encoding-metrics True --store-penultimate False --results-dir $results_dir/$id_name/$k --dataset-dir $dataset_dir --sample $i" >> run_job_no_pen_dropout.sh
        echo "python scripts/train.py --config  $config  --model-name lin_pen  --dropout-penultimate False --lr $lr  --encoding-metrics True --store-penultimate False --results-dir $results_dir/$id_name/$k --dataset-dir $dataset_dir --sample $i" >> run_job_lin_pen.sh
        echo "python scripts/train.py --config  $config  --model-name lin_pen  --dropout-penultimate True --lr $lr  --encoding-metrics True --store-penultimate False --results-dir $results_dir/$id_name/$k --dataset-dir $dataset_dir --sample $i" >> run_job_lin_pen_dropout.sh
        echo "python scripts/train.py --config  $config  --model-name nonlin_pen  --dropout-penultimate False --lr $lr --encoding-metrics True --store-penultimate False --results-dir $results_dir/$id_name/$k --dataset-dir $dataset_dir --sample $i" >> run_job_nonlin_pen.sh

        sbatch ./run_job_bin_enc.sh 
        sbatch ./run_job_bin_enc_wide.sh 
        sbatch ./run_job_no_pen.sh 
        sbatch ./run_job_no_pen_dropout.sh 
        sbatch ./run_job_lin_pen.sh 
        sbatch ./run_job_lin_pen_dropout.sh 
        sbatch ./run_job_nonlin_pen.sh 

        cp $config $results_dir/$id_name/$k/config.yml

        rm run_job*

        tmp_lr=$lr
        lr=$(echo "scale=9; $factor_lr*$tmp_lr" | bc)

    done

done