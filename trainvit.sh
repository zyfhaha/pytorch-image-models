export PYTHONPATH=/mnt/lustre/share/pymc/new:$PYTHONPATH
srun -p caif_dev -n 1 \
    -w SH-IDC1-10-140-1-10 \
    --job-name=test_fsdp --ntasks 8 --ntasks-per-node 8 --cpus-per-task 10 --gres=gpu:8 \
    python -u train_slurm.py  /mnt/lustre/share/images --val-split val -b 64 --model vit_large_patch16_224 --sched cosine \
    --fsdp
    --warmup-epochs 5 \
    --decay-epochs 30 \
    --drop-path 0.3 \
    --smoothing 0.1 \
    --epochs 300 \
    --lr 0.001 \
    --warmup-lr 1.0e-06 \
    --min-lr 1.0e-05 \
    --decay-rate 0.1 \
    --weight-decay 0.05 \
    --clip-grad 5.0 \
    --aa rand-m9-mstd0.5-inc1 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --opt adamw \
    --reprob 0.25 \
    --remode pixel \
    --checkpoint-hist 20 \
    --experiment mp_test \
    --output output/train_policy