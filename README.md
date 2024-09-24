

## this can run our model to get latent representation
python /scratch/hajorlou/D-VAE/train.py \
        --data-name final_structures6 \
        --data-type ENAS \
        --save-interval 50 \
        --lr 1e-3 \
        --save-appendix _DCN \
        --epochs 300 \
        --batch-size 128 \
        --model DCN \
        --nz 56 \
        --nvt 6 \
        --readout linear \
        --hs 501 \
        --bidirectional \
        --in_dim 16



## This can train a sgp on the representation to measure performance
python /scratch/hajorlou/D-VAE/bayesian_optimization/bo.py \
  --data-name final_structures6 \
  --save-appendix DCN \
  --checkpoint 200 \
  --res-dir="ENAS_results/" \
  --BO-rounds 10 \
  --BO-batch-size 50 \
  --nz 56
