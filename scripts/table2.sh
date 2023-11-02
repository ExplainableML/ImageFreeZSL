### Recreates the ablation results of our ICIS model (Table 2)

### CUB
# Baseline MLP
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=CUB --image_embedding=res101_finetuned --class_embedding=att --single_autoencoder_baseline --num_layers 2 --beta1 0.9 --lr 0.00001 --batch_size 16 --embed_dim 2048 --strict_eval --early_stopping_slope --calc_entropy
# Cosine loss
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=CUB --image_embedding=res101_finetuned --class_embedding=att --single_autoencoder_baseline --cos_sim_loss --num_layers 2 --beta1 0.9 --lr 0.00001 --batch_size 16 --embed_dim 2048 --strict_eval --early_stopping_slope --calc_entropy
# Single-modal
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=CUB --image_embedding=res101_finetuned --class_embedding=att --single_modal_ablation --cos_sim_loss --num_layers 2 --beta1 0.9 --lr 0.00001 --batch_size 16 --embed_dim 2048 --strict_eval --early_stopping_slope --calc_entropy
# Cross-modal
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=CUB --image_embedding=res101_finetuned --class_embedding=att --cos_sim_loss --num_layers 2 --beta1 0.9 --lr 0.00001 --batch_size 16 --embed_dim 2048 --strict_eval --early_stopping_slope --calc_entropy
# Full
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=CUB --image_embedding=res101_finetuned --class_embedding=att --cos_sim_loss --include_unseen --num_layers 2 --beta1 0.9 --lr 0.00001 --batch_size 16 --embed_dim 2048 --strict_eval --early_stopping_slope --calc_entropy

### AWA2
# Baseline MLP
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=AWA2 --image_embedding=res101_finetuned --class_embedding=att --single_autoencoder_baseline --num_layers 2 --beta1 0.9 --lr 0.00001 --batch_size 20 --embed_dim 2048 --strict_eval --early_stopping_slope --calc_entropy
# Cosine loss
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=AWA2 --image_embedding=res101_finetuned --class_embedding=att --single_autoencoder_baseline --cos_sim_loss --num_layers 2 --beta1 0.9 --lr 0.00001 --batch_size 20 --embed_dim 2048 --strict_eval --early_stopping_slope --calc_entropy
# Single-modal
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=AWA2 --image_embedding=res101_finetuned --class_embedding=att --single_modal_ablation --cos_sim_loss --num_layers 2 --beta1 0.9 --lr 0.00001 --batch_size 20 --embed_dim 2048 --strict_eval --early_stopping_slope --calc_entropy
# Cross-modal
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=AWA2 --image_embedding=res101_finetuned --class_embedding=att --cos_sim_loss --num_layers 2 --beta1 0.9 --lr 0.00001 --batch_size 20 --embed_dim 2048 --strict_eval --early_stopping_slope --calc_entropy
# Full
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=AWA2 --image_embedding=res101_finetuned --class_embedding=att --cos_sim_loss --include_unseen --num_layers 2 --beta1 0.9 --lr 0.00001 --batch_size 20 --embed_dim 2048 --strict_eval --early_stopping_slope --calc_entropy

### SUN
# Baseline MLP
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=SUN --image_embedding=res101_finetuned --class_embedding=att --single_autoencoder_baseline --num_layers 2 --beta1 0.9 --lr 0.00001 --batch_size 16 --embed_dim 4096 --strict_eval --early_stopping_slope --calc_entropy
# Cosine loss
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=SUN --image_embedding=res101_finetuned --class_embedding=att --single_autoencoder_baseline --cos_sim_loss --num_layers 2 --beta1 0.9 --lr 0.00001 --batch_size 16 --embed_dim 4096 --strict_eval --early_stopping_slope --calc_entropy
# Single-modal
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=SUN --image_embedding=res101_finetuned --class_embedding=att --single_modal_ablation --cos_sim_loss --num_layers 2 --beta1 0.9 --lr 0.00001 --batch_size 16 --embed_dim 4096 --strict_eval --early_stopping_slope --calc_entropy
# Cross-modal
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=SUN --image_embedding=res101_finetuned --class_embedding=att --cos_sim_loss --num_layers 2 --beta1 0.9 --lr 0.00001 --batch_size 16 --embed_dim 4096 --strict_eval --early_stopping_slope --calc_entropy
# Full
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=SUN --image_embedding=res101_finetuned --class_embedding=att --cos_sim_loss --include_unseen --num_layers 2 --beta1 0.9 --lr 0.00001 --batch_size 16 --embed_dim 4096 --strict_eval --early_stopping_slope --calc_entropy
