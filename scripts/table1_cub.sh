# Table 1, CUB dataset
# Ours
echo "ICIS:"
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=CUB --image_embedding=res101_finetuned --class_embedding=att --cos_sim_loss --include_unseen --num_layers 2 --beta1 0.9 --lr 0.00001 --batch_size 16 --embed_dim 2048 --strict_eval --early_stopping_slope --calc_entropy
# ConSE
echo "ConSE:"
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=CUB --image_embedding=res101_finetuned --class_embedding=att --conse_benchmark
# COSTA
echo "COSTA:"
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=CUB --image_embedding=res101_finetuned --class_embedding=att --costa_benchmark
# Sub. Reg.
echo "Sub. Reg.:"
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=CUB --image_embedding=res101_finetuned --class_embedding=att --single_autoencoder_baseline --num_layers 2 --beta1 0.9 --lr 0.00001 --batch_size 16 --embed_dim 2048 --strict_eval --early_stopping_slope --subspace_proj
# wDAE
echo "wDAE:"
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=CUB --image_embedding=res101_finetuned --class_embedding=att --single_autoencoder_baseline --num_layers 2 --beta1 0.9 --lr 0.00001 --batch_size 16 --embed_dim 2048 --strict_eval --early_stopping_slope --daegnn
# WAvg
echo "WAvg:"
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=CUB --image_embedding=res101_finetuned --vgse_baseline=wavg --class_embedding=att --norm_scale_heuristic
# SMO
echo "SMO:"
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=CUB --image_embedding=res101_finetuned --vgse_baseline=smo --class_embedding=att --vgse_alpha=0 --norm_scale_heuristic