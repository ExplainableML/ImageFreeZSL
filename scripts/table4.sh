echo "Sub. Reg., CUB:"
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=CUB --image_embedding=res101_finetuned --class_embedding=att --cos_sim_loss --include_unseen --num_layers 2 --beta1 0.9 --lr 0.00001 --batch_size 16 --embed_dim 2048 --strict_eval --early_stopping_slope --subspace_proj

echo "Sub. Reg. + ICIS, CUB:"
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=CUB --image_embedding=res101_finetuned --class_embedding=att --cos_sim_loss --include_unseen --num_layers 2 --beta1 0.9 --lr 0.00001 --batch_size 16 --embed_dim 2048 --strict_eval --early_stopping_slope --daegnn
