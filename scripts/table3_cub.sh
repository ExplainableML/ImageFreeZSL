# Table 3, CUB dataset
echo "-- Using Wiki2Vec class label embeddings --"
# Ours
echo "ICIS:"
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=CUB --image_embedding=pretrained_resnet101 --class_embedding=wiki2vec --cos_sim_loss --include_unseen --num_layers 2 --beta1 0.9 --lr 0.00001 --batch_size 16 --embed_dim 2048 --strict_eval --early_stopping_slope --calc_entropy --norm_scale_heuristic --zst --zstfrom=imagenet
# ConSE
echo "ConSE:"
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=CUB --image_embedding=pretrained_resnet101 --class_embedding=wiki2vec --conse_benchmark --norm_scale_heuristic --zst --zstfrom=imagenet
# COSTA
echo "COSTA:"
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=CUB --image_embedding=pretrained_resnet101 --class_embedding=wiki2vec --costa_benchmark --norm_scale_heuristic --zst --zstfrom=imagenet
# Sub. Reg.
echo "Sub. Reg.:"
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=CUB --image_embedding=pretrained_resnet101 --class_embedding=wiki2vec --single_autoencoder_baseline --num_layers 2 --beta1 0.9 --lr 0.00001 --batch_size 16 --embed_dim 2048 --strict_eval --early_stopping_slope --subspace_proj --norm_scale_heuristic --zst --zstfrom=imagenet
# wDAE
echo "wDAE:"
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=CUB --image_embedding=pretrained_resnet101 --class_embedding=wiki2vec --single_autoencoder_baseline --num_layers 2 --beta1 0.9 --lr 0.00001 --batch_size 16 --embed_dim 2048 --strict_eval --early_stopping_slope --daegnn --norm_scale_heuristic --zst --zstfrom=imagenet
# WAvg
echo "WAvg:"
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=CUB --image_embedding=pretrained_resnet101 --vgse_baseline=wavg --class_embedding=wiki2vec --norm_scale_heuristic --zst --zstfrom=imagenet
# SMO
echo "SMO:"
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=CUB --image_embedding=pretrained_resnet101 --vgse_baseline=smo --class_embedding=wiki2vec --vgse_alpha=0 --norm_scale_heuristic --zst --zstfrom=imagenet

echo "-- Using ConceptNet class label embeddings --"
# Ours
echo "ICIS:"
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=CUB --image_embedding=pretrained_resnet101 --class_embedding=cn --cos_sim_loss --include_unseen --num_layers 2 --beta1 0.9 --lr 0.00001 --batch_size 16 --embed_dim 2048 --strict_eval --early_stopping_slope --calc_entropy --norm_scale_heuristic  --zst --zstfrom=imagenet
# ConSE
echo "ConSE:"
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=CUB --image_embedding=pretrained_resnet101 --class_embedding=cn --conse_benchmark --norm_scale_heuristic  --zst --zstfrom=imagenet
# COSTA
echo "COSTA:"
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=CUB --image_embedding=pretrained_resnet101 --class_embedding=cn --costa_benchmark --norm_scale_heuristic  --zst --zstfrom=imagenet
# Sub. Reg.
echo "Sub. Reg.:"
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=CUB --image_embedding=pretrained_resnet101 --class_embedding=cn --single_autoencoder_baseline --num_layers 2 --beta1 0.9 --lr 0.00001 --batch_size 16 --embed_dim 2048 --strict_eval --early_stopping_slope --subspace_proj --norm_scale_heuristic  --zst --zstfrom=imagenet
# wDAE
echo "wDAE:"
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=CUB --image_embedding=pretrained_resnet101 --class_embedding=cn --single_autoencoder_baseline --num_layers 2 --beta1 0.9 --lr 0.00001 --batch_size 16 --embed_dim 2048 --strict_eval --early_stopping_slope --daegnn --norm_scale_heuristic  --zst --zstfrom=imagenet
# WAvg
echo "WAvg:"
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=CUB --image_embedding=pretrained_resnet101 --vgse_baseline=wavg --class_embedding=cn --norm_scale_heuristic  --zst --zstfrom=imagenet
# SMO
echo "SMO:"
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=CUB --image_embedding=pretrained_resnet101 --vgse_baseline=smo --class_embedding=cn --vgse_alpha=0 --norm_scale --norm_scale_heuristic  --zst --zstfrom=imagenet

echo "-- Using CLIP class label embeddings --"
# Ours
echo "ICIS:"
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=CUB --image_embedding=pretrained_resnet101 --class_embedding=clip --cos_sim_loss --include_unseen --num_layers 2 --beta1 0.9 --lr 0.00001 --batch_size 16 --embed_dim 2048 --strict_eval --early_stopping_slope --calc_entropy --norm_scale_heuristic  --zst --zstfrom=imagenet
# ConSE
echo "ConSE:"
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=CUB --image_embedding=pretrained_resnet101 --class_embedding=clip --conse_benchmark --norm_scale_heuristic  --zst --zstfrom=imagenet
# COSTA
echo "COSTA:"
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=CUB --image_embedding=pretrained_resnet101 --class_embedding=clip --costa_benchmark --norm_scale_heuristic  --zst --zstfrom=imagenet
# Sub. Reg.
echo "Sub. Reg.:"
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=CUB --image_embedding=pretrained_resnet101 --class_embedding=clip --single_autoencoder_baseline --num_layers 2 --beta1 0.9 --lr 0.00001 --batch_size 16 --embed_dim 2048 --strict_eval --early_stopping_slope --subspace_proj --norm_scale_heuristic  --zst --zstfrom=imagenet
# wDAE
echo "wDAE:"
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=CUB --image_embedding=pretrained_resnet101 --class_embedding=clip --single_autoencoder_baseline --num_layers 2 --beta1 0.9 --lr 0.00001 --batch_size 16 --embed_dim 2048 --strict_eval --early_stopping_slope --daegnn --norm_scale_heuristic  --zst --zstfrom=imagenet
# WAvg
echo "WAvg:"
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=CUB --image_embedding=pretrained_resnet101 --vgse_baseline=wavg --class_embedding=clip --norm_scale_heuristic  --zst --zstfrom=imagenet
# SMO
echo "SMO:"
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --manualSeed 0 --dataset=CUB --image_embedding=pretrained_resnet101 --vgse_baseline=smo --class_embedding=clip --vgse_alpha=0 --norm_scale --norm_scale_heuristic  --zst --zstfrom=imagenet