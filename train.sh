#%%bash
# launch final training with five random seeds for VTAB-dmlab, sun397 and eurosat. The hyperparameters are the same from our paper.
model_root="pre-trained_weights"
data_path="datasets"
output_dir="output"

#### 跑整个数据集
# CUB meta-net
for seed in "42" "44" "82"; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config-file configs/prompt/cub.yaml \
        MODEL.TYPE "vit" MODEL.PROMPT.ADD_INSTANCE_FEATURE "True"\
        DATA.BATCH_SIZE "64" \
        MODEL.PROMPT.NUM_TOKENS "100" \
        MODEL.PROMPT.DEEP "False" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.NAME "CUB" \
        DATA.NUMBER_CLASSES "200" \
        SOLVER.BASE_LR "0.5" \
        SOLVER.WEIGHT_DECAY "0.001" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}/CUB_200_2011" \
        OUTPUT_DIR "${output_dir}/seed_new${seed}"
done

# OxfordFlowers meta-net
#for seed in "42" "44" "82"; do
#    CUDA_VISIBLE_DEVICES=0 python train.py \
#        --config-file configs/prompt/cub.yaml \
#        MODEL.TYPE "vit" MODEL.PROMPT.ADD_INSTANCE_FEATURE "True" \
#        DATA.BATCH_SIZE "64" \
#        MODEL.PROMPT.NUM_TOKENS "100" \
#        MODEL.PROMPT.DEEP "False" \
#        MODEL.PROMPT.DROPOUT "0.1" \
#        DATA.FEATURE "sup_vitb16_imagenet21k" \
#        DATA.NAME "OxfordFlowers" \
#        DATA.NUMBER_CLASSES "102" \
#        SOLVER.BASE_LR "5.0" \
#        SOLVER.WEIGHT_DECAY "0.001" \
#        SEED ${seed} \
#        MODEL.MODEL_ROOT "${model_root}" \
#        DATA.DATAPATH "${data_path}/Oxford_Flowers" \
#        OUTPUT_DIR "${output_dir}/seed_new${seed}"
#done

# StanfordCars meta-net
#for seed in "42" "44" "82"; do
#    CUDA_VISIBLE_DEVICES=0 python train.py \
#        --config-file configs/prompt/cub.yaml \
#        MODEL.TYPE "vit" MODEL.PROMPT.ADD_INSTANCE_FEATURE "True" \
#        DATA.BATCH_SIZE "64" \
#        MODEL.PROMPT.NUM_TOKENS "100" \
#        MODEL.PROMPT.DEEP "False" \
#        MODEL.PROMPT.DROPOUT "0.1" \
#        DATA.FEATURE "sup_vitb16_imagenet21k" \
#        DATA.NAME "StanfordCars" \
#        DATA.NUMBER_CLASSES "196" \
#        SOLVER.BASE_LR "500.0" \
#        SOLVER.WEIGHT_DECAY "0.0" \
#        SEED ${seed} \
#        MODEL.MODEL_ROOT "${model_root}" \
#        DATA.DATAPATH "${data_path}/Stanford_Cars" \
#        OUTPUT_DIR "${output_dir}/seed_new${seed}"
#done

# StanfordDogs meta-net
#for seed in "42" "44" "82"; do
#    CUDA_VISIBLE_DEVICES=0 python train.py \
#        --config-file configs/prompt/cub.yaml \
#        MODEL.TYPE "vit" MODEL.PROMPT.ADD_INSTANCE_FEATURE "True" \
#        DATA.BATCH_SIZE "64" \
#        MODEL.PROMPT.NUM_TOKENS "100" \
#        MODEL.PROMPT.DEEP "False" \
#        MODEL.PROMPT.DROPOUT "0.1" \
#        DATA.FEATURE "sup_vitb16_imagenet21k" \
#        DATA.NAME "StanfordDogs" \
#        DATA.NUMBER_CLASSES "120" \
#        SOLVER.BASE_LR "1.0" \
#        SOLVER.WEIGHT_DECAY "0.0001" \
#        SEED ${seed} \
#        MODEL.MODEL_ROOT "${model_root}" \
#        DATA.DATAPATH "${data_path}/Stanford_Dogs" \
#        OUTPUT_DIR "${output_dir}/seed_new${seed}"
#done

# nabirds meta-net
#for seed in "42" "44" "82"; do
#    CUDA_VISIBLE_DEVICES=0 python train.py \
#        --config-file configs/prompt/cub.yaml \
#        MODEL.TYPE "vit" MODEL.PROMPT.ADD_INSTANCE_FEATURE "True" \
#        DATA.BATCH_SIZE "128" \
#        MODEL.PROMPT.NUM_TOKENS "50" \
#        MODEL.PROMPT.DEEP "False" \
#        MODEL.PROMPT.DROPOUT "0.1" \
#        DATA.FEATURE "sup_vitb16_imagenet21k" \
#        DATA.NAME "nabirds" \
#        DATA.NUMBER_CLASSES "555" \
#        SOLVER.BASE_LR "10.0" \
#        SOLVER.WEIGHT_DECAY "0.0" \
#        SEED ${seed} \
#        MODEL.MODEL_ROOT "${model_root}" \
#        DATA.DATAPATH "${data_path}/nabirds" \
#        OUTPUT_DIR "${output_dir}/seed_new${seed}"
#done

# oxford_iiit_pet meta-net
#for seed in "42" "44" "82"; do
#    CUDA_VISIBLE_DEVICES=0 python train.py \
#        --config-file configs/prompt/cub.yaml \
#        MODEL.TYPE "vit" \
#        DATA.BATCH_SIZE "128" MODEL.PROMPT.ADD_INSTANCE_FEATURE "True"\
#        MODEL.PROMPT.NUM_TOKENS "50" \
#        MODEL.PROMPT.DEEP "False" \
#        MODEL.PROMPT.DROPOUT "0.1" \
#        DATA.FEATURE "sup_vitb16_imagenet21k" \
#        DATA.NAME "vtab-oxford_iiit_pet" \
#        DATA.NUMBER_CLASSES "37" \
#        SOLVER.BASE_LR "2.5" \
#        SOLVER.WEIGHT_DECAY "0.001" \
#        SEED ${seed} \
#        MODEL.MODEL_ROOT "${model_root}" \
#        DATA.DATAPATH "${data_path}/vtab_data" \
#        OUTPUT_DIR "${output_dir}/seed_new${seed}"
#done

# CUB original
#for seed in "42" "44" "82" "100" "800"; do
#    CUDA_VISIBLE_DEVICES=0 python train.py \
#        --config-file configs/prompt/cub.yaml \
#        MODEL.TYPE "vit" \
#        DATA.BATCH_SIZE "64" \
#        MODEL.PROMPT.NUM_TOKENS "100" \
#        MODEL.PROMPT.DEEP "False" \
#        MODEL.PROMPT.DROPOUT "0.1" \
#        DATA.FEATURE "sup_vitb16_imagenet21k" \
#        DATA.NAME "CUB" \
#        DATA.NUMBER_CLASSES "200" \
#        SOLVER.BASE_LR "0.5" \
#        SOLVER.WEIGHT_DECAY "0.001" \
#        SEED ${seed} \
#        MODEL.MODEL_ROOT "${model_root}" \
#        DATA.DATAPATH "${data_path}/CUB_200_2011" \
#        OUTPUT_DIR "${output_dir}/seed${seed}"
#done

## OxfordFlowers original
#for seed in "42" "44" "82" "100" "800"; do
#    CUDA_VISIBLE_DEVICES=0 python train.py \
#        --config-file configs/prompt/cub.yaml \
#        MODEL.TYPE "vit" \
#        DATA.BATCH_SIZE "64" \
#        MODEL.PROMPT.NUM_TOKENS "100" \
#        MODEL.PROMPT.DEEP "False" \
#        MODEL.PROMPT.DROPOUT "0.1" \
#        DATA.FEATURE "sup_vitb16_imagenet21k" \
#        DATA.NAME "OxfordFlowers" \
#        DATA.NUMBER_CLASSES "102" \
#        SOLVER.BASE_LR "5.0" \
#        SOLVER.WEIGHT_DECAY "0.001" \
#        SEED ${seed} \
#        MODEL.MODEL_ROOT "${model_root}" \
#        DATA.DATAPATH "${data_path}/Oxford_Flowers" \
#        OUTPUT_DIR "${output_dir}/seed${seed}"
#done

# StanfordCars original
#for seed in "42" "44" "82"; do
#    CUDA_VISIBLE_DEVICES=0 python train.py \
#        --config-file configs/prompt/cub.yaml \
#        MODEL.TYPE "vit" \
#        DATA.BATCH_SIZE "64" \
#        MODEL.PROMPT.NUM_TOKENS "100" \
#        MODEL.PROMPT.DEEP "False" \
#        MODEL.PROMPT.DROPOUT "0.1" \
#        DATA.FEATURE "sup_vitb16_imagenet21k" \
#        DATA.NAME "StanfordCars" \
#        DATA.NUMBER_CLASSES "196" \
#        SOLVER.BASE_LR "500.0" \
#        SOLVER.WEIGHT_DECAY "0.0" \
#        SEED ${seed} \
#        MODEL.MODEL_ROOT "${model_root}" \
#        DATA.DATAPATH "${data_path}/Stanford_Cars" \
#        OUTPUT_DIR "${output_dir}/seed${seed}"
#done

# StanfordDogs original
#for seed in "42" "44" "82"; do
#    CUDA_VISIBLE_DEVICES=0 python train.py \
#        --config-file configs/prompt/cub.yaml \
#        MODEL.TYPE "vit" \
#        DATA.BATCH_SIZE "64" \
#        MODEL.PROMPT.NUM_TOKENS "100" \
#        MODEL.PROMPT.DEEP "False" \
#        MODEL.PROMPT.DROPOUT "0.1" \
#        DATA.FEATURE "sup_vitb16_imagenet21k" \
#        DATA.NAME "StanfordDogs" \
#        DATA.NUMBER_CLASSES "120" \
#        SOLVER.BASE_LR "1.0" \
#        SOLVER.WEIGHT_DECAY "0.0001" \
#        SEED ${seed} \
#        MODEL.MODEL_ROOT "${model_root}" \
#        DATA.DATAPATH "${data_path}/Stanford_Dogs" \
#        OUTPUT_DIR "${output_dir}/seed${seed}"
#done

# nabirds original
#for seed in "42" "44" "82"; do
#    CUDA_VISIBLE_DEVICES=0 python train.py \
#        --config-file configs/prompt/cub.yaml \
#        MODEL.TYPE "vit" \
#        DATA.BATCH_SIZE "128" \
#        MODEL.PROMPT.NUM_TOKENS "50" \
#        MODEL.PROMPT.DEEP "False" \
#        MODEL.PROMPT.DROPOUT "0.1" \
#        DATA.FEATURE "sup_vitb16_imagenet21k" \
#        DATA.NAME "nabirds" \
#        DATA.NUMBER_CLASSES "555" \
#        SOLVER.BASE_LR "10.0" \
#        SOLVER.WEIGHT_DECAY "0.0" \
#        SEED ${seed} \
#        MODEL.MODEL_ROOT "${model_root}" \
#        DATA.DATAPATH "${data_path}/nabirds" \
#        OUTPUT_DIR "${output_dir}/seed${seed}"
#done

# oxford_iiit_pet original
#for seed in "42" "44" "82" "100" "800"; do
#    CUDA_VISIBLE_DEVICES=0 python train.py \
#        --config-file configs/prompt/cub.yaml \
#        MODEL.TYPE "vit" \
#        DATA.BATCH_SIZE "128" \
#        MODEL.PROMPT.NUM_TOKENS "50" \
#        MODEL.PROMPT.DEEP "False" \
#        MODEL.PROMPT.DROPOUT "0.1" \
#        DATA.FEATURE "sup_vitb16_imagenet21k" \
#        DATA.NAME "vtab-oxford_iiit_pet" \
#        DATA.NUMBER_CLASSES "37" \
#        SOLVER.BASE_LR "2.5" \
#        SOLVER.WEIGHT_DECAY "0.001" \
#        SEED ${seed} \
#        MODEL.MODEL_ROOT "${model_root}" \
#        DATA.DATAPATH "${data_path}" \
#        OUTPUT_DIR "${output_dir}/seed${seed}"
#done

# 例子
# vtab-structured: dmlab
# base_lr = 1.0
# lr = base_lr / 256 * cfg.DATA.BATCH_SIZE
#for seed in "42" "44" "82" "100" "800"; do
#    python train.py \
#        --config-file configs/prompt/cub.yaml \
#        MODEL.TYPE "vit" \
#        DATA.BATCH_SIZE "64" \
#        MODEL.PROMPT.NUM_TOKENS "100" \
#        MODEL.PROMPT.DEEP "True" \
#        MODEL.PROMPT.DROPOUT "0.1" \
#        DATA.FEATURE "sup_vitb16_imagenet21k" \
#        DATA.NAME "vtab-dmlab" \
#        DATA.NUMBER_CLASSES "6" \
#        SOLVER.BASE_LR "0.25" \
#        SOLVER.WEIGHT_DECAY "0.001" \
#        SEED ${seed} \
#        MODEL.MODEL_ROOT "${model_root}" \
#        DATA.DATAPATH "${data_path}" \
#        OUTPUT_DIR "${output_dir}/seed${seed}"
#done

# vtab-natural: sun397
# base_lr = 25
# lr = base_lr / 256 * cfg.DATA.BATCH_SIZE
#for seed in "42" "44" "82" "100" "800"; do
#    python train.py \
#        --config-file configs/prompt/cub.yaml \
#        MODEL.TYPE "vit" \
#        DATA.BATCH_SIZE "128" \
#        MODEL.PROMPT.NUM_TOKENS "5" \
#        MODEL.PROMPT.DEEP "True" \
#        MODEL.PROMPT.DROPOUT "0.1" \
#        DATA.FEATURE "sup_vitb16_imagenet21k" \
#        DATA.NAME "vtab-sun397" \
#        DATA.NUMBER_CLASSES "397" \
#        SOLVER.BASE_LR "12.5" \
#        SOLVER.WEIGHT_DECAY "0.0001" \
#        SOLVER.TOTAL_EPOCH "100" \
#        SEED ${seed} \
#        MODEL.MODEL_ROOT "${model_root}" \
#        DATA.DATAPATH "${data_path}" \
#        OUTPUT_DIR "${output_dir}/seed${seed}"
#done

# vtab-specialized: vtab-eurosat
# base_lr = 1
# lr = base_lr / 256 * cfg.DATA.BATCH_SIZE
#for seed in "42" "44" "82" "100" "800"; do
#    python train.py \
#        --config-file configs/prompt/cub.yaml \
#        MODEL.TYPE "vit" \
#        DATA.BATCH_SIZE "64" \
#        MODEL.PROMPT.NUM_TOKENS "100" \
#        MODEL.PROMPT.DEEP "True" \
#        MODEL.PROMPT.DROPOUT "0.1" \
#        DATA.FEATURE "sup_vitb16_imagenet21k" \
#        DATA.NAME "vtab-eurosat" \
#        DATA.NUMBER_CLASSES "10" \
#        SOLVER.BASE_LR "0.25" \
#        SOLVER.WEIGHT_DECAY "0.001" \
#        SOLVER.TOTAL_EPOCH "100" \
#        SEED ${seed} \
#        MODEL.MODEL_ROOT "${model_root}" \
#        DATA.DATAPATH "${data_path}" \
#        OUTPUT_DIR "${output_dir}/seed${seed}"
#done