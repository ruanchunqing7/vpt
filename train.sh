#%%bash
# launch final training with five random seeds for VTAB-dmlab, sun397 and eurosat. The hyperparameters are the same from our paper.
model_root="pre-trained_weights"
data_path="datasets"
output_dir="output"

## caltech101 meta-net 数据集还没下载下来
#for seed in "42"; do
#    CUDA_VISIBLE_DEVICES=0 python train.py \
#        --config-file configs/prompt/cub.yaml \
#        MODEL.TYPE "vit" \
#        DATA.BATCH_SIZE "128" MODEL.PROMPT.ADD_INSTANCE_FEATURE "True"\
#        MODEL.PROMPT.NUM_TOKENS "5" \
#        MODEL.PROMPT.DEEP "True" \
#        MODEL.PROMPT.DROPOUT "0.1" \
#        DATA.FEATURE "sup_vitb16_imagenet21k" \
#        DATA.NAME "vtab-caltech101" \
#        DATA.NUMBER_CLASSES "102" \
#        SOLVER.BASE_LR "2.5" \
#        SOLVER.WEIGHT_DECAY "0.001" \
#        SEED ${seed} \
#        MODEL.MODEL_ROOT "${model_root}" \
#        DATA.DATAPATH "${data_path}/vtab_data" \
#        OUTPUT_DIR "${output_dir}/seed_new_deep${seed}"
#done

## caltech101 original 数据集还没下载下来
#for seed in "42"; do
#    CUDA_VISIBLE_DEVICES=0 python train.py \
#        --config-file configs/prompt/cub.yaml \
#        MODEL.TYPE "vit" \
#        DATA.BATCH_SIZE "128" \
#        MODEL.PROMPT.NUM_TOKENS "5" \
#        MODEL.PROMPT.DEEP "True" \
#        MODEL.PROMPT.DROPOUT "0.1" \
#        DATA.FEATURE "sup_vitb16_imagenet21k" \
#        DATA.NAME "vtab-caltech101" \
#        DATA.NUMBER_CLASSES "102" \
#        SOLVER.BASE_LR "2.5" \
#        SOLVER.WEIGHT_DECAY "0.001" \
#        SEED ${seed} \
#        MODEL.MODEL_ROOT "${model_root}" \
#        DATA.DATAPATH "${data_path}/vtab_data" \
#        OUTPUT_DIR "${output_dir}/seed_old_deep${seed}"
#done

## cifar(num_classes=100) meta-net
#for seed in "42"; do
#    CUDA_VISIBLE_DEVICES=0 python train.py \
#        --config-file configs/prompt/cub.yaml \
#        MODEL.TYPE "vit" \
#        DATA.BATCH_SIZE "64" MODEL.PROMPT.ADD_INSTANCE_FEATURE "True"\
#        MODEL.PROMPT.NUM_TOKENS "100" \
#        MODEL.PROMPT.DEEP "True" \
#        MODEL.PROMPT.DROPOUT "0.1" \
#        DATA.FEATURE "sup_vitb16_imagenet21k" \
#        DATA.NAME "vtab-cifar(num_classes=100)" \
#        DATA.NUMBER_CLASSES "100" \
#        SOLVER.BASE_LR "10.0" \
#        SOLVER.WEIGHT_DECAY "0.001" \
#        SEED ${seed} \
#        MODEL.MODEL_ROOT "${model_root}" \
#        DATA.DATAPATH "${data_path}/vtab_data" \
#        OUTPUT_DIR "${output_dir}/seed_new_deep${seed}"
#done

# cifar(num_classes=100) original
for seed in "42"; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config-file configs/prompt/cub.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "64" \
        MODEL.PROMPT.NUM_TOKENS "100" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.NAME "vtab-cifar(num_classes=100)" \
        DATA.NUMBER_CLASSES "100" \
        SOLVER.BASE_LR "10.0" \
        SOLVER.WEIGHT_DECAY "0.001" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}/vtab_data" \
        OUTPUT_DIR "${output_dir}/seed_old_deep${seed}"
done

# dtd meta-net
for seed in "42"; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config-file configs/prompt/cub.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "128" MODEL.PROMPT.ADD_INSTANCE_FEATURE "True"\
        MODEL.PROMPT.NUM_TOKENS "1" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.NAME "vtab-dtd" \
        DATA.NUMBER_CLASSES "47" \
        SOLVER.BASE_LR "10.0" \
        SOLVER.WEIGHT_DECAY "0.0" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}/vtab_data" \
        OUTPUT_DIR "${output_dir}/seed_new_deep${seed}"
done

# dtd original
for seed in "42"; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config-file configs/prompt/cub.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "128" \
        MODEL.PROMPT.NUM_TOKENS "1" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.NAME "vtab-dtd" \
        DATA.NUMBER_CLASSES "47" \
        SOLVER.BASE_LR "10.0" \
        SOLVER.WEIGHT_DECAY "0.0" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}/vtab_data" \
        OUTPUT_DIR "${output_dir}/seed_old_deep${seed}"
done

# oxford_flowers102 meta-net
for seed in "42"; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config-file configs/prompt/cub.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "64" MODEL.PROMPT.ADD_INSTANCE_FEATURE "True"\
        MODEL.PROMPT.NUM_TOKENS "200" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.NAME 'vtab-oxford_flowers102' \
        DATA.NUMBER_CLASSES "102" \
        SOLVER.BASE_LR "25.0" \
        SOLVER.WEIGHT_DECAY "0.001" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}/vtab_data" \
        OUTPUT_DIR "${output_dir}/seed_new_deep${seed}"
done

# oxford_flowers102 original
for seed in "42"; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config-file configs/prompt/cub.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "64" \
        MODEL.PROMPT.NUM_TOKENS "200" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.NAME 'vtab-oxford_flowers102' \
        DATA.NUMBER_CLASSES "102" \
        SOLVER.BASE_LR "25.0" \
        SOLVER.WEIGHT_DECAY "0.001" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}/vtab_data" \
        OUTPUT_DIR "${output_dir}/seed_old_deep${seed}"
done

# oxford_iiit_pet meta-net
for seed in "42"; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config-file configs/prompt/cub.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "128" MODEL.PROMPT.ADD_INSTANCE_FEATURE "True"\
        MODEL.PROMPT.NUM_TOKENS "50" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.NAME "vtab-oxford_iiit_pet" \
        DATA.NUMBER_CLASSES "37" \
        SOLVER.BASE_LR "2.5" \
        SOLVER.WEIGHT_DECAY "0.001" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}/vtab_data" \
        OUTPUT_DIR "${output_dir}/seed_new_deep${seed}"
done

# oxford_iiit_pet original
for seed in "42"; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config-file configs/prompt/cub.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "128" \
        MODEL.PROMPT.NUM_TOKENS "50" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.NAME "vtab-oxford_iiit_pet" \
        DATA.NUMBER_CLASSES "37" \
        SOLVER.BASE_LR "2.5" \
        SOLVER.WEIGHT_DECAY "0.001" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}/vtab_data" \
        OUTPUT_DIR "${output_dir}/seed_old_deep${seed}"
done

## sun397 meta-net 数据集还没下载下来
#for seed in "42"; do
#    CUDA_VISIBLE_DEVICES=0 python train.py \
#        --config-file configs/prompt/cub.yaml \
#        MODEL.TYPE "vit" \
#        DATA.BATCH_SIZE "128" MODEL.PROMPT.ADD_INSTANCE_FEATURE "True"\
#        MODEL.PROMPT.NUM_TOKENS "1" \
#        MODEL.PROMPT.DEEP "True" \
#        MODEL.PROMPT.DROPOUT "0.1" \
#        DATA.FEATURE "sup_vitb16_imagenet21k" \
#        DATA.NAME 'vtab-sun397' \
#        DATA.NUMBER_CLASSES "397" \
#        SOLVER.BASE_LR "1.0" \
#        SOLVER.WEIGHT_DECAY "0.0" \
#        SEED ${seed} \
#        MODEL.MODEL_ROOT "${model_root}" \
#        DATA.DATAPATH "${data_path}/vtab_data" \
#        OUTPUT_DIR "${output_dir}/seed_new_deep${seed}"
#done

## sun397 original 数据集还没下载下来
#for seed in "42"; do
#    CUDA_VISIBLE_DEVICES=0 python train.py \
#        --config-file configs/prompt/cub.yaml \
#        MODEL.TYPE "vit" \
#        DATA.BATCH_SIZE "128" \
#        MODEL.PROMPT.NUM_TOKENS "1" \
#        MODEL.PROMPT.DEEP "True" \
#        MODEL.PROMPT.DROPOUT "0.1" \
#        DATA.FEATURE "sup_vitb16_imagenet21k" \
#        DATA.NAME 'vtab-sun397' \
#        DATA.NUMBER_CLASSES "397" \
#        SOLVER.BASE_LR "1.0" \
#        SOLVER.WEIGHT_DECAY "0.0" \
#        SEED ${seed} \
#        MODEL.MODEL_ROOT "${model_root}" \
#        DATA.DATAPATH "${data_path}/vtab_data" \
#        OUTPUT_DIR "${output_dir}/seed_old_deep${seed}"
#done

# svhn meta-net
for seed in "42"; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config-file configs/prompt/cub.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "64" MODEL.PROMPT.ADD_INSTANCE_FEATURE "True"\
        MODEL.PROMPT.NUM_TOKENS "200" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.NAME 'vtab-svhn' \
        DATA.NUMBER_CLASSES "10" \
        SOLVER.BASE_LR "0.5" \
        SOLVER.WEIGHT_DECAY "0.01" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}/vtab_data" \
        OUTPUT_DIR "${output_dir}/seed_new_deep${seed}"
done

# svhn original
for seed in "42"; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config-file configs/prompt/cub.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "64" \
        MODEL.PROMPT.NUM_TOKENS "200" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.NAME 'vtab-svhn' \
        DATA.NUMBER_CLASSES "10" \
        SOLVER.BASE_LR "0.5" \
        SOLVER.WEIGHT_DECAY "0.01" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}/vtab_data" \
        OUTPUT_DIR "${output_dir}/seed_old_deep${seed}"
done

## patch_camelyon meta-net 数据集还没下载下来
#for seed in "42"; do
#    CUDA_VISIBLE_DEVICES=0 python train.py \
#        --config-file configs/prompt/cub.yaml \
#        MODEL.TYPE "vit" \
#        DATA.BATCH_SIZE "128" MODEL.PROMPT.ADD_INSTANCE_FEATURE "True"\
#        MODEL.PROMPT.NUM_TOKENS "5" \
#        MODEL.PROMPT.DEEP "True" \
#        MODEL.PROMPT.DROPOUT "0.1" \
#        DATA.FEATURE "sup_vitb16_imagenet21k" \
#        DATA.NAME 'vtab-patch_camelyon' \
#        DATA.NUMBER_CLASSES "2" \
#        SOLVER.BASE_LR "1.0" \
#        SOLVER.WEIGHT_DECAY "0.01" \
#        SEED ${seed} \
#        MODEL.MODEL_ROOT "${model_root}" \
#        DATA.DATAPATH "${data_path}/vtab_data" \
#        OUTPUT_DIR "${output_dir}/seed_new_deep${seed}"
#done

## patch_camelyon original 数据集还没下载下来
#for seed in "42"; do
#    CUDA_VISIBLE_DEVICES=0 python train.py \
#        --config-file configs/prompt/cub.yaml \
#        MODEL.TYPE "vit" \
#        DATA.BATCH_SIZE "128" \
#        MODEL.PROMPT.NUM_TOKENS "5" \
#        MODEL.PROMPT.DEEP "True" \
#        MODEL.PROMPT.DROPOUT "0.1" \
#        DATA.FEATURE "sup_vitb16_imagenet21k" \
#        DATA.NAME 'vtab-patch_camelyon' \
#        DATA.NUMBER_CLASSES "2" \
#        SOLVER.BASE_LR "1.0" \
#        SOLVER.WEIGHT_DECAY "0.01" \
#        SEED ${seed} \
#        MODEL.MODEL_ROOT "${model_root}" \
#        DATA.DATAPATH "${data_path}/vtab_data" \
#        OUTPUT_DIR "${output_dir}/seed_old_deep${seed}"
#done

# resisc45 meta-net
for seed in "42"; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config-file configs/prompt/cub.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "128" MODEL.PROMPT.ADD_INSTANCE_FEATURE "True"\
        MODEL.PROMPT.NUM_TOKENS "50" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.NAME 'vtab-resisc45' \
        DATA.NUMBER_CLASSES "45" \
        SOLVER.BASE_LR "10.0" \
        SOLVER.WEIGHT_DECAY "0.001" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}/vtab_data" \
        OUTPUT_DIR "${output_dir}/seed_new_deep${seed}"
done

# resisc45 original
for seed in "42"; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config-file configs/prompt/cub.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "128" \
        MODEL.PROMPT.NUM_TOKENS "50" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.NAME 'vtab-resisc45' \
        DATA.NUMBER_CLASSES "45" \
        SOLVER.BASE_LR "10.0" \
        SOLVER.WEIGHT_DECAY "0.001" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}/vtab_data" \
        OUTPUT_DIR "${output_dir}/seed_old_deep${seed}"
done

# eurosat meta-net
for seed in "42"; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config-file configs/prompt/cub.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "128" MODEL.PROMPT.ADD_INSTANCE_FEATURE "True"\
        MODEL.PROMPT.NUM_TOKENS "50" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.NAME 'vtab-eurosat' \
        DATA.NUMBER_CLASSES "10" \
        SOLVER.BASE_LR "100.0" \
        SOLVER.WEIGHT_DECAY "0.001" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}/vtab_data" \
        OUTPUT_DIR "${output_dir}/seed_new_deep${seed}"
done

# eurosat original
for seed in "42"; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config-file configs/prompt/cub.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "128" \
        MODEL.PROMPT.NUM_TOKENS "50" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.NAME 'vtab-eurosat' \
        DATA.NUMBER_CLASSES "10" \
        SOLVER.BASE_LR "100.0" \
        SOLVER.WEIGHT_DECAY "0.001" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}/vtab_data" \
        OUTPUT_DIR "${output_dir}/seed_old_deep${seed}"
done

# ImportError: cannot import name 'Any' from 'typing_extensions'
## diabetic_retinopathy(config="btgraham-300") meta-net
#for seed in "42"; do
#    CUDA_VISIBLE_DEVICES=0 python train.py \
#        --config-file configs/prompt/cub.yaml \
#        MODEL.TYPE "vit" \
#        DATA.BATCH_SIZE "128" MODEL.PROMPT.ADD_INSTANCE_FEATURE "True"\
#        MODEL.PROMPT.NUM_TOKENS "10" \
#        MODEL.PROMPT.DEEP "True" \
#        MODEL.PROMPT.DROPOUT "0.1" \
#        DATA.FEATURE "sup_vitb16_imagenet21k" \
#        DATA.NAME 'vtab-diabetic_retinopathy(config="btgraham-300")' \
#        DATA.NUMBER_CLASSES "5" \
#        SOLVER.BASE_LR "0.1" \
#        SOLVER.WEIGHT_DECAY "0.001" \
#        SEED ${seed} \
#        MODEL.MODEL_ROOT "${model_root}" \
#        DATA.DATAPATH "${data_path}/vtab_data" \
#        OUTPUT_DIR "${output_dir}/seed_new_deep${seed}"
#done

# ImportError: cannot import name 'Any' from 'typing_extensions'
## diabetic_retinopathy(config="btgraham-300") original
#for seed in "42"; do
#    CUDA_VISIBLE_DEVICES=0 python train.py \
#        --config-file configs/prompt/cub.yaml \
#        MODEL.TYPE "vit" \
#        DATA.BATCH_SIZE "128" \
#        MODEL.PROMPT.NUM_TOKENS "10" \
#        MODEL.PROMPT.DEEP "True" \
#        MODEL.PROMPT.DROPOUT "0.1" \
#        DATA.FEATURE "sup_vitb16_imagenet21k" \
#        DATA.NAME 'vtab-diabetic_retinopathy(config="btgraham-300")' \
#        DATA.NUMBER_CLASSES "5" \
#        SOLVER.BASE_LR "0.1" \
#        SOLVER.WEIGHT_DECAY "0.001" \
#        SEED ${seed} \
#        MODEL.MODEL_ROOT "${model_root}" \
#        DATA.DATAPATH "${data_path}/vtab_data" \
#        OUTPUT_DIR "${output_dir}/seed_old_deep${seed}"
#done

# dmlab meta-net
for seed in "42"; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config-file configs/prompt/cub.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "64" MODEL.PROMPT.ADD_INSTANCE_FEATURE "True"\
        MODEL.PROMPT.NUM_TOKENS "100" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.NAME 'vtab-dmlab' \
        DATA.NUMBER_CLASSES "6" \
        SOLVER.BASE_LR "500.0" \
        SOLVER.WEIGHT_DECAY "0.0" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}/vtab_data" \
        OUTPUT_DIR "${output_dir}/seed_new_deep${seed}"
done

# dmlab original
for seed in "42"; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config-file configs/prompt/cub.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "64" \
        MODEL.PROMPT.NUM_TOKENS "100" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.NAME 'vtab-dmlab' \
        DATA.NUMBER_CLASSES "6" \
        SOLVER.BASE_LR "500.0" \
        SOLVER.WEIGHT_DECAY "0.0" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}/vtab_data" \
        OUTPUT_DIR "${output_dir}/seed_old_deep${seed}"
done

# kitti(task="closest_vehicle_distance") meta-net
for seed in "42"; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config-file configs/prompt/cub.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "64" MODEL.PROMPT.ADD_INSTANCE_FEATURE "True"\
        MODEL.PROMPT.NUM_TOKENS "100" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.NAME 'vtab-kitti(task="closest_vehicle_distance")' \
        DATA.NUMBER_CLASSES "4" \
        SOLVER.BASE_LR "250.0" \
        SOLVER.WEIGHT_DECAY "0.0" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}/vtab_data" \
        OUTPUT_DIR "${output_dir}/seed_new_deep${seed}"
done

# kitti(task="closest_vehicle_distance") original
for seed in "42"; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config-file configs/prompt/cub.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "64" \
        MODEL.PROMPT.NUM_TOKENS "100" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.NAME 'vtab-kitti(task="closest_vehicle_distance")' \
        DATA.NUMBER_CLASSES "4" \
        SOLVER.BASE_LR "250.0" \
        SOLVER.WEIGHT_DECAY "0.0" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}/vtab_data" \
        OUTPUT_DIR "${output_dir}/seed_old_deep${seed}"
done

# smallnorb(predicted_attribute="label_azimuth") meta-net
for seed in "42"; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config-file configs/prompt/cub.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "64" MODEL.PROMPT.ADD_INSTANCE_FEATURE "True"\
        MODEL.PROMPT.NUM_TOKENS "200" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.NAME 'vtab-smallnorb(predicted_attribute="label_azimuth")' \
        DATA.NUMBER_CLASSES "18" \
        SOLVER.BASE_LR "1.0" \
        SOLVER.WEIGHT_DECAY "0.01" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}/vtab_data" \
        OUTPUT_DIR "${output_dir}/seed_new_deep${seed}"
done

# smallnorb(predicted_attribute="label_azimuth") original
for seed in "42"; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config-file configs/prompt/cub.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "64" \
        MODEL.PROMPT.NUM_TOKENS "200" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.NAME 'vtab-smallnorb(predicted_attribute="label_azimuth")' \
        DATA.NUMBER_CLASSES "18" \
        SOLVER.BASE_LR "1.0" \
        SOLVER.WEIGHT_DECAY "0.01" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}/vtab_data" \
        OUTPUT_DIR "${output_dir}/seed_old_deep${seed}"
done

# smallnorb(predicted_attribute="label_elevation") meta-net
for seed in "42"; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config-file configs/prompt/cub.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "64" MODEL.PROMPT.ADD_INSTANCE_FEATURE "True"\
        MODEL.PROMPT.NUM_TOKENS "200" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.NAME 'vtab-smallnorb(predicted_attribute="label_elevation")' \
        DATA.NUMBER_CLASSES "9" \
        SOLVER.BASE_LR "500.0" \
        SOLVER.WEIGHT_DECAY "0.0" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}/vtab_data" \
        OUTPUT_DIR "${output_dir}/seed_new_deep${seed}"
done

# smallnorb(predicted_attribute="label_elevation") original
for seed in "42"; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config-file configs/prompt/cub.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "64" \
        MODEL.PROMPT.NUM_TOKENS "200" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.NAME 'vtab-smallnorb(predicted_attribute="label_elevation")' \
        DATA.NUMBER_CLASSES "9" \
        SOLVER.BASE_LR "500.0" \
        SOLVER.WEIGHT_DECAY "0.0" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}/vtab_data" \
        OUTPUT_DIR "${output_dir}/seed_old_deep${seed}"
done

# dsprites(predicted_attribute="label_x_position",num_classes=16) meta-net
for seed in "42"; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config-file configs/prompt/cub.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "64" MODEL.PROMPT.ADD_INSTANCE_FEATURE "True"\
        MODEL.PROMPT.NUM_TOKENS "100" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.NAME 'vtab-dsprites(predicted_attribute="label_x_position",num_classes=16)' \
        DATA.NUMBER_CLASSES "16" \
        SOLVER.BASE_LR "1.0" \
        SOLVER.WEIGHT_DECAY "0.01" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}/vtab_data" \
        OUTPUT_DIR "${output_dir}/seed_new_deep${seed}"
done

# dsprites(predicted_attribute="label_x_position",num_classes=16) original
for seed in "42"; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config-file configs/prompt/cub.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "64" \
        MODEL.PROMPT.NUM_TOKENS "100" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.NAME 'vtab-dsprites(predicted_attribute="label_x_position",num_classes=16)' \
        DATA.NUMBER_CLASSES "16" \
        SOLVER.BASE_LR "1.0" \
        SOLVER.WEIGHT_DECAY "0.01" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}/vtab_data" \
        OUTPUT_DIR "${output_dir}/seed_old_deep${seed}"
done

# dsprites(predicted_attribute="label_orientation",num_classes=16) meta-net
for seed in "42"; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config-file configs/prompt/cub.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "64" MODEL.PROMPT.ADD_INSTANCE_FEATURE "True"\
        MODEL.PROMPT.NUM_TOKENS "100" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.NAME 'vtab-dsprites(predicted_attribute="label_orientation",num_classes=16)' \
        DATA.NUMBER_CLASSES "16" \
        SOLVER.BASE_LR "0.5" \
        SOLVER.WEIGHT_DECAY "0.01" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}/vtab_data" \
        OUTPUT_DIR "${output_dir}/seed_new_deep${seed}"
done

# dsprites(predicted_attribute="label_orientation",num_classes=16) original
for seed in "42"; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config-file configs/prompt/cub.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "64" \
        MODEL.PROMPT.NUM_TOKENS "100" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.NAME 'vtab-dsprites(predicted_attribute="label_orientation",num_classes=16)' \
        DATA.NUMBER_CLASSES "16" \
        SOLVER.BASE_LR "0.5" \
        SOLVER.WEIGHT_DECAY "0.01" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}/vtab_data" \
        OUTPUT_DIR "${output_dir}/seed_old_deep${seed}"
done

# clevr(task="closest_object_distance") meta-net
for seed in "42"; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config-file configs/prompt/cub.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "64" MODEL.PROMPT.ADD_INSTANCE_FEATURE "True"\
        MODEL.PROMPT.NUM_TOKENS "200" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.NAME 'vtab-clevr(task="closest_object_distance")' \
        DATA.NUMBER_CLASSES "6" \
        SOLVER.BASE_LR "2.5" \
        SOLVER.WEIGHT_DECAY "0.01" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}/vtab_data" \
        OUTPUT_DIR "${output_dir}/seed_new_deep${seed}"
done

# clevr(task="closest_object_distance") original
for seed in "42"; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config-file configs/prompt/cub.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "64" \
        MODEL.PROMPT.NUM_TOKENS "200" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.NAME 'vtab-clevr(task="closest_object_distance")' \
        DATA.NUMBER_CLASSES "6" \
        SOLVER.BASE_LR "2.5" \
        SOLVER.WEIGHT_DECAY "0.01" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}/vtab_data" \
        OUTPUT_DIR "${output_dir}/seed_old_deep${seed}"
done

# clevr(task="count_all") meta-net
for seed in "42"; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config-file configs/prompt/cub.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "64" MODEL.PROMPT.ADD_INSTANCE_FEATURE "True"\
        MODEL.PROMPT.NUM_TOKENS "100" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.NAME 'vtab-clevr(task="count_all")' \
        DATA.NUMBER_CLASSES "8" \
        SOLVER.BASE_LR "500.0" \
        SOLVER.WEIGHT_DECAY "0.0" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}/vtab_data" \
        OUTPUT_DIR "${output_dir}/seed_new_deep${seed}"
done

# clevr(task="count_all") original
for seed in "42"; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config-file configs/prompt/cub.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "64" \
        MODEL.PROMPT.NUM_TOKENS "100" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.NAME 'vtab-clevr(task="count_all")' \
        DATA.NUMBER_CLASSES "8" \
        SOLVER.BASE_LR "500.0" \
        SOLVER.WEIGHT_DECAY "0.0" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}/vtab_data" \
        OUTPUT_DIR "${output_dir}/seed_old_deep${seed}"
done

# CUB meta-net
#for seed in "42" "44"; do
#    CUDA_VISIBLE_DEVICES=0 python train.py \
#        --config-file configs/prompt/cub.yaml \
#        MODEL.TYPE "vit" MODEL.PROMPT.ADD_INSTANCE_FEATURE "True" \
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
#        OUTPUT_DIR "${output_dir}/seed_new_deep_complex${seed}"
#done

# CUB original
#for seed in "42" "44" "82"; do
#    CUDA_VISIBLE_DEVICES=0 python train.py \
#        --config-file configs/prompt/cub.yaml \
#        MODEL.TYPE "vit" \
#        DATA.BATCH_SIZE "64" \
#        MODEL.PROMPT.NUM_TOKENS "100" \
#        MODEL.PROMPT.DEEP "True" \
#        MODEL.PROMPT.DROPOUT "0.1" \
#        DATA.FEATURE "sup_vitb16_imagenet21k" \
#        DATA.NAME "CUB" \
#        DATA.NUMBER_CLASSES "200" \
#        SOLVER.BASE_LR "0.5" \
#        SOLVER.WEIGHT_DECAY "0.001" \
#        SEED ${seed} \
#        MODEL.MODEL_ROOT "${model_root}" \
#        DATA.DATAPATH "${data_path}/CUB_200_2011" \
#        OUTPUT_DIR "${output_dir}/seed_origin_deep${seed}"
#done

# OxfordFlowers meta-net
#for seed in "42" "44"; do
#    CUDA_VISIBLE_DEVICES=0 python train.py \
#        --config-file configs/prompt/cub.yaml \
#        MODEL.TYPE "vit" MODEL.PROMPT.ADD_INSTANCE_FEATURE "True" \
#        DATA.BATCH_SIZE "64" \
#        MODEL.PROMPT.NUM_TOKENS "100" \
#        MODEL.PROMPT.DEEP "True" \
#        MODEL.PROMPT.DROPOUT "0.1" \
#        DATA.FEATURE "sup_vitb16_imagenet21k" \
#        DATA.NAME "OxfordFlowers" \
#        DATA.NUMBER_CLASSES "102" \
#        SOLVER.BASE_LR "5.0" \
#        SOLVER.WEIGHT_DECAY "0.001" \
#        SEED ${seed} \
#        MODEL.MODEL_ROOT "${model_root}" \
#        DATA.DATAPATH "${data_path}/Oxford_Flowers" \
#        OUTPUT_DIR "${output_dir}/seed_new_deep_complex${seed}"
#done

## OxfordFlowers original
#for seed in "42" "44" "82"; do
#    CUDA_VISIBLE_DEVICES=0 python train.py \
#        --config-file configs/prompt/cub.yaml \
#        MODEL.TYPE "vit" \
#        DATA.BATCH_SIZE "64" \
#        MODEL.PROMPT.NUM_TOKENS "100" \
#        MODEL.PROMPT.DEEP "True" \
#        MODEL.PROMPT.DROPOUT "0.1" \
#        DATA.FEATURE "sup_vitb16_imagenet21k" \
#        DATA.NAME "OxfordFlowers" \
#        DATA.NUMBER_CLASSES "102" \
#        SOLVER.BASE_LR "5.0" \
#        SOLVER.WEIGHT_DECAY "0.001" \
#        SEED ${seed} \
#        MODEL.MODEL_ROOT "${model_root}" \
#        DATA.DATAPATH "${data_path}/Oxford_Flowers" \
#        OUTPUT_DIR "${output_dir}/seed_origin_deep${seed}"
#done

# StanfordCars meta-net
#for seed in "42" "44" "82"; do
#    CUDA_VISIBLE_DEVICES=0 python train.py \
#        --config-file configs/prompt/cub.yaml \
#        MODEL.TYPE "vit" MODEL.PROMPT.ADD_INSTANCE_FEATURE "True" \
#        DATA.BATCH_SIZE "64" \
#        MODEL.PROMPT.NUM_TOKENS "100" \
#        MODEL.PROMPT.DEEP "True" \
#        MODEL.PROMPT.DROPOUT "0.1" \
#        DATA.FEATURE "sup_vitb16_imagenet21k" \
#        DATA.NAME "StanfordCars" \
#        DATA.NUMBER_CLASSES "196" \
#        SOLVER.BASE_LR "500.0" \
#        SOLVER.WEIGHT_DECAY "0.0" \
#        SEED ${seed} \
#        MODEL.MODEL_ROOT "${model_root}" \
#        DATA.DATAPATH "${data_path}/Stanford_Cars" \
#        OUTPUT_DIR "${output_dir}/seed_new_deep${seed}"
#done

# StanfordCars original
#for seed in "42" "44" "82"; do
#    CUDA_VISIBLE_DEVICES=0 python train.py \
#        --config-file configs/prompt/cub.yaml \
#        MODEL.TYPE "vit" \
#        DATA.BATCH_SIZE "64" \
#        MODEL.PROMPT.NUM_TOKENS "100" \
#        MODEL.PROMPT.DEEP "True" \
#        MODEL.PROMPT.DROPOUT "0.1" \
#        DATA.FEATURE "sup_vitb16_imagenet21k" \
#        DATA.NAME "StanfordCars" \
#        DATA.NUMBER_CLASSES "196" \
#        SOLVER.BASE_LR "500.0" \
#        SOLVER.WEIGHT_DECAY "0.0" \
#        SEED ${seed} \
#        MODEL.MODEL_ROOT "${model_root}" \
#        DATA.DATAPATH "${data_path}/Stanford_Cars" \
#        OUTPUT_DIR "${output_dir}/seed_origin_deep${seed}"
#done

# StanfordDogs meta-net
#for seed in "42" "44"; do
#    CUDA_VISIBLE_DEVICES=0 python train.py \
#        --config-file configs/prompt/cub.yaml \
#        MODEL.TYPE "vit" MODEL.PROMPT.ADD_INSTANCE_FEATURE "True" \
#        DATA.BATCH_SIZE "64" \
#        MODEL.PROMPT.NUM_TOKENS "100" \
#        MODEL.PROMPT.DEEP "True" \
#        MODEL.PROMPT.DROPOUT "0.1" \
#        DATA.FEATURE "sup_vitb16_imagenet21k" \
#        DATA.NAME "StanfordDogs" \
#        DATA.NUMBER_CLASSES "120" \
#        SOLVER.BASE_LR "1.0" \
#        SOLVER.WEIGHT_DECAY "0.0001" \
#        SEED ${seed} \
#        MODEL.MODEL_ROOT "${model_root}" \
#        DATA.DATAPATH "${data_path}/Stanford_Dogs" \
#        OUTPUT_DIR "${output_dir}/seed_new_deep_complex${seed}"
#done

# StanfordDogs original
#for seed in "42" "44" "82"; do
#    CUDA_VISIBLE_DEVICES=0 python train.py \
#        --config-file configs/prompt/cub.yaml \
#        MODEL.TYPE "vit" \
#        DATA.BATCH_SIZE "64" \
#        MODEL.PROMPT.NUM_TOKENS "100" \
#        MODEL.PROMPT.DEEP "True" \
#        MODEL.PROMPT.DROPOUT "0.1" \
#        DATA.FEATURE "sup_vitb16_imagenet21k" \
#        DATA.NAME "StanfordDogs" \
#        DATA.NUMBER_CLASSES "120" \
#        SOLVER.BASE_LR "1.0" \
#        SOLVER.WEIGHT_DECAY "0.0001" \
#        SEED ${seed} \
#        MODEL.MODEL_ROOT "${model_root}" \
#        DATA.DATAPATH "${data_path}/Stanford_Dogs" \
#        OUTPUT_DIR "${output_dir}/seed_origin_deep${seed}"
#done

# nabirds meta-net
#for seed in "42" "44" "82"; do
#    CUDA_VISIBLE_DEVICES=0 python train.py \
#        --config-file configs/prompt/cub.yaml \
#        MODEL.TYPE "vit" MODEL.PROMPT.ADD_INSTANCE_FEATURE "True" \
#        DATA.BATCH_SIZE "128" \
#        MODEL.PROMPT.NUM_TOKENS "50" \
#        MODEL.PROMPT.DEEP "True" \
#        MODEL.PROMPT.DROPOUT "0.1" \
#        DATA.FEATURE "sup_vitb16_imagenet21k" \
#        DATA.NAME "nabirds" \
#        DATA.NUMBER_CLASSES "555" \
#        SOLVER.BASE_LR "10.0" \
#        SOLVER.WEIGHT_DECAY "0.0" \
#        SEED ${seed} \
#        MODEL.MODEL_ROOT "${model_root}" \
#        DATA.DATAPATH "${data_path}/nabirds" \
#        OUTPUT_DIR "${output_dir}/seed_new_deep${seed}"
#done

# nabirds original
#for seed in "42" "44" "82"; do
#    CUDA_VISIBLE_DEVICES=0 python train.py \
#        --config-file configs/prompt/cub.yaml \
#        MODEL.TYPE "vit" \
#        DATA.BATCH_SIZE "128" \
#        MODEL.PROMPT.NUM_TOKENS "50" \
#        MODEL.PROMPT.DEEP "True" \
#        MODEL.PROMPT.DROPOUT "0.1" \
#        DATA.FEATURE "sup_vitb16_imagenet21k" \
#        DATA.NAME "nabirds" \
#        DATA.NUMBER_CLASSES "555" \
#        SOLVER.BASE_LR "10.0" \
#        SOLVER.WEIGHT_DECAY "0.0" \
#        SEED ${seed} \
#        MODEL.MODEL_ROOT "${model_root}" \
#        DATA.DATAPATH "${data_path}/nabirds" \
#        OUTPUT_DIR "${output_dir}/seed_origin_deep${seed}"
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