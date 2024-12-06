# clusters (modified for single machine)
worker_cnt=1
gpus_per_node=1
memory=400000
cpu=4000

# data
batch_size=8
selected_cols=0,4,2,3
max_src_length=80
max_tgt_length=30
patch_image_size=512
constraint_range=58457,59457

# optimization
lr=1e-04
clip_grad=1.0
schedule='polynomial_decay'
label_smoothing=0.1
weight_decay=0.01

# generation
beam=5
max_len_a=0
max_len_b=4
min_len=4

# distill config
kd_loss_weight=10000.0
kd_loss_type=ce_with_mask
intermediate_matches="first:attention_mse_sum:encoder,first:attention_mse_sum:decoder"


# save
student_model_config=ofa-tiny
load_student_model="ofa-tiny"
teacher_model_path="/content/drive/MyDrive/RefCoco-Visualgrounding/hg-base"
task=refcoco


save=.

data_dir=/content/drive/MyDrive/RefCoco-Visualgrounding/refcocog_data
DATA=${data_dir}/refcocog_train.tsv,${data_dir}/refcocog_val.tsv
half='fp16'
ckpt_frequency=2
init_method="load_pretrain"

export CUDA_VISIBLE_DEVICES="0" # Ensure the GPU is selected 0 if gpu none if cpu 

# Simplified execution for single-machine setup
python /content/OFA-Compress/main_distill.py \
    --tables=${DATA} \
    --selected-cols=${selected_cols} \
    --task=${task} \
    --schedule=${schedule} \
    --label-smoothing=${label_smoothing} \
    --kd-loss-weight=${kd_loss_weight} \
    --kd-loss-type=${kd_loss_type} \
    --intermediate-matches=${intermediate_matches} \
    --max-src-length=${max_src_length} \
    --max-tgt-length=${max_tgt_length} \
    --patch-image-size=${patch_image_size} \
    --constraint-range=${constraint_range} \
    --beam=${beam} \
    --max-len-a=${max_len_a} \
    --max-len-b=${max_len_b} \
    --min-len=${min_len} \
    --weight-decay=${weight_decay} \
    --clip-grad=${clip_grad} \
    --lr=${lr} \
    --fp16 \
    --batch-size=${batch_size} \
    --init-method=${init_method} \
    --student-model-config=${student_model_config} \
    --micro-batch-size=${batch_size} \
    --num-epochs=5 \
    --best-score=10e10 \
    --metric=ap \
    --do-train \
    --do-predict \
    --ckpt-frequency=${ckpt_frequency} \
    --output-dir=${save} \
    --load-teacher-model=${teacher_model_path} \
    --load-student-model=${load_student_model} \
    --worker-cnt=${worker_cnt} \
    --gpus-per-node=${gpus_per_node}\ 
     
   




/bin/bash