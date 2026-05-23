export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

deepspeed train.py --deepspeed --config anima1.toml
