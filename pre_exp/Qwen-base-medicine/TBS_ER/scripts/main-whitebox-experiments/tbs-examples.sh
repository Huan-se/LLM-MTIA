modelname=qwen2.5-7binst # or qwen2.5coder-7binst
dataset=evolcode
i=10
python attack_batch.py     \
    --model-name $modelname \
    --dataset $dataset \
    --range $i:$((i+1)) \
    --folder long-context-$modelname/$dataset  \
    --access-layer-id 14  \
    --attack tbs \
    --num-steps 50000 \
    --lr 0.0005 \
    --w-dm 0 \
    --in-state-loss mse \
    --device cuda:0 \
    --optim AdamW

# cos distance as loss
modelname=qwen2.5-7binst # or qwen2.5coder-7binst
dataset=evolcode
i=10
python attack_batch.py     \
    --model-name $modelname \
    --dataset $dataset \
    --range $i:$((i+1)) \
    --folder long-context-$modelname-cos/$dataset  \
    --access-layer-id 14  \
    --attack tbs \
    --num-steps 50000 \
    --lr 0.0005 \
    --w-dm 0 \
    --in-state-loss cos \
    --device cuda:0 \
    --optim AdamW


# llama3, dm penalty
modelname=llama3-8binst
dataset=evolcode
i=10
python attack_batch.py     \
    --model-name $modelname \
    --dataset $dataset \
    --range $i:$((i+1)) \
    --folder long-context-$modelname/$dataset-lr1e-4-dm  \
    --access-layer-id 16  \
    --attack tbs \
    --num-steps 50000 \
    --lr 0.0001 \
    --w-dm 0.001 \
    --in-state-loss mse \
    --device cuda:0 \
    --optim AdamW

# llama3, singular basis
modelname=llama3-8binst
dataset=evolcode
for i in {0..9}
do
    python attack_batch.py     \
        --model-name $modelname \
        --dataset $dataset \
        --range $i:$((i+1)) \
        --folder newserver/long-context-$modelname/$dataset-lr1e-4-osb  \
        --access-layer-id 16  \
        --attack tbs \
        --num-steps 50000 \
        --lr 0.0001 \
        --w-dm 0 \
        --in-state-loss mse \
        --device cuda:0 \
        --optim AdamW \
        --singular 
done
