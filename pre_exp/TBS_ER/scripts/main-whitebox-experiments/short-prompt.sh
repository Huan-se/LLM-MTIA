
# TS

python attack_batch.py     --model-name llama2-7bchat     --dataset chat_2m     --range 0:200 --folder xxxcompare    --access-layer-id 2     --attack ts    --num-steps 50000     --lr 0.1   --wd 0.001  --in-state-loss mse     --device cuda

python attack_batch.py     --model-name llama2-7bchat     --dataset chat_2m     --range 200:479 --folder compare    --access-layer-id 2     --attack ts    --num-steps 50000     --lr 0.1   --wd 0.001  --in-state-loss mse     --device cuda

python attack_batch.py     --model-name llama2-7bchat     --dataset chat_2m     --range 0:180 --folder compare    --access-layer-id 8     --attack ts    --num-steps 50000     --lr 0.01   --wd 0.001  --in-state-loss mse     --device cuda

python attack_batch.py     --model-name llama2-7bchat     --dataset chat_2m     --range 180:479 --folder compare    --access-layer-id 8     --attack ts    --num-steps 50000     --lr 0.01   --wd 0.001  --in-state-loss mse     --device cuda




# ER

python attack_batch.py     --model-name llama2-7bchat     --dataset chat_2m     --range 0:479 --folder test    --access-layer-id 2     --attack er    --num-steps 50000     --lr 0.001     --in-state-loss mse     --device cuda # 27H

python attack_batch.py     --model-name llama2-7bchat     --dataset chat_2m     --range 0:479 --folder test    --access-layer-id 4     --attack er    --num-steps 50000     --lr 0.001     --in-state-loss mse     --device cuda # 52H

python attack_batch.py     --model-name llama2-7bchat     --dataset chat_2m     --range 0:200 --folder test    --access-layer-id 8     --attack er    --num-steps 50000     --lr 0.001     --in-state-loss mse     --device cuda # 44H

python attack_batch.py     --model-name llama2-7bchat     --dataset chat_2m     --range 200:479 --folder test    --access-layer-id 8     --attack er    --num-steps 50000     --lr 0.001     --in-state-loss mse     --device cuda # 29H




# L32

python attack_batch.py     --model-name llama2-7bchat     --dataset chat_2m     --range 0:40 --folder test    --access-layer-id 32     --attack tbs    --num-steps 50000     --lr 0.01     --in-state-loss mse     --device cuda # 40H

python attack_batch.py     --model-name llama2-7bchat     --dataset chat_2m     --range 40:90 --folder test    --access-layer-id 32     --attack tbs    --num-steps 50000     --lr 0.01     --in-state-loss mse     --device cuda # 44H

python attack_batch.py     --model-name llama2-7bchat     --dataset chat_2m     --range 90:140 --folder test    --access-layer-id 32     --attack tbs    --num-steps 50000     --lr 0.01     --in-state-loss mse     --device cuda # 35H

python attack_batch.py     --model-name llama2-7bchat     --dataset chat_2m     --range 140:200 --folder test    --access-layer-id 32     --attack tbs    --num-steps 50000     --lr 0.01     --in-state-loss mse     --device cuda # 35H

python attack_batch.py     --model-name llama2-7bchat     --dataset chat_2m     --range  200:280  --folder test    --access-layer-id 32     --attack tbs    --num-steps 50000     --lr 0.01     --in-state-loss mse     --device cuda # 35H

python attack_batch.py     --model-name llama2-7bchat     --dataset chat_2m     --range  280:360  --folder test    --access-layer-id 32     --attack tbs    --num-steps 50000     --lr 0.01     --in-state-loss mse     --device cuda # 25H

python attack_batch.py     --model-name llama2-7bchat     --dataset chat_2m     --range  360:479  --folder test    --access-layer-id 32     --attack tbs    --num-steps 50000     --lr 0.01     --in-state-loss mse     --device cuda # 25H


# L24

python attack_batch.py     --model-name llama2-7bchat     --dataset chat_2m     --range 0:70 --folder test    --access-layer-id 24     --attack tbs    --num-steps 50000     --lr 0.01     --in-state-loss mse     --device cuda # 48H

python attack_batch.py     --model-name llama2-7bchat     --dataset chat_2m     --range 70:150 --folder test    --access-layer-id 24     --attack tbs    --num-steps 50000     --lr 0.01     --in-state-loss mse     --device cuda # 48H

python attack_batch.py     --model-name llama2-7bchat     --dataset chat_2m     --range 150:270 --folder test    --access-layer-id 24     --attack tbs    --num-steps 50000     --lr 0.01     --in-state-loss mse     --device cuda # 48H

python attack_batch.py     --model-name llama2-7bchat     --dataset chat_2m     --range 270:479 --folder test    --access-layer-id 24     --attack tbs    --num-steps 50000     --lr 0.01     --in-state-loss mse     --device cuda # 48H

# L16 TBS

python attack_batch.py     --model-name llama2-7bchat     --dataset chat_2m     --range 0:100 --folder test    --access-layer-id 16     --attack tbs    --num-steps 50000     --lr 0.001     --in-state-loss mse     --device cuda 48H

python attack_batch.py     --model-name llama2-7bchat     --dataset chat_2m     --range 100:240 --folder test    --access-layer-id 16     --attack tbs    --num-steps 50000     --lr 0.001     --in-state-loss mse     --device cuda 48H

python attack_batch.py     --model-name llama2-7bchat     --dataset chat_2m     --range 240:479 --folder test    --access-layer-id 16     --attack tbs    --num-steps 50000     --lr 0.001     --in-state-loss mse     --device cuda 48H

# L8 TBS

python attack_batch.py     --model-name llama2-7bchat     --dataset chat_2m     --range 0:170 --folder test    --access-layer-id 8     --attack tbs    --num-steps 50000     --lr 0.001     --in-state-loss mse     --device cuda 40H

python attack_batch.py     --model-name llama2-7bchat     --dataset chat_2m     --range 170:479 --folder test    --access-layer-id 8     --attack tbs    --num-steps 50000     --lr 0.001     --in-state-loss mse     --device cuda #40H

# L4 TBS

python attack_batch.py     --model-name llama2-7bchat     --dataset chat_2m     --range 0:479 --folder test    --access-layer-id 4     --attack tbs    --num-steps 50000     --lr 0.001     --in-state-loss mse     --device cuda # 60H