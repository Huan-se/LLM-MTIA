PATH_TO_LLAMA31_8B_INSTRUCT='Meta-Llama-3.1-8B-Instruct' # TO SET
PATH_TO_INVERT_MODEL='t5-base' # TO SET
SAVE_FOLDER='long-prompt'

# Generation of training data
python prepare_dataset.py --model-name $PATH_TO_LLAMA31_8B_INSTRUCT --invert-model $PATH_TO_INVERT_MODEL  --dataset-name norobot-train  --access-layer-id 16 --maxseqlen 256 --savefolder $SAVE_FOLDER

# Generation of test data
python prepare_dataset.py --model-name $PATH_TO_LLAMA31_8B_INSTRUCT --invert-model $PATH_TO_INVERT_MODEL  --dataset-name mentalhealth  --access-layer-id 16 --savefolder $SAVE_FOLDER

python prepare_dataset.py --model-name $PATH_TO_LLAMA31_8B_INSTRUCT --invert-model  $PATH_TO_INVERT_MODEL  --dataset-name evolcode  --access-layer-id 16 --savefolder $SAVE_FOLDER

python prepare_dataset.py --model-name $PATH_TO_LLAMA31_8B_INSTRUCT --invert-model  $PATH_TO_INVERT_MODEL  --dataset-name norobot-test  --access-layer-id 16 --savefolder $SAVE_FOLDER
