MODEL=${1:-rf}
python train.py --fold 0 --model $MODEL
python train.py --fold 1 --model $MODEL
python train.py --fold 2 --model $MODEL
python train.py --fold 3 --model $MODEL
python train.py --fold 4 --model $MODEL