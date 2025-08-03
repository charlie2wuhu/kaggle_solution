MODEL=${1:-rf}
FEATURE=${2:-baseline}
CV=${3:-1}
python train.py --fold 0 --model $MODEL --features $FEATURE --cv $CV
python train.py --fold 1 --model $MODEL --features $FEATURE --cv $CV
python train.py --fold 2 --model $MODEL --features $FEATURE --cv $CV
python train.py --fold 3 --model $MODEL --features $FEATURE --cv $CV
python train.py --fold 4 --model $MODEL --features $FEATURE --cv $CV