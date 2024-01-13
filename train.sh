#/bin/bash
if [ "$#" -ne 3 ]; then
    echo "Usage: train.sh <model size> <epochs> <batch size>"
    echo "Model Size: s, m, l, x"
    echo "Epochs: 1, 2, 3, ..."
    echo "Batch Size: 2, 4, 8, ..."
    exit 1
fi

MODEL_SIZE=$1
EPOCHS=$2
BATCH_SIZE=$3

if [ ! -d "model" ]; then
    echo "First run fetch.sh to download the required files."
    exit 1
fi


if [ "$MODEL_SIZE" = "s" ]; then
    WEIGHTS="yolov5s.pt"
elif [ "$MODEL_SIZE" = "m" ]; then
    WEIGHTS="yolov5m.pt"
elif [ "$MODEL_SIZE" = "l" ]; then
    WEIGHTS="yolov5l.pt"
elif [ "$MODEL_SIZE" = "x" ]; then
    WEIGHTS="yolov5x.pt"
else
    echo "Invalid model size. Please use s, m, l or x."
    exit 1
fi

echo "Training model with size ${MODEL_SIZE} for ${EPOCHS} epochs with batch size ${BATCH_SIZE}."
python3 model/yolov5/train.py --img 640 --batch ${BATCH_SIZE} --epochs ${EPOCHS} --data Markers.yaml --weights ${WEIGHTS} --cache ram

# From model/yolov5/runs/train find the latest run and copy the best.pt file to the working directory.
# This is the model that will be used for inference.

RUN=$(ls -td model/yolov5/runs/train/* | head -n 1)
cp ${RUN}/weights/best.pt best.pt

echo "Training complete. The best model is available as best.pt."