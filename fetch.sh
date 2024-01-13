#/bin/bash
if [ -d "model" ]; then
    rm -rf model/
fi
mkdir model

echo "Retrieving source URI."
uri=$(cat source.json | jq '.uri' | tr -d '"')
echo "URI detected: ${uri}"
curl -L ${uri} >> data.zip && unzip -q data.zip -d model && rm data.zip

# Fetch the model from GitHub, and place the data.yaml to the correct path, also append the models folder to the path variable.

echo "Retrieving YOLOv5 from GitHub."
git clone https://github.com/ultralytics/yolov5.git model/yolov5
cp model/data.yaml model/yolov5/data/Markers.yaml
export PATH=$PATH:$(pwd)/model/yolov5/models