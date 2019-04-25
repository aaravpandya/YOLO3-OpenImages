
# YOLOV3 implementation on Open Images + Color Detection

## Which model to use
Use any of the below ways to build the model. Put the .h5 model in model_data
 1. Convert pjreddie's darknet model to keras using qqwweee's tool below.
 2. Use [oimodel.py](https://github.com/marcu5fen1x/YOLO3-OpenImages/blob/master/oimodel.py) to build a model with randomly initialized weights and train it yourself.
 3. Use [oimodel.py](https://github.com/marcu5fen1x/YOLO3-OpenImages/blob/master/oimodel.py) and comment out the setweights line to use pjreddie's weights. Then customize the layers as you wish. 
 4. Use [layer_config.json](https://github.com/marcu5fen1x/YOLO3-OpenImages/blob/master/layer_config.json) to customize your own layers and their parameters. See keras docs for loading json config files as pre trained models.
 ## Docker Build Instructions
 On a linux machine, enter the following in the terminal

    sudo docker build -t <yourtagname> .
tagname will be of the format "username/imagename"
Push it to dockerhub by

    docker push <tagname>
Deploy :D
## Credits
https://github.com/qqwweee/keras-yolo3 for converting the darknet model to .h5 keras model

