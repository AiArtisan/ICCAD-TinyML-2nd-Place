## How do I run these scripts?

You can run this classifier code by installing the requirements

    pip install requirements.txt

and running

    python train.py 
    python testing_performances.py

where `data_set` is a folder of the given [training dataset](https://drive.google.com/file/d/11lOiocENt7TVRqwYbUkIgd1_-ZCUNIrH/view?usp=sharing), `data_indices` is a folder of data indices (the given training dataset has been partitioned into training and testing dataset), `result` is a folder for saving our models and statistics outputs. The [TinyML Contest 2022 web-page](https://tinymlcontest.github.io/TinyML-Design-Contest/Problems.html) provides a description of the data files.

After running the scripts, one of the scoring metrics (i.e., **F-B**) will be reported in the file *test_seg_stat.txt* in the folder `result`. 

## How do I deploy the model on the board?

we will deploy the model on the board NUCLEO-L432KC with STM32CubeMX and the package X-Cube-AI. 

You can firstly convert the model to onnx format by running

```shell
python pkl2onnx.py 
```
