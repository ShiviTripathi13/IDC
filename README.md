# IDC

### Official pytorch implementation of the paper: "IDC: Internal Diverse Image Completion"

With IDC you can train a generative model from a partial single image in order to generate at inference diverse solution to the completion task.

## Code

### Install dependencies (based on SinGAN code)

```
python -m pip install -r requirements.txt
```

This code was tested with python 3.8

###  Train
To train IDC model on your own image, put the desire training image and mask under Input/Images, and run

```
python main_train.py --input_name <input_file_name> --input_mask <input_file_mask>

This will also generate diverse solutions. 
To control the diversity you can add --diversity=#(a number between 1 to 8).
This will control the diversity of the results when 1 - means lowest diversity and 8 - means highest diversity.
