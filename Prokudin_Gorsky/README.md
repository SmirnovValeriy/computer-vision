# Prokudin-Gorsky

Prokudin-Gorsky is the pioneer of Russian color photography. Each his photo represents combination of three images corresponding to three channels: red, green and blue. 

![](https://cdn1.savepice.ru/uploads/2020/3/31/7ca9774aba931d18b3353ffcd20bc0a6-full.png "ThreeChannelsImgExample.png")

In the task you need to write functions combining these images into a colored photo.

![](https://cdn1.savepice.ru/uploads/2020/3/31/82a6f955eab6e3ec1f02bc0988afca6f-full.png  "ColoredExampleImg.png")

At the entrance you have a picture with three channels. It is necessary to сut edges and then cobmine three channels in such a way that MSE-metrics is maximum.
To reduce the execution time, we suggest using the pyramid of images:
1. let's sequentially reduce by half the image resolution to a certain size;
2. then starting from the smallest image find the optimal shift;
3. return to the bigger image and correct the shift and so on.

Solution for the task is presented in jupyter-notebook **[prokudinGorsky.ipynb](prokudinGorsky.ipynb)**. Python code for the solution you can find in **[align.py](align.py)**.

You can test the functions using **[run.py](run.py)** and **[public_tests](public_tests)** located in the same folder: \
**`$ ./run.py public_tests`** \
***NB:*** you might have to change python3 interpretator in **run.py** (for example I have changed it to **/home/valeriy/anaconda3/bin/python3**).

A whole task description is available only in Russian. You can find it in **[WholeTaskDescription.pdf](WholeTaskDescription.pdf)**

