#1 import
import numpy as np
import random 
from PIL import Image

num = np.arange(84000)
random.shuffle(num)



# Input Dataset
img = np.zeros((84000,28*28),dtype = 'float32')
img_ = np.zeros((84000,28*28),dtype = 'float32')

for n in range(1,84001):

    #0
    if n<=5923: 
        im = Image.open("training/0/"+ str(n) +".png")

    #1
    elif 5924<= n and n <= 12665:
        im = Image.open("training/1/"+ str(n-5923) +".png")

    #2
    elif 12666<= n and n <= 18623:
        im = Image.open("training/2/"+ str(n-12665) +".png")

    #3
    elif 18624<= n and n <= 24754:
        im = Image.open("training/3/"+ str(n-18623) +".png")

    #4
    elif 24755<= n and n <= 30596:
        im = Image.open("training/4/"+ str(n-24754) +".png")

    #5
    elif 30597<= n and n <= 36017:
        im = Image.open("training/5/"+ str(n-30596) +".png")

    #6
    elif 36018<= n and n <= 41935:
        im = Image.open("training/6/"+ str(n-36017) +".png")

    #7
    elif 41936<= n and n <= 48200:
        im = Image.open("training/7/"+ str(n-41935) +".png")

    #8
    elif 48201<= n and n <= 54051:
        im = Image.open("training/8/"+ str(n-48200) +".png")

    #9
    elif 54052 <= n and n <= 60000:
        im = Image.open("training/9/"+ str(n-54051) +".png")

    #10
    elif 60001<= n and n <= 66000:
        im = Image.open("training/10/"+ str(n-60000) +".png")

    #11
    elif 66001<= n and n <= 72000:
        im = Image.open("training/11/"+ str(n-66000) +".png")

    #12
    elif 72001<= n and n <= 78000:
        im = Image.open("training/12/"+ str(n-72000) +".png")

    #13
    elif 78001<= n and n <= 84000:
        im = Image.open("training/13/"+ str(n-78000) +".png")

    
    for i in range(28):
        for j in range(28):
            img[n-1,28*j + i] = im.getpixel((i,j)) /255.

for n in range(84000):
    img_[n] = img[num[n]]

np.savetxt('training_pmmd.txt', img_,fmt = '%.1e', delimiter=' ')



# Output Dataset
oimg = np.zeros((84000,14),dtype = 'float32')
oimg_ = np.zeros((84000,14),dtype = 'float32')
for n in range(84000):

    #0
    if n<5923: 
         oimg[n][0] = 1

    #1
    elif 5923<= n and n < 12665:
         oimg[n][1] = 1

    #2
    elif 12665<= n and n < 18623:
         oimg[n][2] = 1

    #3
    elif 18623<= n and n < 24754:
         oimg[n][3] = 1

    #4
    elif 24754<= n and n < 30596:
        oimg[n][4] = 1

    #5
    elif 30596<= n and n < 36017:
         oimg[n][5] = 1

    #6
    elif 36017<= n and n < 41935:
         oimg[n][6] = 1

    #7
    elif 41935<= n and n < 48200:
         oimg[n][7] = 1

    #8
    elif 48200<= n and n < 54051:
         oimg[n][8] = 1

    #9
    elif 54051<= n and n < 60000:
         oimg[n][9] = 1

    #10
    elif 60000<= n and n < 66000:
         oimg[n][10] = 1

    #11
    elif 66000<= n and n < 72000:
         oimg[n][11] = 1

    #12
    elif 72000<= n and n < 78000:
         oimg[n][12] = 1

    #13
    elif 78000<= n and n < 84000:
         oimg[n][13] = 1

for n in range(84000):
    oimg_[n] = oimg[num[n]]

np.savetxt('training_pmmd_y.txt', oimg_, delimiter=' ')  


print(num)
print(img_)
print()
print(oimg_)
