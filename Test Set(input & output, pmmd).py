#1 import
import numpy as np
import random
from PIL import Image


num = np.arange(14000)
random.shuffle(num)

img = np.zeros((14000,28*28),dtype = 'float32')
img_ = np.zeros((14000,28*28),dtype = 'float32')



# Input Dataset
for n in range(1,14001):

    #0
    if n<=980: 
        im = Image.open("testing/0/"+ str(n) +".png")

    #1
    elif 981<= n and n <= 2115:
        im = Image.open("testing/1_/"+ str(n-980) +".png")

    #2
    elif 2116<= n and n <= 3147:
        im = Image.open("testing/2/"+ str(n-2115) +".png")

    #3
    elif 3148<= n and n <= 4157:
        im = Image.open("testing/3/"+ str(n-3147) +".png")

    #4
    elif 4158<= n and n <= 5139:
        im = Image.open("testing/4/"+ str(n-4157) +".png")

    #5
    elif 5140<= n and n <= 6031:
        im = Image.open("testing/5/"+ str(n-5139) +".png")

    #6
    elif 6032<= n and n <= 6989:
        im = Image.open("testing/6/"+ str(n-6031) +".png")

    #7
    elif 6990<= n and n <= 8017:
        im = Image.open("testing/7/"+ str(n-6989) +".png")

    #8
    elif 8018<= n and n <= 8991:
        im = Image.open("testing/8/"+ str(n-8017) +".png")

    #9
    elif 8992<= n and n <= 10000:
        im = Image.open("testing/9/"+ str(n-8991) +".png")

    #10
    elif 10001 <= n and n <= 11000:
        im = Image.open("testing/10/"+ str(n-10000) +".png")

    #11
    elif 11001 <= n and n <= 12000:
        im = Image.open("testing/11/"+ str(n-11000) +".png")

    #12
    elif 12001 <= n and n <= 13000:
        im = Image.open("testing/12/"+ str(n-12000) +".png")

    #13
    elif 13001 <= n and n <= 14000:
        im = Image.open("testing/13/"+ str(n-13000) +".png")
    
    for i in range(28):
        for j in range(28):
            img[n-1,28*j + i] = im.getpixel((i,j)) /255.

for n in range(14000):
    img_[n] = img[num[n]]

np.savetxt('test_pmmd.txt', img_,fmt = '%.1e', delimiter=' ')   



# Output Dataset
oimg = np.zeros((14000,14),dtype = 'float32')
oimg_ = np.zeros((14000,14),dtype = 'float32')
for n in range(14000):

    #0
    if n<980: 
        oimg[n][0] = 1

    #1
    elif 980<= n and n < 2115:
        oimg[n][1] = 1

    #2
    elif 2115<= n and n < 3147:
        oimg[n][2] = 1

    #3
    elif 3147<= n and n < 4157:
        oimg[n][3] = 1

    #4
    elif 4157<= n and n < 5139:
        oimg[n][4] = 1

    #5
    elif 5139<= n and n < 6031:
        oimg[n][5] = 1

    #6
    elif 6031<= n and n < 6989:
        oimg[n][6] = 1

    #7
    elif 6989<= n and n < 8017:
        oimg[n][7] = 1

    #8
    elif 8017<= n and n < 8991:
        oimg[n][8] = 1

    #9
    elif 8991<= n and n < 10000:
        oimg[n][9] = 1

    #10
    elif 10000 <= n and n < 11000:
        oimg[n][10] = 1

    #11
    elif 11000 <= n and n < 12000:
        oimg[n][11] = 1

    #12
    elif 12000 <= n and n < 13000:
        oimg[n][12] = 1

    #13
    elif 13000 <= n and n < 14000:
        oimg[n][13] = 1

for n in range(14000):
    oimg_[n] = oimg[num[n]]

np.savetxt('test_pmmd_y.txt', oimg_, delimiter=' ')   


print(num)
print(img_)
print()
print(oimg_)
