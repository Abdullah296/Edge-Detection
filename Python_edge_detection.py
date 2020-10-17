import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
from skimage import io
from operator import truediv
import numpy.matlib

img = io.imread('house.jpg', as_gray=True)
img = img*255

plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.show()

imgplot = plt.imshow(img)

Sobelx = [[1, 0, 1],[ 1, 0, -1], [1, 0, -1]]
Sobely = [[1, 2, 1],[ 0, 0, 0], [-1, -2, -1]]       # sobel

#Sobelx = [[-1, 0, 1],[ -2, 0, 2], [-1, 0, 1]]         # canny
#Sobely = [[-1, -2, -1],[ 0, 0, 0], [1, 2, 1]]

#Sobelx = [[1, 0, -1],[ 1, 0, -1], [1, 0, -1]]        # prewitt
#Sobely = [[1, 1, 1],[ 0, 0, 0], [-1, -1, -1]]

I_Size = np.array(np.shape(img))
S_X_Size = np.array(np.shape(Sobelx))

M = I_Size // S_X_Size
N_Dimension = 3*M

N_image= np.zeros((N_Dimension[0]+2, N_Dimension[1]+2))
N_image[0:I_Size[0], 0:I_Size[1]] = img

#Image1 = img[0:N_Dimension[0], 0:N_Dimension[1]]
Sobel_kernel_X = np.matlib.repmat(Sobelx, M[0], M[1])
Sobel_kernel_Y = np.matlib.repmat(Sobely, M[0], M[1])

Image1 = N_image[0:N_Dimension[0], 0:N_Dimension[1]]
Image2 = N_image[0:N_Dimension[0], 1:N_Dimension[1]+1]
Image3 = N_image[0:N_Dimension[0], 2:N_Dimension[1]+2]
Image4 = N_image[1:N_Dimension[0]+1, 0:N_Dimension[1]]
Image5 = N_image[2:N_Dimension[0]+2, 0:N_Dimension[1]]

Mul_1_1 = np.multiply(Image1, Sobel_kernel_X)
Mul_2_1 = np.multiply(Image1, Sobel_kernel_Y)

Mul_1_2 = np.multiply(Image2, Sobel_kernel_X)
Mul_2_2 = np.multiply(Image2, Sobel_kernel_Y)

Mul_1_3 = np.multiply(Image3, Sobel_kernel_X)
Mul_2_3 = np.multiply(Image3, Sobel_kernel_Y)

Mul_1_4 = np.multiply(Image4, Sobel_kernel_X)
Mul_2_4 = np.multiply(Image4, Sobel_kernel_Y)

Mul_1_5 = np.multiply(Image5, Sobel_kernel_X)
Mul_2_5 = np.multiply(Image5, Sobel_kernel_Y)

tempXX1 = np.zeros((N_Dimension[0], N_Dimension[1]))
tempXX2 = np.zeros((N_Dimension[0], N_Dimension[1]))

tempYY1 = np.zeros((N_Dimension[0], N_Dimension[1]))
tempYY2 = np.zeros((N_Dimension[0], N_Dimension[1]))

ImageX= np.zeros((N_Dimension[0]+2, N_Dimension[1]+2))
ImageY= np.zeros((N_Dimension[0]+2, N_Dimension[1]+2))
#///////////////////////////////////////////#
for i in range(0,N_Dimension[1]-2, 3):
    tempXX1[:,i+1]=Mul_1_1[:,i]+Mul_1_1[:,i+1]+Mul_1_1[:,i+2];    

for i in  range(0,N_Dimension[0]-2, 3):
    tempXX2[i+1,:]=tempXX1[i,:]+tempXX1[i+1,:]+tempXX1[i+2,:];


for i in range(0,N_Dimension[1]-2, 3):
    tempYY1[:,i+1]=Mul_2_1[:,i]+Mul_2_1[:,i+1]+Mul_2_1[:,i+2];    

for i in  range(0,N_Dimension[0]-2, 3):
    tempYY2[i+1,:]=tempYY1[i,:]+tempYY1[i+1,:]+tempYY1[i+2,:];
    
    ImageX[0:N_Dimension[0], 0:N_Dimension[1]] = tempXX2 + ImageX[0:N_Dimension[0], 0:N_Dimension[1]]
    ImageY[0:N_Dimension[0], 0:N_Dimension[1]] = tempYY2 + ImageY[0:N_Dimension[0], 0:N_Dimension[1]]
#///////////////////////////////////////////#
    
for i in range(0,N_Dimension[1]-2, 3):
    tempXX1[:,i+1]=Mul_1_2[:,i]+Mul_1_2[:,i+1]+Mul_1_2[:,i+2];    

for i in  range(0,N_Dimension[0]-2, 3):
    tempXX2[i+1,:]=tempXX1[i,:]+tempXX1[i+1,:]+tempXX1[i+2,:];


for i in range(0,N_Dimension[1]-2, 3):
    tempYY1[:,i+1]=Mul_2_2[:,i]+Mul_2_2[:,i+1]+Mul_2_2[:,i+2];    

for i in  range(0,N_Dimension[0]-2, 3):
    tempYY2[i+1,:]=tempYY1[i,:]+tempYY1[i+1,:]+tempYY1[i+2,:];
    
    ImageX[0:N_Dimension[0], 1:N_Dimension[1]+1] = tempXX2 + ImageX[0:N_Dimension[0], 1:N_Dimension[1]+1]
    ImageY[0:N_Dimension[0], 1:N_Dimension[1]+1] = tempYY2 + ImageY[0:N_Dimension[0], 1:N_Dimension[1]+1] 
#///////////////////////////////////////////#
for i in range(0,N_Dimension[1]-2, 3):
    tempXX1[:,i+1]=Mul_1_3[:,i]+Mul_1_3[:,i+1]+Mul_1_3[:,i+2];    

for i in  range(0,N_Dimension[0]-2, 3):
    tempXX2[i+1,:]=tempXX1[i,:]+tempXX1[i+1,:]+tempXX1[i+2,:];


for i in range(0,N_Dimension[1]-2, 3):
    tempYY1[:,i+1]=Mul_2_3[:,i]+Mul_2_3[:,i+1]+Mul_2_3[:,i+2];    

for i in  range(0,N_Dimension[0]-2, 3):
    tempYY2[i+1,:]=tempYY1[i,:]+tempYY1[i+1,:]+tempYY1[i+2,:];
    
    ImageX[0:N_Dimension[0], 2:N_Dimension[1]+2] = tempXX2 + ImageX[0:N_Dimension[0], 2:N_Dimension[1]+2]
    ImageY[0:N_Dimension[0], 2:N_Dimension[1]+2] = tempYY2 + ImageY[0:N_Dimension[0], 2:N_Dimension[1]+2]
#///////////////////////////////////////////#
for i in range(0,N_Dimension[1]-2, 3):
    tempXX1[:,i+1]=Mul_1_4[:,i]+Mul_1_4[:,i+1]+Mul_1_4[:,i+2];    

for i in  range(0,N_Dimension[0]-2, 3):
    tempXX2[i+1,:]=tempXX1[i,:]+tempXX1[i+1,:]+tempXX1[i+2,:];


for i in range(0,N_Dimension[1]-2, 3):
    tempYY1[:,i+1]=Mul_2_4[:,i]+Mul_2_4[:,i+1]+Mul_2_4[:,i+2];    

for i in  range(0,N_Dimension[0]-2, 3):
    tempYY2[i+1,:]=tempYY1[i,:]+tempYY1[i+1,:]+tempYY1[i+2,:];
    
    ImageX[1:N_Dimension[0]+1, 0:N_Dimension[1]] = tempXX2 + ImageX[1:N_Dimension[0]+1, 0:N_Dimension[1]]
    ImageY[1:N_Dimension[0]+1, 0:N_Dimension[1]] = tempYY2 + ImageY[1:N_Dimension[0]+1, 0:N_Dimension[1]] 
#///////////////////////////////////////////#
for i in range(0,N_Dimension[1]-2, 3):
    tempXX1[:,i+1]=Mul_1_5[:,i]+Mul_1_5[:,i+1]+Mul_1_5[:,i+2];    

for i in  range(0,N_Dimension[0]-2, 3):
    tempXX2[i+1,:]=tempXX1[i,:]+tempXX1[i+1,:]+tempXX1[i+2,:];


for i in range(0,N_Dimension[1]-2, 3):
    tempYY1[:,i+1]=Mul_2_5[:,i]+Mul_2_5[:,i+1]+Mul_2_5[:,i+2];    

for i in  range(0,N_Dimension[0]-2, 3):
    tempYY2[i+1,:]=tempYY1[i,:]+tempYY1[i+1,:]+tempYY1[i+2,:];
    
    ImageX[2:N_Dimension[0]+2, 0:N_Dimension[1]] = tempXX2 + ImageX[2:N_Dimension[0]+2, 0:N_Dimension[1]]
    ImageY[2:N_Dimension[0]+2, 0:N_Dimension[1]] = tempYY2 + ImageY[2:N_Dimension[0]+2, 0:N_Dimension[1]] 
#///////////////////////////////////////////#    
ImageA = ImageX + ImageY
    
#imgplot = plt.imshow(ImageA)
plt.imshow(ImageA, cmap='gray', vmin=0, vmax=255)
plt.show()