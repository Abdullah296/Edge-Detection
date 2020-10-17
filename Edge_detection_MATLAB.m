Image = imread('house.jpg');
Image = rgb2gray(Image);
%imshow(Image);
                                                                                 
%Sobelx = [-1,0,1;-2,0,2;-1,0,1];   %sobel
%Sobely=[1 2 1;0 0 0;-1 -2 -1];

Sobelx = [-1,0,1;-2,0,2;-1,0,1];   %canny
Sobely = [-1 -2 -1;0 0 0;1 2 1];
                                                                                 
%Sobelx = [1 0 -1;1 0 -1;1 0 -1];  % prewitt
%Sobely=[1 1 1;0 0 0;-1 -1 -1];

I_Size = size(Image); 
S_X_Size = size(Sobelx);

M_3 = floor(I_Size./S_X_Size);
N_Dimension = (3*M_3);
X = zeros(N_Dimension+2);
Image = double(Image(1:N_Dimension(1), 1:N_Dimension(2)));
X(1:N_Dimension(1),1:N_Dimension(2)) = Image;

Sobel_kernal_X = repmat(Sobelx, M_3(1), M_3(2));
Sobel_kernal_Y = repmat(Sobely, M_3(1), M_3(2));

Image1 = double(X(1:N_Dimension(1), 1:N_Dimension(2)));
Image2 = double(X(1:N_Dimension(1), 2:N_Dimension(2)+1));
Image3 = double(X(1:N_Dimension(1), 3:N_Dimension(2)+2));
Image4 = double(X(2:N_Dimension(1)+1, 1:N_Dimension(2)));
Image5 = double(X(3:N_Dimension(1)+2, 1:N_Dimension(2)));

Mul_1_1 = Image1 .* Sobel_kernal_X;
Mul_2_1 = Image1 .* Sobel_kernal_Y;

Mul_1_2 = Image2 .* Sobel_kernal_X;
Mul_2_2 = Image2 .* Sobel_kernal_Y;

Mul_1_3 = Image3 .* Sobel_kernal_X;
Mul_2_3 = Image3 .* Sobel_kernal_Y;

Mul_1_4 = Image4 .* Sobel_kernal_X;
Mul_2_4 = Image4 .* Sobel_kernal_Y;

Mul_1_5 = Image5 .* Sobel_kernal_X;
Mul_2_5 = Image5 .* Sobel_kernal_Y;

%###################################################################%

%###################################################################%
tempXX1 = zeros(N_Dimension);
tempXX2 = zeros(N_Dimension);
tempYY1 = zeros(N_Dimension);
tempYY2 = zeros(N_Dimension);
image11x = zeros(N_Dimension+2);
image11y = zeros(N_Dimension+2);

   %######################################################################################### 
for i=1:3:(N_Dimension(2))
    tempXX1(:,i+1)=Mul_1_1(:,i)+Mul_1_1(:,i+1)+Mul_1_1(:,i+2);    
end

for i=1:3:(N_Dimension(1)-2)
    tempXX2(i+1,:)=tempXX1(i,:)+tempXX1(i+1,:)+tempXX1(i+2,:);
end

for i=1:3:(N_Dimension(2)-2)
    tempYY1(:,i+1)=Mul_2_1(:,i)+Mul_2_1(:,i+1)+Mul_2_1(:,i+2);    
end

for i=1:3:(N_Dimension(1)-2)
    tempYY2(i+1,:)=tempYY1(i,:)+tempYY1(i+1,:)+tempYY1(i+2,:);
end
image11x(1:N_Dimension(1),1:N_Dimension(2)) = image11x(1:N_Dimension(1),1:N_Dimension(2))+tempXX2;
image11y(1:N_Dimension(1),1:N_Dimension(2)) = image11y(1:N_Dimension(1),1:N_Dimension(2))+ tempYY2;
%{#########################################################################################
for i=1:3:(N_Dimension(2)-2)
    tempXX1(:,i+1)=Mul_1_2(:,i)+Mul_1_2(:,i+1)+Mul_1_2(:,i+2);    
end

for i=1:1:(N_Dimension(1)-2)
    tempXX2(i+1,:)=tempXX1(i,:)+tempXX1(i+1,:)+tempXX1(i+2,:);
end

for i=1:1:(N_Dimension(2)-2)
    tempYY1(:,i+1)=Mul_2_2(:,i)+Mul_2_2(:,i+1)+Mul_2_2(:,i+2);    
end

for i=1:1:(N_Dimension(1)-2)
    tempYY2(i+1,:)=tempYY1(i,:)+tempYY1(i+1,:)+tempYY1(i+2,:);
end

image11x(1:N_Dimension(1),2:N_Dimension(2)+1) = image11x(1:N_Dimension(1),2:N_Dimension(2)+1)+ tempXX2;
image11y(1:N_Dimension(1),2:N_Dimension(2)+1) = image11y(1:N_Dimension(1),2:N_Dimension(2)+1)+ tempYY2;
%#########################################################################################
for i=1:3:(N_Dimension(2)-2)
    tempXX1(:,i+1)=Mul_1_3(:,i)+Mul_1_3(:,i+1)+Mul_1_3(:,i+2);    
end

for i=1:1:(N_Dimension(1)-2)
    tempXX2(i+1,:)=tempXX1(i,:)+tempXX1(i+1,:)+tempXX1(i+2,:);
end

for i=1:1:(N_Dimension(2)-2)
    tempYY1(:,i+1)=Mul_2_3(:,i)+Mul_2_3(:,i+1)+Mul_2_3(:,i+2);    
end

for i=1:1:(N_Dimension(1)-2)
    tempYY2(i+1,:)=tempYY1(i,:)+tempYY1(i+1,:)+tempYY1(i+2,:);
end

image11x(1:N_Dimension(1),3:N_Dimension(2)+ 2) = image11x(1:N_Dimension(1),3:N_Dimension(2)+ 2)+tempXX2;
image11y(1:N_Dimension(1),3:N_Dimension(2)+ 2) = image11y(1:N_Dimension(1),3:N_Dimension(2)+ 2)+tempYY2;
%#########################################################################################
for i=1:3:(N_Dimension(2)-2)
    tempXX1(:,i+1)=Mul_1_4(:,i)+Mul_1_4(:,i+1)+Mul_1_4(:,i+2);    
end

for i=1:1:(N_Dimension(1)-2)
    tempXX2(i+1,:)=tempXX1(i,:)+tempXX1(i+1,:)+tempXX1(i+2,:);
end

for i=1:1:(N_Dimension(2)-2)
    tempYY1(:,i+1)=Mul_2_4(:,i)+Mul_2_4(:,i+1)+Mul_2_4(:,i+2);    
end

for i=1:1:(N_Dimension(1)-2)
    tempYY2(i+1,:)=tempYY1(i,:)+tempYY1(i+1,:)+tempYY1(i+2,:);
end

image11x(2:N_Dimension(1)+1,1:N_Dimension(2)) = image11x(2:N_Dimension(1)+1,1:N_Dimension(2))+tempXX2;
image11y(2:N_Dimension(1)+1,1:N_Dimension(2)) = image11y(2:N_Dimension(1)+1,1:N_Dimension(2))+tempYY2;
%#########################################################################################
for i=1:3:(N_Dimension(2)-2)
    tempXX1(:,i+1)=Mul_1_5(:,i)+Mul_1_5(:,i+1)+Mul_1_5(:,i+2);    
end

for i=1:1:(N_Dimension(1)-2)
    tempXX2(i+1,:)=tempXX1(i,:)+tempXX1(i+1,:)+tempXX1(i+2,:);
end

for i=1:1:(N_Dimension(2)-2)
    tempYY1(:,i+1)=Mul_2_5(:,i)+Mul_2_5(:,i+1)+Mul_2_5(:,i+2);    
end

for i=1:1:(N_Dimension(1)-2)
    tempYY2(i+1,:)=tempYY1(i,:)+tempYY1(i+1,:)+tempYY1(i+2,:);
end

image11x(3:N_Dimension(1)+2,1:N_Dimension(2)) =image11x(3:N_Dimension(1)+2,1:N_Dimension(2))+ tempXX2;
image11y(3:N_Dimension(1)+2,1:N_Dimension(2)) =image11y(3:N_Dimension(1)+2,1:N_Dimension(2)) + tempYY2;
%}%#########################################################################################
temp_image = image11x + image11y;

figure;
imshow(uint8(Image))
figure;
imshow(uint8(temp_image))


