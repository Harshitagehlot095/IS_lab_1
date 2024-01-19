%Classification using perceptron

%Reading apple images
A1=imread('apple_04.jpg');
A2=imread('apple_05.jpg');
A3=imread('apple_06.jpg');
A4=imread('apple_07.jpg');
A5=imread('apple_11.jpg');
A6=imread('apple_12.jpg');
A7=imread('apple_13.jpg');
A8=imread('apple_17.jpg');
A9=imread('apple_19.jpg');

%Reading pears images
P1=imread('pear_01.jpg');
P2=imread('pear_02.jpg');
P3=imread('pear_03.jpg');
P4=imread('pear_09.jpg');

%Calculate for each image, colour and roundness
%For Apples
%1st apple image(A1)
hsv_value_A1=spalva_color(A1); %color
metric_A1=apvalumas_roundness(A1); %roundness
%2nd apple image(A2)
hsv_value_A2=spalva_color(A2); %color
metric_A2=apvalumas_roundness(A2); %roundness
%3rd apple image(A3)
hsv_value_A3=spalva_color(A3); %color
metric_A3=apvalumas_roundness(A3); %roundness
%4th apple image(A4)
hsv_value_A4=spalva_color(A4); %color
metric_A4=apvalumas_roundness(A4); %roundness
%5th apple image(A5)
hsv_value_A5=spalva_color(A5); %color
metric_A5=apvalumas_roundness(A5); %roundness
%6th apple image(A6)
hsv_value_A6=spalva_color(A6); %color
metric_A6=apvalumas_roundness(A6); %roundness
%7th apple image(A7)
hsv_value_A7=spalva_color(A7); %color
metric_A7=apvalumas_roundness(A7); %roundness
%8th apple image(A8)
hsv_value_A8=spalva_color(A8); %color
metric_A8=apvalumas_roundness(A8); %roundness
%9th apple image(A9)
hsv_value_A9=spalva_color(A9); %color
metric_A9=apvalumas_roundness(A9); %roundness

%For Pears
%1st pear image(P1)
hsv_value_P1=spalva_color(P1); %color
metric_P1=apvalumas_roundness(P1); %roundness
%2nd pear image(P2)
hsv_value_P2=spalva_color(P2); %color
metric_P2=apvalumas_roundness(P2); %roundness
%3rd pear image(P3)
hsv_value_P3=spalva_color(P3); %color
metric_P3=apvalumas_roundness(P3); %roundness
%2nd pear image(P4)
hsv_value_P4=spalva_color(P4); %color
metric_P4=apvalumas_roundness(P4); %roundness

%selecting features(color, roundness, 3 apples and 2 pears)
%A1,A2,A3,P1,P2
%building matrix 2x5
X1=[hsv_value_A1 hsv_value_A2 hsv_value_A3 hsv_value_P1 hsv_value_P2];
X2=[metric_A1 metric_A2 metric_A3 metric_P1 metric_P2];
% estimated features are stored in matrix P:
P=[X1;X2];

X1 = [0.21835  0.14115  0.37022  0.31565  0.36484];

X2 = [0.81884  0.83535  0.8111   0.83101  0.8518];

%Desired output vector
T=[1  1  1  -1  -1];

%% train single perceptron with two inputs and one output

% generate random initial values of w1, w2 and b
w1 = randn(1);
w2 = randn(1);
b = randn(1);

% Calculate the weighted sum

V1 = X1(1)*W1 + X2(1)*W2 + b;


% Apply Activation Function

if V1>0
     Y = 1;
else
     Y = -1;
end

% Compare with the desired answer

e1 = T(1)-Y;

%==================================================

% Calculate the weighted sum

V2 = X1(2)*W1 + X2(2)*W2 + b;

% Apply Activation Function

if V2>0
     Y = 1;
else
     Y = -1;
end

% Compare with the desired answer

e2 = T(2)-Y;

%==================================================

% Calculate the weighted sum

V3 = X1(3)*W1 + X2(3)*W2 + b;

% Apply Activation Function

if V3>0
     Y = 1;
else
     Y = -1;
end

% Compare with the desired answer

e3 = T(3)-Y;

%==================================================

% Calculate the weighted sum

V4 = X1(4)*W1 + X2(4)*W2 + b;

% Apply Activation Function

if V4>0
     Y = 1;
else
     Y = -1;
end

% Compare with the desired answer

e4 = T(4)-Y;

%==================================================

% Calculate the weighted sum

V5 = X1(5)*W1 + X2(5)*W2 + b;

% Apply Activation Function

if V5>0
     Y = 1;
    else
     Y = -1;
end


% Compare with the desired answer

e5 = T(5)-Y;

%==================================================


% calculate the total error for these 5 inputs 
e = abs(e1) + abs(e2) + abs(e3) + abs(e4) + abs(e5);


% UPDATE COEFFICIENTS

Learning_rate = 0.20;

W1 = W1+Learning_rate*e1*X1(1);
W2 = W2+Learning_rate*e1*X2(1);
b = b+Learning_rate*e1;

W1 = W1+Learning_rate*e2*X1(2);
W2 = W2+Learning_rate*e2*X2(2);
b = b+Learning_rate*e2;

W1 = W1+Learning_rate*e3*X1(3);
W2 = W2+Learning_rate*e3*X2(3);
b = b+Learning_rate*e3;

W1 = W1+Learning_rate*e4*X1(4);
W2 = W2+Learning_rate*e4*X2(4);
b = b+Learning_rate*e4;

W1 = W1+Learning_rate*e5*X1(5);
W2 = W2+Learning_rate*e5*X2(5);
b = b+Learning_rate*e5;



