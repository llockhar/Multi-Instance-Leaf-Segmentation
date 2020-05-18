%% This program calculates performance for an entire dataset
% Output images for each stage found in their respective Results folders
%% Choose training set
[DiC,SBD,FBD] = Training(2); % options {1,2,3}

avg_best = mean(SBD)*100;
avg_FGBG = mean(FBD)*100;
avg_numLeaf = mean(DiC);
fprintf('The symmetric best dice score is: %.2f%%\n',avg_best);
fprintf('The foreground background dice score is: %.2f%%\n',avg_FGBG);
fprintf('The absolute difference in leaf count is: %.1f\n',avg_numLeaf);

%% Function that does the training and processing of each image in dataset
function [DiC,SBD,FBD] = Training(training_set)
currentFolder = pwd;
addpath(genpath(currentFolder))
cd Full_Datasets
cd training_data
if training_set == 1
    cd Ara2012
elseif training_set == 2
    cd Ara2013-Canon
elseif training_set == 3
    cd Tobacco
end
filesTrainImage = dir('*rgb*.png');
filesTrainLabel = dir('*label*.png');
trainFolder = pwd;
cd(currentFolder);

cd Full_Datasets
cd testing_data
test = sprintf('A%d',training_set);
cd(test)
testFolder = pwd;
filesTestImage = dir('*rgb*.png');
filesTestLabel = dir('*label*.png');
cd(currentFolder);

% Extract features from training set
X = zeros(size(1,6));
for i = 1:length(filesTrainImage)
    filenameTrainImage = filesTrainImage(i).name;
    filenameTrainLabel = filesTrainLabel(i).name;
    img = imread(fullfile(trainFolder,filenameTrainImage));
    label = imread(fullfile(trainFolder,filenameTrainLabel));
    [Xtemp,ytemp] = ExtractTrainFeatures(img,label);
    if ~sum(X > 0)
        X = Xtemp;
        y = ytemp;
    else
        X = [X;Xtemp];
        y = [y;ytemp];
    end
end
disp('Training Parameter Extraction Complete!');

% Extract features from testing set
Xtest = zeros(size(1,6));
image_sizes = zeros(Xtest,2);
for i = 1:length(filesTestImage)
    filenameTestImage = filesTestImage(i).name;
    img = imread(fullfile(testFolder,filenameTestImage));
    image_sizes(i,:) = [size(img,1) size(img,2)];
    Xtemp = ExtractFeatures(img);
    if ~sum(Xtest > 0)
        Xtest = Xtemp;
    else
        Xtest = [Xtest;Xtemp];
    end
end

disp('Testing Parameter Extraction Complete!');

%% Train neural network

input_layer_size = size(X,2); % 6 features per sample
hidden_layer_size = 10; % 10 hidden units
numLabels = 2; % plant or background

initial_Theta1 = randInitializeWeights(input_layer_size,hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size,numLabels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:);initial_Theta2(:)];

% Run through 100 iterations for weight optimization
options = optimset('MaxIter', 100);

lambda = 1; % Regularization parameter

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p,input_layer_size,hidden_layer_size,...
                                   numLabels,X,y,lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params,~] = fmincg(costFunction,initial_nn_params,options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size+1)),...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1+(hidden_layer_size*(input_layer_size+1))):end),...
                 numLabels, (hidden_layer_size + 1));
             
% Test on test dataset
if training_set == 3 % too big for MATLAB, break in half
    half_val = round(size(Xtest,1)/2);
    pred1 = predict(Theta1, Theta2, Xtest(1:half_val,:));
    pred2 = predict(Theta1, Theta2, Xtest(half_val+1:end,:));
    pred = [pred1;pred2];
else
    pred = predict(Theta1, Theta2, Xtest);
end

% Image output pathway
cd Results
cd PlantNNResults
outpath = pwd;
cd(currentFolder);
cd Results
cd PlantEndResults
outpath2 = pwd;
cd(currentFolder);

start_val = 1;
folder_length = length(image_sizes);
SBD = zeros(length(image_sizes),1);
FBD = zeros(length(image_sizes),1);
DiC = zeros(length(image_sizes),1);

%% Process each image
for i = 1:folder_length
    fprintf('\nProcessing image %d of %d\n',i,folder_length);
    num_elements = image_sizes(i,1)*image_sizes(i,2);
    trainedImage = zeros(image_sizes(i,:));
    % reconstruct individual image from row vector
    trainedImage(:) = pred(start_val:(start_val+num_elements-1));
    start_val = start_val+num_elements;
    % trainedImage == 1 is foreground trainedImage == 2 is background
    trainedImage(trainedImage == 2) = 0;
    outfile = [outpath,'\TRAINED',sprintf('A%d%03d',training_set,i),'.png'];
    imwrite(trainedImage,outfile);
    
    % for training_set #1 or #3, RGB image needed for cleaning
    if training_set == 1
        filename_rgb = filesTestImage(i).name;
        rgb_img = imread(fullfile(testFolder,filename_rgb));
        cleanedImage = A1_Cleaning(trainedImage,rgb_img);
    elseif training_set == 2
        cleanedImage = A2_Cleaning(trainedImage);
    elseif training_set == 3
        filename_rgb = filesTestImage(i).name;
        rgb_img = imread(fullfile(testFolder,filename_rgb));
        cleanedImage = A3_Cleaning(trainedImage,rgb_img);
    end
    outfile = [outpath2,'\CLEANED',sprintf('A%d%03d',training_set,i),'.png'];
    imwrite(cleanedImage,outfile);
    
    % perform watershed algorithm and region merging
    wateredImage = Watershed(cleanedImage,training_set,i);
    
    % get label image for the test image and ground truth, evaluate
    inLabel = wateredImage;
    filename_label = filesTestLabel(i).name;
    gtLabel = imread(fullfile(testFolder,filename_label));
    SBD(i) = BestDice(inLabel,gtLabel);
    FBD(i) = FGBGDice(inLabel,gtLabel);
    DiC(i) = AbsDiffFGLabels(inLabel,gtLabel);
    
end

end


%% Sigmoid calculator
function g = sigmoid(z)
    g = 1 ./ (1 + exp(-z));
end

%% Sigmoid Gradient calculator
function g = sigmoidGradient(z)
    g = sigmoid(z) .* (1 - sigmoid(z));
end

%% Set random initial weights
function W = randInitializeWeights(L_in,L_out)
    INIT_EPSILON = 0.12;
    W = rand(L_out,1+L_in) * (2 * INIT_EPSILON) - INIT_EPSILON;
end

%% Calculate the neural network cost function
function [J,grad] = nnCostFunction(nn_params,input_layer_size, ...
                                   hidden_layer_size,num_labels,X,y,lambda)
% Reshape nn_params back into the parameters Theta1 and Theta2, the weight 
% matrices for our 2-layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size*(input_layer_size+1)),...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1+(hidden_layer_size*(input_layer_size+1))):end),...
                 num_labels, (hidden_layer_size + 1));

m = size(X,1);
       
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);

a1 = [ones(size(X,1),1) X];

z2 = a1 * transpose(Theta1);
a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1) a2];
z3 = a2 * transpose(Theta2);
h = sigmoid(z3);

temp = log(h) .* y_matrix + log(1 - h) .* (1 - y_matrix);
J = J + sum(temp(:));
J = -J / m + (sumsqr(Theta1(:,2:end)) + sumsqr(Theta2(:,2:end))) * ...
    lambda / (2 * m);

delta3 = h - y_matrix;
delta2 = (delta3 * Theta2(:,2:end)) .* sigmoidGradient(z2);

Delta2 = transpose(delta3) * a2;
Delta1 = transpose(delta2) * a1;

temp1 = Theta1;
temp1(:,1) = 0;
temp2 = Theta2;
temp2(:,1) = 0;

Theta1_grad = Delta1 / m + lambda * temp1 / m;
Theta2_grad = Delta2 / m + lambda * temp2 / m;

% Unroll gradients
grad = [Theta1_grad(:);Theta2_grad(:)];

end

%% Predict function
function p = predict(Theta1,Theta2,X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1)

m = size(X,1);

h1 = sigmoid([ones(m,1) X] * Theta1');
h2 = sigmoid([ones(m,1) h1] * Theta2');
[~,p] = max(h2,[],2);

end

%% Extract NN Training Set Features
function [X,y] = ExtractTrainFeatures(img,label)
% img is an RGB image, lab is a plant mask, 
% to train NN, we only need 3000 pixels max per class

plantMask = label > 0;

redChannel = img(:,:,1);
greenChannel = img(:,:,2);
blueChannel = img(:,:,3);

Wd = 3;
stdDevImage = stdfilt(greenChannel,ones(Wd));
varFiltImage = (stdDevImage .^ 2)/Wd^2;
varFiltImage = varFiltImage/max(varFiltImage(:));
varFiltImage = cast(varFiltImage,'uint8');
varImage = greenChannel - varFiltImage;
[GradMag] = imgradient(greenChannel);
GradMag = GradMag/max(GradMag(:));
GradMag = cast(GradMag,'uint8');
gradImage = greenChannel - GradMag/Wd^2;

plantMask = plantMask(:); 
redChannel = redChannel(:);
greenChannel = greenChannel(:);
blueChannel = blueChannel(:);

greenExcess = zeros(size(greenChannel)); % 2G-RB value - green2 later on

for i = 1:length(greenChannel)
    if (redChannel(i) > greenChannel(i) || blueChannel(i) > ...
            greenChannel(i))
        greenExcess(i) = greenChannel(i);
    else
        greenExcess(i) = 2 * greenChannel(i);
    end
end

if (bwarea(plantMask) > 3000) % 3000 samples max
    
    plantIndex = find(plantMask > 0);

    sampleIndex = randperm(numel(plantIndex),3000);
    plantSamples = plantIndex(sampleIndex);
    
    redPlant = redChannel(plantSamples);
    greenPlant = greenChannel(plantSamples);
    bluePlant = blueChannel(plantSamples);
    green2Plant = greenExcess(plantSamples);
    varPlant = varImage(plantSamples);
    gradPlant = gradImage(plantSamples);
    
else % use all samples of plant pixels available
    redPlant = redChannel(plantMask);
    greenPlant = greenChannel(plantMask);
    bluePlant = blueChannel(plantMask);
    green2Plant = greenExcess(plantMask);
    varPlant = varImage(plantMask);
    gradPlant = gradImage(plantMask);
end

plantInputs = [redPlant,greenPlant,bluePlant,green2Plant,varPlant,gradPlant];

if (bwarea(imcomplement(plantMask)) > 3000)
    
    backgroundIndex = find(plantMask == 0);

    sampleIndex = randperm(numel(backgroundIndex),3000);
    backgroundSamples = backgroundIndex(sampleIndex);
    
    redBack = redChannel(backgroundSamples); 
    greenBack = greenChannel(backgroundSamples);
    blueBack = blueChannel(backgroundSamples);
    green2Back = greenExcess(backgroundSamples);
    varBack = varImage(backgroundSamples);
    gradBack = gradImage(backgroundSamples);
    
else % use all samples of background pixels available
    redBack = redChannel(~plantMask);
    greenBack = greenChannel(~plantMask);
    blueBack = blueChannel(~plantMask);
    green2Back = greenExcess(~plantMask);
    varBack = varImage(~plantMask);
    gradBack = gradImage(~plantMask);
end

backgroundInputs = [redBack,greenBack,blueBack,green2Back,varBack,gradBack];

X = double([plantInputs; backgroundInputs]);
y = [ones(size(plantInputs,1),1); zeros(size(backgroundInputs,1),1)];
y(y==0) = 2;

end

%% Extract NN Testing Set Features
function X = ExtractFeatures(img)
% img is an RGB image, lab is a plant mask, 
% for the test set, all features are extracted


redChannel = img(:,:,1);
greenChannel = img(:,:,2);
blueChannel = img(:,:,3);

Wd = 3;
stdDevImage = stdfilt(greenChannel,ones(Wd));
varFiltImage = (stdDevImage .^ 2)/Wd^2;
varFiltImage = varFiltImage/max(varFiltImage(:));
varFiltImage = cast(varFiltImage,'uint8');
varImage = greenChannel - varFiltImage;
[GradMag] = imgradient(greenChannel);
GradMag = GradMag/max(GradMag(:));
GradMag = cast(GradMag,'uint8');
gradImage = greenChannel - GradMag/Wd^2;

greenExcess = zeros(size(greenChannel)); % 2G-RB value

for i = 1:length(greenChannel)
    if (redChannel(i) > greenChannel(i) || blueChannel(i) > ...
            greenChannel(i))
        greenExcess(i) = greenChannel(i);
    else
        greenExcess(i) = 2 * greenChannel(i);
    end
end

X = double([redChannel(:),greenChannel(:),blueChannel(:),greenExcess(:),varImage(:),gradImage(:)]);

end

%% Morphological cleaning function for dataset #1
function A1_out = A1_Cleaning(img,rgb_img)
% This function takes as input a binary and an rgb image and outputs a
% binary image
% Keep largest blob, YUV colour threshold, LAB colour threshold if compact,
% Large opening to get rid of moss/planter rim
img = img > 0;
stats = regionprops(img,'Area');
areas = cat(1,stats.Area);
max_area = max(areas);
Label = bwlabel(img);
Keep = find(areas == max_area); 
largest_blob = ismember(Label,Keep);
A1_temp = img.*largest_blob;
A1_temp = double(A1_temp);

stats2 = regionprops(A1_temp,'Perimeter','Area');
A = stats2.Area;
P = stats2.Perimeter;
C = P^2 / (4*pi*A);

discard = find(A1_temp == 0);

R = rgb_img(:,:,1);
G = rgb_img(:,:,2);
B = rgb_img(:,:,3);
R(discard) = 0;
G(discard) = 0;
B(discard) = 0;
A1_rgb = uint8(zeros(size(rgb_img)));

% RGB-YUV conversion values from Wikipedia
RGB2YUV = [0.299,0.587,0.114;-0.147,-0.289,0.436;0.615,-0.515,-0.100];
YUV2RGB = [1 0 1.13983; 1 -0.39465 -0.5806; 1 2.03211 0];

Y = RGB2YUV(1)*R + RGB2YUV(4)*G + RGB2YUV(7)*B;
U = RGB2YUV(2)*R + RGB2YUV(5)*G + RGB2YUV(8)*B;
V = RGB2YUV(3)*R + RGB2YUV(6)*G + RGB2YUV(9)*B;

discard = find(Y(:,:,1) < 85);

Y(discard) = 0;
U(discard) = 0;
V(discard) = 0;

R = YUV2RGB(1)*Y + YUV2RGB(4)*U + YUV2RGB(7)*V;
G = YUV2RGB(2)*Y + YUV2RGB(5)*U + YUV2RGB(8)*V;
B = YUV2RGB(3)*Y + YUV2RGB(6)*U + YUV2RGB(9)*V;

A1_rgb(:,:,1) = R;
A1_rgb(:,:,2) = G;
A1_rgb(:,:,3) = B;

if C > 20
    A1_lab = rgb2lab(A1_rgb);
    discard = find(A1_lab(:,:,1) < 45);
end

R(discard) = 0;
G(discard) = 0;
B(discard) = 0;

A1_temp = ones(size(img));
A1_temp(R==0 & G==0 & B==0) = 0;

stats = regionprops(A1_temp,'Area');
max_area = max(cat(1,stats.Area));
if ~isempty(max_area)
    A1_out1 = bwareaopen(A1_temp,ceil(0.1*max_area));
    SE = strel('disk',ceil(0.0001*max_area),4);
    A1_out = imopen(A1_out1,SE);
else
    A1_out = A1_temp;
end

A1_out = A1_out .* A1_temp;

end

%% Morphological cleaning function for dataset #2
function A2_out =  A2_Cleaning(img)
% This function takes as input a binary image and outputs a binary image
% Erosion followed by propagation, then an opening
img = img > 0;
stats = regionprops(img,'Area');
total_area = sum(cat(1,stats.Area));
SE =  strel('disk',round(0.05*sqrt(total_area)),4);
I_temp = imerode(img,SE);

SE = strel('diamond',1);

while true
    temp = imdilate(I_temp,SE);
    temp = temp .* img;
    if temp == I_temp
        break;
    end
    I_temp = temp;
end

stats = regionprops(I_temp,'Area');
total_area = sum(cat(1,stats.Area));
SE = strel('disk',round(0.02*sqrt(total_area)),4);
A2_out = imopen(I_temp,SE);

end

%% Morphological cleaning function for dataset #3
function A3_out =  A3_Cleaning(img,rgb_img)
% This function takes as input a binary and an rgb image and outputs a
% binary image
% Keep pixels within 1.5*radius from image centroid, YUV colour threshold,
% then an opening and closing to fill small holes in foreground/background
img = img > 0;
R = rgb_img(:,:,1);
G = rgb_img(:,:,2);
B = rgb_img(:,:,3);

stats = regionprops(img,'Area','Centroid');
[~,max_area] = max(cat(1,stats.Area));
centroid = stats(max_area).Centroid;
total_area = sum(cat(1,stats.Area));
radius = sqrt(total_area/pi);
[xgrid,ygrid] = meshgrid(1:size(img,2),1:size(img,1));
mask = (xgrid-centroid(1)).^2 + (ygrid-centroid(2)).^2 <= 1.5 * radius.^2;
img = img .* mask;

discard = find(img == 0);
R(discard) = 0;
G(discard) = 0;
B(discard) = 0;

% RGB-YUV conversion values from Wikipedia
RGB2YUV = [0.299,0.587,0.114;-0.147,-0.289,0.436;0.615,-0.515,-0.100];
YUV2RGB = [1 0 1.13983; 1 -0.39465 -0.5806; 1 2.03211 0];

Y = RGB2YUV(1)*R + RGB2YUV(4)*G + RGB2YUV(7)*B;
U = RGB2YUV(2)*R + RGB2YUV(5)*G + RGB2YUV(8)*B;
V = RGB2YUV(3)*R + RGB2YUV(6)*G + RGB2YUV(9)*B;

discard = find(Y < 45);

Y(discard) = 0;
U(discard) = 0;
V(discard) = 0;

R = YUV2RGB(1)*Y + YUV2RGB(4)*U + YUV2RGB(7)*V;
G = YUV2RGB(2)*Y + YUV2RGB(5)*U + YUV2RGB(8)*V;
B = YUV2RGB(3)*Y + YUV2RGB(6)*U + YUV2RGB(9)*V;

A3_temp = ones(size(img));
A3_temp(R==0 & G==0 & B==0) = 0;

stats = regionprops(A3_temp,'Area');
max_area = max(cat(1,stats.Area));
if ~isempty(max_area)
    A3_out1 = bwareaopen(A3_temp,ceil(0.1*max_area));
    SE = strel('disk',5,4);
    A3_out = imclose(A3_out1,SE);
else
    A3_out = A3_temp;
end

end

%% Watershed Algorithm Function
function test_label = Watershed(img,training_set,g)
currentFolder = pwd;
cd Results
cd WateredResults
outpath = pwd;
cd(currentFolder);

D = bwdist(imcomplement(img(:,:,1)),'cityblock');
if training_set == 3
    D = (D - mod(D,2))/2;
end
D = double(padarray(D,[1 1],nan));
imi = -D + max(D(:));
max_val = max(imi(:));
imi(imi == max_val) = nan;

MASK = -2;
WSHED = 0;
INIT = -1;
fictitious_pixel = -3;

queue = [];

imo = zeros(size(D));
imo(imo == 0) = INIT;

current_label = 0;
imd = zeros(size(D));

alldists = unique(imi(~isnan(imi)));

% Main loop, iterate through imi intensities
for h = 1:numel(alldists)
    
    h_index = find(imi == alldists(h));
        
    % Iterate through every pixel at intensity h
    for i = 1:length(h_index)
        imo(h_index(i)) = MASK;
        
        % find neighbouring pixels
        [h_subx,h_suby] = ind2sub(size(imi),h_index(i));
        nhood = [-1 -1; -1 0; -1 1; 0 -1; 0 1; 1 -1; 1 0; 1 1];
        p_nhood = 0;
        p_temp = [0 0];
        for k = 1:length(nhood)
            p_temp = [h_subx,h_suby] + nhood(k,:);
            p_nhood = sub2ind(size(imi),p_temp(1),p_temp(2));
            if imo(p_nhood) > 0 || imo(p_nhood) ==  WSHED
                imd(h_index(i)) = 1;
                queue(end+1) = h_index(i);
            end
        end
    end
    
    current_dist = 1;
    queue(end+1) = fictitious_pixel;
    
    while true
        
        p = queue(1);
        queue(1) = [];
        if p == fictitious_pixel
            if isempty(queue)
                break
            else 
                queue(end+1) = fictitious_pixel;
                current_dist = current_dist + 1;
                p = queue(1);
                queue(1) = [];
            end
        end
        
        % find neighbouring pixels
        [p_subx,p_suby] = ind2sub(size(imi),p);
        p_nhood = 0;
        p_temp = [0 0];
        for i = 1:length(nhood)
            p_temp = [p_subx,p_suby] + nhood(i,:);
            p_nhood = sub2ind(size(imi),p_temp(1),p_temp(2));
            if imd(p_nhood) < current_dist && (imo(p_nhood) > 0 ||...
                    imo(p_nhood) == WSHED) % part of closer basin or WSHED
                if imo(p_nhood) > 0 % neighbour part of a closer basin
                    if imo(p) == MASK || imo(p) == WSHED % pixel is unset
                        % put middle pixel in neighbour's basin
                        imo(p) = imo(p_nhood);
                    elseif imo(p) ~= imo(p_nhood) 
                        % middle pixel in another basin from neighbour
                        imo(p) = WSHED;
                    end
                elseif imo(p) == MASK 
                    % neighbour is part of WSHED with unset middle pixel
                    imo(p) = WSHED;
                end
            elseif imo(p_nhood) == MASK && imd(p_nhood) == 0 % unset
                % neighbour is further from minima and later evaluated
                imd(p_nhood) = current_dist + 1;
                queue(end+1) = p_nhood;
            end
        end
    end
        
    % Check if new minima have been discovered
    for i = 1:length(h_index)
        imd(h_index(i)) = 0;
        if imo(h_index(i)) == MASK % is not part of a basin or WSHED
            current_label = current_label + 1;
            queue(end+1) = h_index(i);
            imo(h_index(i)) = current_label;
            while ~isempty(queue)
                p2 = queue(1);
                queue(1) = [];
                % find neighbouring pixels
                [p2_subx,p2_suby] = ind2sub(size(imi),p2);
                p2_nhood = 0;
                p2_temp = [0 0];
                for j = 1:length(nhood) 
                    % give neighbouring pixels new label
                    p2_temp = [p2_subx,p2_suby] + nhood(j,:);
                    p2_nhood = sub2ind(size(imi),p2_temp(1),p2_temp(2));
                    if imo(p2_nhood) == MASK
                      queue(end+1) = p2_nhood;  
                      imo(p2_nhood) = current_label;
                    end
                end
            end
        end
    end
end

imo(1,:) = [];
imo(end,:) = [];
imo(:,1) = [];
imo(:,end) = [];

imo_temp = imo;
imo_temp(imo_temp<0) = 0;
out_colour = label2rgb(imo_temp,'jet',[0 0 0]);
outfile = [outpath,'\WATERED',sprintf('A%d%03d',training_set,g),'.png'];
imwrite(uint8(out_colour),outfile);

test_label = Merge_Basins(training_set,current_label,imo,g);

end

%% Region Merging Function
function test_label = Merge_Basins(training_set,current_label,imo,g)

% Set output path for binary output images
currentFolder = pwd;
cd Results
cd FinalResults
outpath = pwd;
cd(currentFolder);

if training_set == 1
    thresh = 30; % weak boundary threshold
    level = 0.2; % relative size threshold
elseif training_set == 2
    thresh = 58;
    level = 0.2;
elseif training_set == 3
    thresh = 70;
    level = 0.2;
end

% The basins are successively merged if they are separated by a watershed 
% that is smaller than a given threshold (30, 58, and 70 for the datasets 
% ‘A1’, ‘A2’, and ‘A3’ respectively). 
% Also consider relative region perimeters to avoid over-merging

if training_set == 3 && current_label > 10
    level = 0.5;
end
    
    for l = 1:current_label
    
        mask = zeros(size(imo));
        mask(imo == l) = 1;
        stats = regionprops(mask,'Perimeter');
        if isempty(stats)
            continue;
        else
            l_perim = stats.Perimeter;
        end
        SE = strel('diamond',1);
        mask = imdilate(mask,SE);
        mask2 = imdilate(mask,SE);
        mask2(mask == 1) = 0;
        mask(imo == 1) = 0;

        rgn_bnd = find(mask == 1);
        region_adj = imo(mask2 == 1);
        region_adj(region_adj <= 0) = [];
        rgn_adj = unique(region_adj);

        for m = 1:length(rgn_adj)
            mask = zeros(size(imo));
            mask(imo == rgn_adj(m)) = 1;
            stats2 = regionprops(mask,'Perimeter');
            m_perim = stats2.Perimeter;

            mask2 = imdilate(mask,SE);
            mask2(mask == 1) = 0;
            rgn_bnd2 = find(mask2 == 1);

            F = length(intersect(rgn_bnd,rgn_bnd2));
            if F < thresh && (l_perim < level*m_perim || m_perim < level*l_perim)
                label_min = min([l;rgn_adj(m)]);
                label_max = max([l;rgn_adj(m)]);
                imo(imo == label_max) = label_min;
                imo(intersect(rgn_bnd,rgn_bnd2)) = label_min;
            end
        end
    end

    imo(imo<0) = 0;
    test_label = imo;
    out_colour = label2rgb(imo,'jet',[0 0 0]);
    outfile = [outpath,'\MERGED',sprintf('A%d%03d',training_set,g),'.png'];
    imwrite(uint8(out_colour),outfile);
end