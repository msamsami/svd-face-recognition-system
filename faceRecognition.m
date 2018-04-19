clc;
close all;
clear;

%-------------------STEP 1-------------------
I =  imread(['orl_faces\s' int2str(1) '\' int2str(1) '.pgm']);
I = im2double(I);
figure, imshow (I);
title ('First train picture of first folder');

% Because the size of all of the images are the same, we just use 
% the size(I) once and do not repeat it for next steps.
[r, c] = size(I);
X = ['Matrix I has ', num2str(r), ' rows and ', num2str(c), ' columns.'];
disp (X);

% Vectorizing the images and making the 10304 by 200 train matrix
TrainI = [];
for i = 1:40
    for j = 1:5
        I = im2double(imread(['orl_faces\s' int2str(i) '\' int2str(j) '.pgm']));
        I = reshape (I, r*c ,1);
        TrainI = [TrainI, I];
    end
end
%-------------------STEP 1-------------------


%-------------------STEP 2-------------------
% Normalizing the training data set
D = TrainI - mean(TrainI);
%-------------------STEP 2-------------------


%-------------------STEP 3-------------------
% Making and normalizing the the 10304 by 200 test matrix
TestI = [];
for i = 1:40
    for j = 6:10
        I = im2double(imread(['orl_faces\s' int2str(i) '\' int2str(j) '.pgm']));
        I = reshape (I, r*c ,1);
        TestI = [TestI, I];
    end
end
NormalTest = TestI - mean(TestI);
%-------------------STEP 3-------------------



%-------------------STEP 4-------------------
for i = 1:200
    d = NormalTest(:, i) - D;
    for j = 1:200
       norms(j, 1) = norm(d(:, j));
    end
    [minNorm, indMin] = min(norms);
    Indexes(i, 1) = indMin;
end

% Now we should see how much wrong recognition we have
errors = 0;
index = 5;

for i = 1:200
   if Indexes(i, 1) > index || Indexes (i, 1) < index - 4
       errors = errors + 1;
   end   
   if mod (i, 5) == 0 
       index = index + 5;
   end
end

X = ['There were ', num2str(errors), ' wrong recognitions using method 1.'];
disp (X);
%-------------------STEP 4-------------------


%-------------------STEP 5-------------------
[U, S, V] = svd (D);
%-------------------STEP 5-------------------


%-------------------STEP 6-------------------
figure
sValues = diag(S);
plot (sValues);
title ('Singular Values');
X = ['Largest singular value: ', num2str(max(sValues)), ', Smallest' ...
     ' singular value: ', num2str(min(sValues))];
disp (X);
%-------------------STEP 6-------------------


%-------------------STEP 7-------------------
%for i = 50:60
    %figure
    %imshow (reshape (U(:,i), r, c), []);
%end
%-------------------STEP 7-------------------


%-------------------STEP 8-------------------
NK = 10;  % Number of basis vectors we want to keep
UKEEP = U (:, 1:NK);
PROJD = (D'*UKEEP)';
%-------------------STEP 8-------------------


%-------------------STEP 9-------------------
PROJT = (NormalTest'*UKEEP)';
%-------------------STEP 9-------------------


%-------------------STEP 10------------------
norms = [];
Indexes = [];
for i = 1:200 
    d = PROJT(:, i) - PROJD;
    for j = 1:200
       norms(j, 1) = norm(d(:, j));
    end
    [minNorm, indMin] = min(norms);
    Indexes(i, 1) = indMin;
end

% Now we should see how much wrong recognition we have
errors = 0;
index = 5;
for i = 1:200
   if Indexes (i, 1) > index || Indexes (i, 1) < index - 4
       errors = errors + 1;
   end 
   if mod (i, 5) == 0 
       index = index + 5;
   end
end

X = ['There were ', num2str(errors), ' wrong recognitions using method 2' ...
    ' (NK=', num2str(NK), ').'];
disp (X);
%-------------------STEP 10------------------


%-------------------STEP 11------------------
for k = 1:200
    UKEEP = [];
    PROJD = [];
    PROJT = [];
    
    NK = k;
    UKEEP = U (:, 1:NK);
    PROJD = (D'*UKEEP)';
    PROJT = (NormalTest'*UKEEP)';

    norms = [];
    Indexes = [];
    
    for i = 1:200 
        d = PROJT(:, i) - PROJD;
        for j = 1:200
           norms(j, 1) = norm(d(:, j));
        end
        [minNorm, indMin] = min(norms);
        Indexes(i, 1) = indMin;
    end

    
    errors = 0;
    index = 5;
    for i = 1:200
       if Indexes (i, 1) > index || Indexes (i, 1) < index - 4
           errors = errors + 1;
       end
       if mod (i, 5) == 0 
           index = index + 5;
       end
    end
    
    errorsNK(k, 1) = errors;
    
end

figure
plot (errorsNK);
title ('Number of errors vs NK');
%-------------------STEP 11------------------

