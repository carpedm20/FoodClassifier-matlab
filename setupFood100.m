function imdb = setupFood100(datasetDir)
% SETUPCALTECH256    Setup Caltech 256 and 101 datasets

opts.lite = false ;
opts.numTrain = 5 ;
opts.numTest = 5 ;
opts.seed = 1 ;
opts.variant = 'food100' ;
opts.autoDownload = true ;

numClasses = 100 ;

% Read classes
imdb = setupGeneric(datasetDir, ...
  'numTrain', opts.numTrain, 'numVal', 0, 'numTest', opts.numTest,  ...
  'expectedNumClasses', numClasses, ...
  'seed', opts.seed, 'lite', opts.lite) ;

