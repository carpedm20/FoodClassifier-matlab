function foodduk()
conf.calDir = 'data/food100' ;
conf.dataDir = 'data/';
conf.numClasses = 10 ;
conf.numWords = 1000 ;
conf.numSpatialX = [2 4] ;
conf.numSpatialY = [2 4] ;
conf.quantizer = 'kdtree' ;
conf.svm.C = 10 ;
conf.phowOpts = {'Step', 3} ;

conf.svm.biasMultiplier = 1 ;
conf.prefix = 'baseline' ;
conf.randSeed = 1 ;

conf.vocabPath = fullfile(conf.dataDir, [conf.prefix '-vocab.mat']) ;
conf.histPath = fullfile(conf.dataDir, [conf.prefix '-hists.mat']) ;
conf.modelPath = fullfile(conf.dataDir, [conf.prefix '-model.mat']) ;
conf.resultPath = fullfile(conf.dataDir, [conf.prefix '-result']) ;

vl_twister('state',conf.randSeed) ;

classes = dir(conf.calDir) ;
classes = classes([classes.isdir]) ;
classes = {classes(3:conf.numClasses+2).name} ;

images = {} ;
imageClass = {} ;

selTrainImages = {};
selTestImages = {};

for ci = 1:length(classes)
  ims = dir(fullfile(conf.calDir, classes{ci}, 'crop_*.jpg'))' ;
  
  numImagesForClass = size(ims,2);
  numTest = floor(numImagesForClass/4);
  numTrain = numImagesForClass - numTest;
  
  ims = vl_colsubset(ims, numImagesForClass) ;
  ims = cellfun(@(x)fullfile(classes{ci},x),{ims.name},'UniformOutput',false) ;
  
  selTrainImages = {selTrainImages{:}, ims{1:numTrain}};
  selTestImages = {selTestImages{:}, ims{1:numTest}};
  
  images = {images{:}, ims{:}} ;
  imageClass{end+1} = ci * ones(1,length(ims)) ;
end

selTrain = cellfun(@(x)find(ismember(images,x)),{selTrainImages},'UniformOutput', false) ;
selTest = cellfun(@(x)find(ismember(images,x)),{selTestImages},'UniformOutput', false) ;

selTrain = selTrain{1:1} ;
selTest = selTest{1:1} ;

imageClass = cat(2, imageClass{:}) ;

model.classes = classes ;
model.phowOpts = conf.phowOpts ;
model.numSpatialX = conf.numSpatialX ;
model.numSpatialY = conf.numSpatialY ;
model.quantizer = conf.quantizer ;
model.vocab = [] ;
model.w = [] ;
model.b = [] ;
model.classify = @classify ;

if ~exist(conf.vocabPath)

  selTrainFeats = vl_colsubset(selTrain, 30) ;
  descrs = {} ;
  %for ii = 1:length(selTrainFeats)
  parfor ii = 1:length(selTrainFeats)
    im = imread(fullfile(conf.calDir, images{selTrainFeats(ii)})) ;
    im = standarizeImage(im) ;
    [drop, descrs{ii}] = vl_phow(im, model.phowOpts{:}) ;
  end

  descrs = vl_colsubset(cat(2, descrs{:}), 10e4) ;
  descrs = single(descrs) ;

  % Quantize the descriptors to get the visual words
  vocab = vl_kmeans(descrs, conf.numWords, 'verbose', 'algorithm', 'elkan', 'MaxNumIterations', 100) ;
  save(conf.vocabPath, 'vocab') ;
else
  load(conf.vocabPath) ;
end



% -------------------------------------------------------------------------
function im = standarizeImage(im)
% -------------------------------------------------------------------------

im = im2single(im) ;
if size(im,1) > 480, im = imresize(im, [480 NaN]) ; end
