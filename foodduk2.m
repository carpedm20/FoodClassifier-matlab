conf.calDir = 'data/food100' ;
conf.dataDir = 'data/';

opts.dataset = 'food100' ;
opts.prefix = 'fv' ;
opts.encoderParams = {'type', 'fv'} ;
opts.seed = 1 ;
opts.lite = false ;
opts.C = 1 ;
opts.kernel = 'linear' ;
opts.dataDir = 'data';

for pass = 1:2
  opts.datasetDir = fullfile(conf.dataDir, opts.dataset) ;
  opts.resultDir = fullfile(conf.dataDir, opts.prefix) ;
  opts.imdbPath = fullfile(opts.resultDir, 'imdb.mat') ;
  opts.encoderPath = fullfile(opts.resultDir, 'encoder.mat') ;
  opts.modelPath = fullfile(opts.resultDir, 'model.mat') ;
  opts.diaryPath = fullfile(opts.resultDir, 'diary.txt') ;
  opts.cacheDir = fullfile(opts.resultDir, 'cache') ;
end

opts.numTrain = 20 ;
opts.numTest = 10 ;
opts.seed = 1 ;
opts.variant = 'food100' ;

numClasses = 100 ;

% Read classes
imdb = setupGeneric(conf.calDir, ...
  'numTrain', opts.numTrain, 'numVal', 0, 'numTest', opts.numTest,  ...
  'expectedNumClasses', numClasses, ...
  'seed', opts.seed, 'lite', opts.lite) ;

if exist(opts.encoderPath)
  encoder = load(opts.encoderPath) ;
else
  numTrain = 5000 ;
  if opts.lite, numTrain = 10 ; end
  train = vl_colsubset(find(imdb.images.set <= 2), numTrain, 'uniform') ;
  encoder = trainEncoder(fullfile(imdb.imageDir,imdb.images.name(train)), ...
                         opts.encoderParams{:}, ...
                         'lite', opts.lite) ;
  save(opts.encoderPath, '-struct', 'encoder') ;
  diary off ;
  diary on ;
end

descrs = encodeImage(encoder, fullfile(imdb.imageDir, imdb.images.name), ...
  'cacheDir', opts.cacheDir) ;
diary off ;
diary on ;


if isfield(imdb.images, 'class')
  classRange = unique(imdb.images.class) ;
else
  classRange = 1:numel(imdb.classes.imageIds) ;
end
numClasses = numel(classRange) ;

% apply kernel maps
switch opts.kernel
  case 'linear'
  case 'hell'
    descrs = sign(descrs) .* sqrt(abs(descrs)) ;
  case 'chi2'
    descrs = vl_homkermap(descrs,1,'kchi2') ;
  otherwise
    assert(false) ;
end
descrs = bsxfun(@times, descrs, 1./sqrt(sum(descrs.^2))) ;

% train and test
train = find(imdb.images.set <= 2) ;
test = find(imdb.images.set == 3) ;
lambda = 1 / (opts.C*numel(train)) ;
par = {'Solver', 'sdca', 'Verbose', ...
       'BiasMultiplier', 1, ...
       'Epsilon', 0.001, ...
       'MaxNumIterations', 100 * numel(train)} ;

scores = cell(1, numel(classRange)) ;
ap = zeros(1, numel(classRange)) ;
ap11 = zeros(1, numel(classRange)) ;
w = cell(1, numel(classRange)) ;
b = cell(1, numel(classRange)) ;
for c = 1:numel(classRange)
  if isfield(imdb.images, 'class')
    y = 2 * (imdb.images.class == classRange(c)) - 1 ;
  else
    y = - ones(1, numel(imdb.images.id)) ;
    [~,loc] = ismember(imdb.classes.imageIds{classRange(c)}, imdb.images.id) ;
    y(loc) = 1 - imdb.classes.difficult{classRange(c)} ;
  end
  if all(y <= 0), continue ; end

  [w{c},b{c}] = vl_svmtrain(descrs(:,train), y(train), lambda, par{:}) ;
  scores{c} = w{c}' * descrs + b{c} ;

  [~,~,info] = vl_pr(y(test), scores{c}(test)) ;
  ap(c) = info.ap ;
  ap11(c) = info.ap_interp_11 ;
  fprintf('class %s AP %.2f; AP 11 %.2f\n', imdb.meta.classes{classRange(c)}, ...
          ap(c) * 100, ap11(c)*100) ;
end
scores = cat(1,scores{:}) ;

diary off ;
diary on ;

% confusion matrix (can be computed only if each image has only one label)
if isfield(imdb.images, 'class')
  [~,preds] = max(scores, [], 1) ;
  confusion = zeros(numClasses) ;
  for c = 1:numClasses
    sel = find(imdb.images.class == classRange(c) & imdb.images.set == 3) ;
    tmp = accumarray(preds(sel)', 1, [numClasses 1]) ;
    tmp = tmp / max(sum(tmp),1e-10) ;
    confusion(c,:) = tmp(:)' ;
  end
else
  confusion = NaN ;
end

% save results
save(opts.modelPath, 'w', 'b') ;
save(fullfile(opts.resultDir,'result.mat'), ...
     'scores', 'ap', 'ap11', 'confusion', 'classRange', 'opts') ;

% figures
meanAccuracy = sprintf('mean accuracy: %f\n', mean(diag(confusion)));
mAP = sprintf('mAP: %.2f %%; mAP 11: %.2f', mean(ap) * 100, mean(ap11) * 100) ;

figure(1) ; clf ;
imagesc(confusion) ; axis square ;
title([opts.prefix ' - ' meanAccuracy]) ;
vl_printsize(1) ;
print('-dpdf', fullfile(opts.resultDir, 'result-confusion.pdf')) ;
print('-djpeg', fullfile(opts.resultDir, 'result-confusion.jpg')) ;

figure(2) ; clf ; bar(ap * 100) ;
title([opts.prefix ' - ' mAP]) ;
ylabel('AP %%') ; xlabel('class') ;
grid on ;
vl_printsize(1) ;
ylim([0 100]) ;
print('-dpdf', fullfile(opts.resultDir,'result-ap.pdf')) ;

disp(meanAccuracy) ;
disp(mAP) ;
diary off ;
