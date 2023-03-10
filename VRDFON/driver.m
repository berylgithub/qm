

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% driver.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main driver to generate feasible x given (x,f) using xgenerator.m 
clear

% standard initialization:
init.paths='minq8';     % mandatory absolute path to minq8
init.n=9;              % problem dimension
                       % For tuning or to see intermediate results, 
                       % a nonstandard initialization may be used.
                       % For details see mintry.m
mintry(init);          % initialize mintry
%nfstuck = 50;            % max nf of getting stuck until reset

% The following loop may be replaced by an arbitrarily complex 
% computing environment. 
% initial ops, the params and fun path MUST be non empty:
path_param = '../data/hyperparamopt/params.txt';
path_fun = '../data/hyperparamopt/fun.txt';
path_fbest = '../data/hyperparamopt/best_fun_params.txt';
path_trackx = '../data/hyperparamopt/xlist.txt'; 
path_trackxraw = '../data/hyperparamopt/xrawlist.txt';
path_trackf = '../data/hyperparamopt/flist.txt';
path_bounds = '../data/hyperparamopt/bounds.txt';

disp("init data...")
bounds = dlmread(path_bounds); % var bounds
data = textread(path_param, "%s");
fdata = textread(path_fun, "%s");
x = zeros(length(data)-1, 1);
% init list of (x,f):
if ~exist(path_trackx) && ~exist(path_trackxraw) && ~exist(path_trackf)
  flist = []; xlist = []; xrawlist = [];
else
  flist = dlmread(path_trackf); 
  xlist = dlmread(path_trackx); 
  xrawlist = dlmread(path_trackxraw);
end
for i=2:length(data)
  x(i-1) = str2double(data{i,1});
end
f = str2double(fdata{2,1}); % here penalty term f'=sum(abs(x-xraw)) = 0
disp("init mintry ops...")
% main loop and (x,f) trackers:
[x, xraw, f, xlist, flist] = paramtracker(x, f, xlist, flist, bounds); 
paramwriter(x, path_param); % write x to file
% normalize the difference (since each entry has different range):
xdiff = abs(x-xraw); 
for i=1:length(xdiff)
  xdiff(i) = xdiff(i)/bounds(2,i);
end
disp("[x, xraw]= ")
disp([x xraw]) % feasible x
disp(sum(xdiff))
disp("x has been written to file..")
% next ops:
unwind_protect
  while true
    newdata = textread(path_fun, "%s");
    % check if the new file is not empty and the uid is new; {1,1} is the uid:
    if (dir(path_fun).bytes > 0) && (~strcmp(newdata{1,1}, fdata{1,1}))
      disp("new incoming data")
      fdata = newdata % fetch new function info
      f = str2double(fdata{2,1}); % get obj value
      % normalize x (since each entry has different range):
      xdiff = abs(x-xraw); 
      for i=1:length(xdiff)
        xdiff(i) = xdiff(i)/bounds(2,i);
      end
      f += sum(xdiff); % add penalty term
      disp("[f, penalty] = ")
      disp([f, sum(xdiff)])
      % append lists:
      xlist = [xlist; x']; xrawlist = [xrawlist; xraw']; flist = [flist f];
      disp("mintry ops")
      % main loop and (x,f) trackers:
      [x, xraw, f, xlist, flist] = paramtracker(x, f, xlist, flist, bounds);
      paramwriter(x, path_param); % write x to file
      disp("[x, xraw]= ")
      disp([x xraw]) % feasible x
      disp("x has been written to file")
    end
    % check new data for each second
    % to accomodate for memory > disk speed
    pause(0.3) 
  end
unwind_protect_cleanup
  disp("running finished")
  [xbest,fbest,info]=mintry('show');
      % This command may also be used inside the loop, 
      % together with an appropiate conditional statement 
      % implementing alternative stopping test. 
  % write best to file:
  strout = num2str(fbest);
  for i=1:length(xbest)
    strout = strcat(strout,"\t",num2str(xbest(i)));
  end
  % if prev best exists, write only if current best is better:
  if exist(path_fbest, 'file')
    prevbest = textread(path_fbest, "%s");
    fprev = str2double(prevbest{1,1});
    if (fprev > fbest) || (isnan(fprev))
      file_id = fopen(path_fbest, 'w');
      fputs(file_id, strout);
      fclose(file_id);
    end
  else % if prev best doesnt exist yet, just write the current best
      file_id = fopen(path_fbest, 'w');
      fputs(file_id, strout);
      fclose(file_id);
  end
  % write list (xlist, flist) to file:
  disp(xlist)
  disp(flist)
  dlmwrite(path_trackx, xlist, "\t");
  dlmwrite(path_trackxraw, xrawlist, "\t"); 
  dlmwrite(path_trackf, flist, "\t");
end_unwind_protect
