

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% driver.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main driver to generate feasible x given (x,f) using xgenerator.m 
clear

% initial ops, the params and fun path MUST be non empty:
path_init_param = '../data/hyperparamopt/init_params.txt';
path_param = '../data/hyperparamopt/params.txt';
path_fun = '../data/hyperparamopt/fun.txt';
path_fbest = '../data/hyperparamopt/best_fun_params.txt';
path_trackx = '../data/hyperparamopt/xlist.txt'; 
path_trackxraw = '../data/hyperparamopt/xrawlist.txt';
path_trackf = '../data/hyperparamopt/flist.txt';
path_bounds = '../data/hyperparamopt/bounds.txt';

disp("init data...")
bounds = dlmread(path_bounds); % var bounds
data = textread(path_init_param, "%s"); % params
xin = zeros(length(data)-1, 1);
xinit = xin;
% init list of (x,f):
if ~exist(path_trackx) && ~exist(path_trackxraw) && ~exist(path_trackf)
  flist = []; xlist = []; xrawlist = [];
else
  flist = dlmread(path_trackf)'; 
  xlist = dlmread(path_trackx); 
  xrawlist = dlmread(path_trackxraw);
end
% accept encoded x:
for i=2:length(data)
  xin(i-1) = str2double(data{i,1});
end
xprev = [];

bm = extractbound(bounds); % compute boundary index matrix
[xout, fpen] = decode(xin, bounds, bm) % decode x
paramwriter(xout) % write decoded x
disp("x has been written to file..")

disp("init mintry ops...")
% mintry init:
init.paths='minq8';     % mandatory absolute path to minq8
init.n=length(xin);              % problem dimension
                       % For tuning or to see intermediate results, 
                       % a nonstandard initialization may be used.
                       % For details see mintry.m
mintry(init);          % initialize mintry

% init some vars:
nfstuck = 5; % max nf of getting stuck until reset
ct = 0; % stuck counter
cpen = 1.; % penalty factor
citer = 1; % iteration counter

% main loop:
unwind_protect
  while true
    newdata = textread(path_fun, "%s"); % get new f
    % check if the new file is not empty and the uid is new; {1,1} is the uid:
    if (dir(path_fun).bytes > 0) && (~strcmp(newdata{1,1}, fdata{1,1}))
      disp("new incoming data")
      fdata = newdata % fetch new function info
      f = str2double(fdata{2,1}); % get obj value
      if citer == 1 % set initial f for reset purpose
        finit = f
      end
      f += cpen*fpen; % add penalty term
      disp("[f, penalty] = ")
      disp([f, fpen])
      disp("mintry ops")
      % check reset counter:
      if ct >= 5
        disp("restart !!")
        ct = 0; % reset counter
        mintry(init); % restart mintry
        xin = xinit; f = finit; % restart (x,f)
        cpen /= 10. % reduce penalty factor
      end
      % compare if new x (components) is equal to prev x:
      if !isempty(xprev)
        if xout(3:9) == xprev(3:9)
          ct += 1
        else
          ct = 0
        end
      end
      % set new prev:
      xprev = xout; fprev = f;
      % append (x,f):
      xlist = [xlist; xout']; xrawlist = [xrawlist; xin']; flist = [flist f];
      % main loop and (x,f) trackers (get new x):
      [xout, xin, f, fpen, xlist, flist] = paramtracker(xin, f, xlist, flist, bounds, bm); 
      paramwriter(xout, path_param); % write x to file
      disp([f fpen])
      disp(xout)
      disp("x has been written to file")
      citer += 1; % increment iteration counter
    end
    % check new data for each second
    % to accomodate for memory > disk speed
    pause(0.5) 
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
  dlmwrite(path_trackf, flist, "\n");
end_unwind_protect
