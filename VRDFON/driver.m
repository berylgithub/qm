
% MinTry, reentrant uncoonstrained optimization
% driver.m      % driver for reentrant uncoonstrained optimization
% mintry.m      % reentrant uncoonstrained optimization algorithm
% VRDFOstep.m   % <solver>step.m must exist for each solver
% initTune.m    % must exist for each solver
% initMem.m     % must exist for each solver
% VRBBONrun.m   % temporary; will ultimately be VRBBON.m



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% driver.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mintry driver for reentrant unconstrained optimization
%
% replace in the driver the starting point and the Rosenbrock function 
% by your own problem data.
%

clear

% standard initialization


init.paths='minq8'; % mandatory absolute path to minq8
init.n=9;              % problem dimension
                       % For tuning or to see intermediate results, 
                       % a nonstandard initialization may be used.
                       % For details see mintry.m

mintry(init);          % initialize mintry

%x=[-1 -1]';            % starting point 
nfstuck = 50;            % max nf of getting stuck until reset

% The following loop may be replaced by an arbitrarily complex 
% computing environment. 
% initial ops, the params and fun path MUST be non empty:
path_param = '../data/hyperparamopt/params.txt';
path_fun = '../data/hyperparamopt/fun.txt';
path_fbest = '../data/hyperparamopt/best_fun_params.txt';
data = textread(path_param, "%s");
fdata = textread(path_fun, "%s");
x = zeros(length(data)-1, 1);
for i=2:length(data)
  x(i-1) = str2double(data{i,1});
end
f = str2double(fdata{2,1});
x = xgenerator(x, f) % contains main loop to generate x given (x,f) and projection to feasible sol
disp(x)
paramwriter(x, path_param) % write x to file
%nf = 1;
% next ops:
unwind_protect
  while true
    newdata = textread(path_fun, "%s");
    if (dir(path_fun).bytes > 0) && (~strcmp(newdata{1,1}, fdata{1,1})) % check if the file is not empty and the file is new; {1,1} is the uid
      fdata = newdata % fetch new function info
      f = str2double(fdata{2,1}); % get obj value
      x = xgenerator(x, f)
      paramwriter(x, path_param) % write x to file, here x is feasible
    end
    %nf += 1
    pause(0.3) % check new data for each second
  end
unwind_protect_cleanup
  [xbest,fbest,info]=mintry('show');
      % This command may also be used inside the loop, 
      % together with an appropiate conditional statement 
      % implementing alternative stopping test. 
  % write best to file:
  strout = num2str(fbest);
  for i=1:length(xbest)
    strout = strcat(strout,"\t",num2str(xbest(i)));
  end
  if exist(path_fbest, 'file') % if prev best exists, write only if current best is better
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
end_unwind_protect
