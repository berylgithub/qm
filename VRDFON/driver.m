
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
init.n=2;              % problem dimension
                       % For tuning or to see intermediate results, 
                       % a nonstandard initialization may be used.
                       % For details see mintry.m

mintry(init);          % initialize mintry

%x=[-1 -1]';            % starting point 
nfmax=1000;            % maximal number of function evaluations

% The following loop may be replaced by an arbitrarily complex 
% computing environment. 
%for nf=1:nfmax,
% initial ops:
data = textread('../params.txt', "%s");
fdata = textread('../fun.txt', "%s");
x = [str2double(data{2,1}) str2double(data{3,1})]' %initialize x
f = str2double(fdata{2,1})
x=mintry(x,f) % the solver
%uid = datestr(now(), 'yyyymmddHHMMSS'); % not good, since this only unique for each second
uid = rand(1);
strout = num2str(uid);
for i=1:length(x)
  strout = strcat(strout,"\t",num2str(x(i)));
end
file_id = fopen('../params.txt', 'w');
fputs(file_id, strout);
fclose(file_id);
nf = 1;
% next ops:
unwind_protect
  while true
    newdata = textread('../fun.txt', "%s");
    if ~strcmp(newdata{1,1}, fdata{1,1}) % {1,1} is the uid
      fdata = newdata % fetch new function info
      f = str2double(fdata{2,1}); % get obj value
      feas=false
      while feas==false
        disp(feas)
        disp(x)
        disp(f)
        x=mintry(x,f) % the solver
        % round x by probablity:
        for i=1:length(x)
          p=rand(1);
          if p < .5; % 50% chance
            x(i) = floor(x(i));
          else
            x(i) = ceil(x(i));
          end
        end
        %feas=paramcheck(x) % check feasibility, repeat until feasible
        if x(1) < 0 || x(2) < 0
          feas=false;
        else
          feas=true;
        end
        if !feas
          f = 50.; % set 50kcal/mol MAE
        end
      end
      % write x to file:
      uid = rand(1);
      strout = num2str(uid);
      for i=1:length(x)
        strout = strcat(strout,"\t",num2str(x(i)));
      end
      file_id = fopen('../params.txt', 'w');
      fputs(file_id, strout);
      fclose(file_id);
    end
    if nf == nfmax
      break
    end
    nf += 1
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
  file_id = fopen('../best_fun_params.txt', 'w');
  fputs(file_id, strout);
  fclose(file_id);
end_unwind_protect