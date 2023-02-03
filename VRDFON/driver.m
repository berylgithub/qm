
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
uid = datestr(now(), 'yyyymmddHHMMSS');
strout = strcat(num2str(uid),"\t",num2str(x(1)),"\t",num2str(x(2)));
file_id = fopen('../params.txt', 'w');
fputs(file_id, strout);
fclose(file_id);
nf = 1;
% next ops:
while true
  newdata = textread('../fun.txt', "%s");
  if ~strcmp(newdata{1,1}, fdata{1,1}) % {1,1} is the uid
    fdata = newdata
    f = str2double(fdata{2,1});
    x=mintry(x,f) % the solver
    % write x to file:
    uid = datestr(now(), 'yyyymmddHHMMSS');
    strout = strcat(num2str(uid),"\t",num2str(x(1)),"\t",num2str(x(2)));
    file_id = fopen('../params.txt', 'w');
    fputs(file_id, strout);
    fclose(file_id);
  end
  %if nf == nfmax
  %  break
  %end
  nf += 1;
  pause(0.5) # check new data for each second
end
%end;
[xbest,fbest,info]=mintry('show');
     % This command may also be used inside the loop, 
     % together with an appropiate conditional statement 
     % implementing alternative stopping test. 
