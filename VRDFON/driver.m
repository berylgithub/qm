
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

x=[-1 -1]';            % starting point 
nfmax=1000;            % maximal number of function evaluations

% The following loop may be replaced by an arbitrarily complex 
% computing environment. 
for nf=1:nfmax,

  f=(x(1)-1)^2+100*(x(2)-x(1)^2)^2; % evaluate Rosenbrock function at x
  % in place of this comment one may wish to save the history, 
  % and/or check stopping tests 
  x=mintry(x,f);
  % report function value and ask for new trial point

end;
[xbest,fbest,info]=mintry('show');
     % This command may also be used inside the loop, 
     % together with an appropiate conditional statement 
     % implementing alternative stopping test. 
disp(xbest)
disp(fbest)