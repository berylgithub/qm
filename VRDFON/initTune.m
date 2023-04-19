function [ctune,itune] = initTune(ctune,itune,nc,ni)
% [ctune,itune,version] = initTune(ctune,itune,version,nc,ni)
% artificial lower and upper bounds to help prevent overflow
% initialize structure containing all tuning parameters 
%

 % version: 0 (defualt), 1 (no line search), 2 (no trust region)
 %          3 (no mixed-integer)
%  if isempty(version), version=0; end


if nc>0
    % ctune % structure containing all tuning parameters 
    %       % all parameters have a default that can be overwritten 
    %       % by specifying it as an input
    if ~exist('ctune'), ctune=[]; end
    if ~isfield(ctune,'clambda'), ctune.clambda =max(6,nc); end
    if ~isfield(ctune,'cmu'), ctune.cmu = 3+ceil(log(nc)); end
    if ~isfield(ctune,'sigmac'), ctune.sigmac = 1; end
    if ~isfield(ctune,'czeta'), ctune.czeta = 1e-20; end
    if ~isfield(ctune,'cnu'), ctune.cnu = 2; end
    if ~isfield(ctune,'ctheta'), ctune.ctheta = 2; end
    if ~isfield(ctune,'csigmamax'), ctune.csigmamax = 1e10;  end
    if ~isfield(ctune,'csigmamin'), ctune.csigmamin = 1e-10;  end

    % factor for adjusting Y
    if ~isfield(ctune,'gammaX'), ctune.gammaX = 1e3; end
    % factor for adjusting gradient
    if ~isfield(ctune,'gammav'), ctune.gammav = 1e2; end
    if ~isfield(ctune,'cDeltamin'), ctune.cDeltamin = 1e-3; end
end

if ni>0
    % itune % structure containing all tuning parameters 
    %       % all parameters have a default that can be overwritten 
    %       % by specifying it as an input

    if ~exist('itune'), itune=[]; end

    if ~isfield(itune,'ilambda'), itune.ilambda =max(6,ni); end
    if ~isfield(itune,'imu'), itune.imu = 3+ceil(log(ni)); end
    if ~isfield(itune,'sigmai')
        itune.sigmai = 1;
    end

    if ~isfield(itune,'izeta'), itune.izeta = 1e-20; end
    if ~isfield(itune,'inu'), itune.inu = 2; end
    if ~isfield(itune,'itheta'), itune.itheta = 2; end
    if ~isfield(itune,'iDeltainit'), itune.iDeltainit = 10;  end
    if ~isfield(itune,'iDeltabar'), itune.iDeltabar = 3;  end
    if ~isfield(itune,'isigmamax'), itune.isigmamax = 1e2;  end
    % factor for adjusting Y
    if ~isfield(itune,'gammaX'), itune.gammaX = 1e3; end
    % factor for adjusting gradient
    if ~isfield(itune,'gammav'), itune.gammav = 1e2; end
end

