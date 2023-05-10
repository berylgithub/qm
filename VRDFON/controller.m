% controls the flow between simulators and mintry:
%   - periodically:
%       - load new x from mintry whenever it is updated
%       - checks for new signal from simulators:
%           - takes new f if the simulator places it, if the current iteration quota is fulfilled, return the best (rounded x, f) to mintry
%           - give rounded x to simulators that sends request
%       
%       

