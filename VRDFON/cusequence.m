

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% cusequence.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function D=cusequence(N,d);
% generates a sequence of N vectors in the d-dimensional unit cube 
% such that for each leading subsequence, arbitrary vectos are 
% close to one of the vectors in the subsequence.
% 2*D-1 can be used as an efficient replacement for Halton sequences
% or other low discrepancy sequences.  
%
% N      % number of sample points wanted 
% d      % dimesnion of sample points wanted
%
% D(:,k) % k-th point of the sequence
% 
function D=cusequence(MA,N,d)

% for k=1 we pick the zero vector 
D=zeros(d,N);             % storage for the sequence to be constructed
% for k=2 we pick the all-one vector

M=max(1000,N);

z=MA.cont.dir*(2*rand(d,M)-1);        % random reservoir of M vectors
if isfield(MA.cont,'pinit')
   z(:,1)=MA.cont.pinit;
end    
zerM=zeros(1,M);    % for later vectorization        

for k=1:N
   if k==1
     j=1;
  else 
     % find the reservoir vector with largest minimum distance
     % from the vectors already chosen
     [~,j]=max(u); 
     j=j(1);             % break ties
   end
   D(:,k)=z(:,j); 
   zj=MA.cont.dir*(2*rand(d,1)-1); 
   z(:,j)=zj;      % update the reservoir
   % update minimum squared distance vector 
   % update minimum squared distance vector 
   onk = ones(1,k);
   u(j)=min(sum((zj(:,onk)-D(:,1:k)).^2,1));
   s=sum((z-D(:,k+zerM)).^2,1);
   if k==1, u=s; else, u=min(u,s); end
end




