%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% iusequence.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function D=iusequence(N,d);
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

function D=iusequence(MA,ilambda,d)

% for k=1 we pick the zero vector 
D=zeros(d,ilambda);             % storage for the sequence to be constructed
M=max(1000,ilambda);

z = randi([-MA.int.dir,MA.int.dir],d,M);
     
if isfield(MA.int,'pinit')
    z(:,1)=MA.int.pinit;
end       
zerM=zeros(1,M);     % for later vectorization        
for k=1:ilambda
  % find the reservoir vector with largest minimum distance
  % from the vectors already chosen
   % pick a vector from the reservoir
  if k==1
     j=1;
  else 
     % find the reservoir vector with largest minimum distance
     % from the vectors already chosen
     [~,j]=max(u); 
     j=j(1);             % break ties
  end
  D(:,k)=z(:,j);  
  
  ii=0;
  while ii<10
      ii=ii+1;
      zj= randi([-MA.int.dir,MA.int.dir],d,1);
      if norm(zj)~=0, break;end
  end

  z(:,j)=zj;  % update the reservoir
  % update minimum squared distance vector 
  onk = ones(1,k); u(j)=min(sum((zj(:,onk)-D(:,1:k)).^2,1));
  s=sum((z-D(:,k+zerM)).^2,1);
  if k==1, u=s; else, u=min(u,s); end
end

