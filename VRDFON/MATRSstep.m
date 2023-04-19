%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% MATRSstep.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% performs one step of MATRS
% 
%
% function MATRSstep(itune,ctune,fun,prt,ni,nc); % initialize solver
%
% itune      % structure contains those tuning integer parameters that  
%            % are explicitly specified; the others take default values
%            %   (for choices and defaults see initTune.m)

% ctune      % structure contains those tuning continuous parameters that  
%            % are explicitly specified; the others take default values
%            %   (for choices and defaults see initTune.m)

% fun          % function handle (empty if used in reentrant mode)
% prt          % print level
% ni           % problem dimension in the subspace of integer vaiables
% nc           % problem dimension in the subspace of continous vaiables

%
% function x=MATRSstep(x,f,prt); % suggest new point
%
% x            % point evaluated (input) or to be evaluated (output)
% f            % function value at input x
% prt          % change print level (optional)
%
% function [xbest,fbest,info]=MATRSstep();  % read results
%
% xbest        % current best point
% fbest        % function value found at xbest
% info         % performance information for MATRS
%  .finit      %   initial function value
%  .ftarget    %   target function value (to compute qf)
%  .qf         %   (ftarget-f)/(finit-f)
%  .initTime   %   inital cputime
%  .done       %   done with the search?
% 
function [x,f,info1] = MATRSstep(x,f,p,indInt,low,upp,budget)

persistent MA Vstate prt itune ctune info ni nc dim

persistent xbest fbest  xfinal ffinal ordering nfmax


if nargin==0
  %%%%%%%%%%%%%%%%%%%%%
  %%%% return info %%%%
  %%%%%%%%%%%%%%%%%%%%%
 
  if ffinal<fbest
     f=ffinal; x=xfinal;
  else
     f=fbest; x=xbest;
  end
  
  info1.nf=info.nf;
%   info1.number_success_intTR=info.niTR;
%   info1.number_success_contTR=info.ncTR;
%   info1.number_success_contLS=info.ncLS;
%   info1.number_success_intLS=info.niLS;
%   info1.number_success_mixedintLS=info.nmiLS;

  if prt>=0
      disp(' ')
      disp(' ') 
      disp('end of main loop of MATRS:')
      disp(' ')
      disp(' ')
  end
  return;
end;

if nargout==0
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%% initialize solver environment %%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  info = [];
  dim = length(indInt); % problem dimension
  info.cI = find(indInt==0); % subspace dimension of continuous variables
  info.iI = setdiff(1:dim,info.cI);% subspace dimension of integer variables
  nc = length(info.cI);
  ni = length(info.iI); 
  if nc>0 % lower and upper bound on continuous variables
    info.clow = low(info.cI); info.cupp = upp(info.cI);
  end
  if ni>0 % lower and upper bound on continuous variables
     info.ilow = low(info.iI); info.iupp = upp(info.iI);
  end
  xc = x.cont; yc = x.int;
  [ctune,itune] = initTune(xc,yc,nc,ni);
%    xc = x.cont; yc = x.int; zc = x.version;
%   [ctune,itune,version] = initTune(xc,yc,zc,nc,ni);
  
  % tuning parameters
  nfmax = budget;
  
  prt = p; info1=info;    Vstate=1; 
  if prt>=0
      disp(' ')
      disp(' ') 
      disp('start of main loop of MATRS:')
      disp(' ')
      disp(' ')
  end
  return;
end

if Vstate==1
    

xbest = x; fbest=f; info.nf=0; 
info.niTR =0; info.ncTR =0; info.niLS =0; info.ncLS =0; 
info.nmiLS =0;

ordering = 'ascend';

if ni>0 
    MA.int=[];
    MA.int.sigma=itune.sigmai;
    MA.int.Parent.y = x;
    MA.int.s = zeros(ni, 1);
    MA.int.M = eye(ni);
    MA.int.echi = sqrt(ni)*(1 - 1/ni/4 - 1/ni/ni/21);
    MA.int.a = ones(itune.ilambda,1); 
    MA.int.pinit = ones(ni,1);
    MA.int.dir=5;
    MA.int.D   = iusequence(MA,itune.ilambda,ni);    
    MA.int.good=1;
end

if nc>0 
    MA.cont=[];
    MA.cont.sigma=ctune.sigmac;
    MA.cont.echi = sqrt(nc)*(1 - 1/nc/4 - 1/nc/nc/21);
    MA.cont.Parent.y = x;
    MA.cont.s = zeros(nc, 1);
    MA.cont.M = eye(nc);
    MA.cont.pinit = ones(nc,1);
    MA.cont.dir=1;
    MA.cont.D   = cusequence(MA,ctune.clambda,nc);  
    MA.cont.a = ones(ctune.clambda,1); 
    MA.cont.good=1;
end
info.xf=Inf*ones(nfmax,dim+1);
info.xf(1,1:dim)=x';
info.xf(1,dim+1)=fbest;
%if nc>0 && ni>0
  info.X = xbest; info.F = fbest;
%end
MA.int.iit=0; MA.cont.cit=0; 
if (nc>0 && ni==0) || (nc>0 && ni>0)
   Vstate=2;
elseif nc==0 && ni>0
    Vstate=30;
end

end
% save the evaluated point and its function value
xfinal = x; ffinal = f;  info.nf=info.nf+1;

while 1
    if nc>0  
       if Vstate==2 
            MA.cont.cit = MA.cont.cit+1;  
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           % phase I: Perform integer mutation
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           if prt>=2 
               fprintf(['start of ',num2str(MA.cont.cit),'th cMutation\n'])
           end
           if ~MA.cont.good, MA.cont.OffspringPop0=MA.cont.OffspringPop;
           else, MA.cont.OffspringPop0 =[];
           end
           info.xbest0  = xbest; 
           MA.cont.goodm  = 0; MA.cont.OffspringPop = []; MA.cont.cm = 1;
           Vstate  = 3;
       end
       while Vstate<14
            if Vstate==3
                MA.cont.Offspring.sd = MA.cont.D(:,MA.cont.cm); 
                MA.cont.Offspring.p  = MA.cont.M*MA.cont.Offspring.sd;
                MA.cont.cxbest = xbest(info.cI);       
                [MA,info]=reqcontLSS(MA,info);
                if ~MA.cont.feasible
                     MA.cont.cm=MA.cont.cm+1;
                      if MA.cont.cm>size(MA.cont.D,2)
                       Vstate=14; 
                         break;% end of the mutation phase
                      else
                         Vstate=3; % go back to continue the mutation phase
                         continue
                      end
                else
                    % calculate dimension of the problem
                    % Build first point for starting linesearch
                    info.yworst     = xbest;
                    info.yworst(info.cI) = ...
                       max(info.clow,min(info.cupp,...
                       xbest(info.cI)+MA.cont.alp0*MA.cont.Offspring.p));
                    Vstate=4;
                    x=info.yworst;
                    return;
                end
            end
            if Vstate==4
                info.fworst   = f; 
                info.xf(info.nf,1:dim) = info.yworst'; 
                info.xf(info.nf,dim+1) = info.fworst;
                info=UpdatePoints(info);
                % cicle for updating alpha
                if info.fworst<fbest
                    % initialize alpha and best point
                    MA.cont.alpha = MA.cont.alp0; 
                    info.ytrial=info.yworst; 
                    info.ftrial=info.fworst;
                    % calculate trial point
                    info.yworst  = xbest;
                    beta  = min(MA.cont.alpmax,ctune.cnu*MA.cont.alpha);
                    info.yworst(info.cI) = ...
                          max(info.clow,min(info.cupp,...
                         xbest(info.cI)+ beta * MA.cont.Offspring.p));
                    Vstate=5;
                    x=info.yworst;
                    return;
                else
                    MA.cont.alpha = 0; info.ytrial = xbest;
                    info.ftrial = Inf;
                    Vstate=8;
                end
            end
            if Vstate==5
                info.fworst  = f; 
                info.xf(info.nf,1:dim) = info.yworst'; 
                info.xf(info.nf,dim+1) = info.fworst;
                info=UpdatePoints(info);
                
                Vstate=6;
            end
            while Vstate==6||Vstate==7
                if Vstate==6
                    % expansion step (increase stepsize)
                    if (MA.cont.alpha<MA.cont.alpmax) && (info.fworst < fbest)
                        % MA.cont.alpha calulation and best point updating
                        MA.cont.alpha = ... 
                          min(MA.cont.alpmax,ctune.cnu*MA.cont.alpha);
                        % best point updating
                        info.ytrial=info.yworst; info.ftrial=info.fworst;
                        % next point to be tested
                        if(MA.cont.alpha < MA.cont.alpmax)
                           info.yworst = xbest; 
                           beta   = ...
                           min(MA.cont.alpmax,ctune.cnu*MA.cont.alpha);
                           info.yworst(info.cI) = max(info.clow,min(...
                                        info.cupp,xbest(info.cI)+...
                                        beta*MA.cont.Offspring.p));
                           Vstate=7;
                           x=info.yworst;
                           return;
                        end
                    else
                        Vstate=13; break;
                    end
                end
               if Vstate==7
                   info.fworst = f; 
                   info.xf(info.nf,1:dim)=info.yworst'; 
                   info.xf(info.nf,dim+1)=info.fworst;
                   info=UpdatePoints(info);
                   
                   Vstate=6;
               end
            end
            if Vstate==8
                MA.cont.Offspring.p = -MA.cont.Offspring.p; 
                MA.cont.cxbest      = xbest(info.cI);       
                [MA,info]          = reqcontLSS(MA,info);
                if ~MA.cont.feasible
                     MA.cont.cm = MA.cont.cm+1;
                      if MA.cont.cm>size(MA.cont.D,2)
                         Vstate=14; 
                         break;% end of the mutation phase
                      else % go back to continue the mutation phase
                         Vstate=3; 
                         continue
                      end
                else
                    info.fworst0=info.fworst;
                    % calculate dimension of the problem
                    % Build first point for starting linesearch
                    info.yworst = xbest;
                    info.yworst(info.cI) = max(info.clow,min(info.cupp,...
                                    xbest(info.cI) + ...
                                    MA.cont.alp0 * MA.cont.Offspring.p));
                    Vstate=9;
                    x=info.yworst;
                    return;
                end
            end
             if Vstate==9
                  info.fworst  = f; 
                  info.xf(info.nf,1:dim) = info.yworst'; 
                  info.xf(info.nf,dim+1) = info.fworst;
                  info=UpdatePoints(info);
                   
                  % cicle for updating MA.cont.alpha
                 if info.fworst<fbest
                    % initialize MA.cont.alpha and best point
                    MA.cont.alpha = MA.cont.alp0; info.ytrial=info.yworst;
                    info.ftrial=info.fworst;
                    % calculate trial point
                    info.yworst     = xbest;
                    beta    = ... 
                            min(MA.cont.alpmax,ctune.cnu*MA.cont.alpha);
                    info.yworst(info.cI) = ...
                        max(info.clow,min(info.cupp,...
                        xbest(info.cI)+beta*MA.cont.Offspring.p));
                    Vstate=10;
                    x=info.yworst;
                    return;
                else
                     MA.cont.Offspring.sd = -MA.cont.Offspring.sd;
                     MA.cont.alpha = 0; info.ytrial = xbest;
                     info.ftrial = Inf;
                     Vstate=13;
                     if info.fworst0<info.fworst
                         info.fworst=info.fworst0; 
                     end
                 end
            end
            if Vstate==10
                info.fworst  = f; 
                info.xf(info.nf,1:dim) = info.yworst'; 
                info.xf(info.nf,dim+1) = info.fworst;
                info=UpdatePoints(info);
                
                Vstate=11;
            end
            while Vstate==11||Vstate==12
                if Vstate==11
                    % expansion step (increase stepsize)
                    if (MA.cont.alpha<MA.cont.alpmax) && (info.fworst < fbest)
                        % MA.cont.alpha calulation and best point updating
                        MA.cont.alpha = ...
                          min(MA.cont.alpmax, ctune.cnu*MA.cont.alpha);
                        % best point updating
                        info.ytrial=info.yworst; info.ftrial=info.fworst;
                        % next point to be tested
                        if(MA.cont.alpha < MA.cont.alpmax)
                           info.yworst = xbest;  
                           beta   = min(MA.cont.alpmax,...
                                         ctune.cnu*MA.cont.alpha);
                           info.yworst(info.cI) = max(info.clow,...
                                    min(info.cupp,xbest(info.cI) + ...
                                    beta * MA.cont.Offspring.p));
                           Vstate=12;
                           x=info.yworst;
                           return;
                        end
                    else
                        Vstate=13; break;
                    end
                end
               if Vstate==12
                   info.fworst  = f; 
                   info.xf(info.nf,1:dim) = info.yworst'; 
                   info.xf(info.nf,dim+1) = info.fworst;
                   info=UpdatePoints(info);
                   
                   Vstate=11;
               end
            end
            if Vstate==13 
                 if MA.cont.alpha>0
                      info.ncLS =info.ncLS+1;
                      MA.cont.Offspring.f = info.ftrial; 
                      MA.cont.OffspringPop = ...
                            [MA.cont.OffspringPop MA.cont.Offspring];
                      if(info.ftrial < fbest)
                         MA.cont.a(MA.cont.cm) = MA.cont.alpha;
                         fbest = info.ftrial; xbest = info.ytrial; 
                         MA.cont.goodm=1;
                         if prt>=3
                          fprintf('continuous line search was successful\n')
                         end
                      else
                          MA.cont.a(MA.cont.cm) = MA.cont.sigma;
                      end
                 else
                      MA.cont.a(MA.cont.cm)= MA.cont.sigma;
                      MA.cont.Offspring.f  = info.fworst;
                      MA.cont.OffspringPop = [MA.cont.OffspringPop ...
                                   MA.cont.Offspring];
                 end
                 if MA.cont.feasible
                      MA.cont.cm=MA.cont.cm+1;
                      if MA.cont.cm>size(MA.cont.D,2)
                          Vstate=14; 
                          break;% end of the mutation phase
                      else % go back to continue the mutation phase
                          Vstate=3; 
                      end
                  end
            end
       end
       if Vstate==14 
          changDc = (norm(xbest(info.cI)-info.xbest0(info.cI)) == 0);
          if ~changDc
               MA.cont.pinit = info.xbest0(info.cI)-xbest(info.cI);
               MA.cont.pinit = 10*MA.cont.pinit/norm(MA.cont.pinit,inf);
                MA.cont.D  = cusequence(MA,ctune.clambda,nc);
               if MA.cont.dir<20 % 20,
                   MA.cont.dir=MA.cont.dir+1;
               else, MA.cont.dir=1;
               end
          else
              MA.cont.D  = randn(nc,ctune.clambda);
              MA.cont.dir    = 1;    
          end
          MA.cont.lambda = length(MA.cont.OffspringPop);
          if ~MA.cont.good && MA.cont.lambda<6
             MA.cont.OffspringPop = ...
                     [MA.cont.OffspringPop0,MA.cont.OffspringPop];
          end
          clambda = length(MA.cont.OffspringPop);
          MA.cont.good = MA.cont.goodm;
       
          if prt>=2
              fprintf(['end of ',num2str(MA.cont.cit),'th cMutation\n'])
          end
          MA.cont.goodr=0; MA.cont.okTR=~MA.cont.goodm;
          if clambda>=1
             cmu       = ceil(clambda/2);
             cwi_raw   = log(clambda/2 + 0.5) - log((1:cmu));
             cwi       = cwi_raw/sum(cwi_raw);
             cmu_eff   = max(1+eps,1/sum(cwi .^2));
             cc_s      = min(1.999,(cmu_eff + 2)/(nc + cmu_eff + 5));
             cc_1      = 2/((nc+1.3)^2 + cmu_eff);
             c_mu     = .... 
                min( 1-cc_1, 2*(cmu_eff - 2 + 1/cmu_eff)/...
                ((nc + 2)^2 + cmu_eff));
             cd_s     = 1 + cc_s + 2*max(0,sqrt((cmu_eff-1)/(nc+1))-1);
             sqrt_s   = sqrt(cc_s*(2-cc_s)*cmu_eff);
             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
             % phase II: perform selection
             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
             if prt>=2
                 fprintf(['start of ',num2str(MA.cont.cit),'th selection\n'])
             end
             cranks = RankPop(MA.cont.OffspringPop, ordering);
             sum_z = zeros(nc, 1);
             sum_d = zeros(nc, 1);
             for m = 1:cmu 
                 sum_z = sum_z + cwi(m)*MA.cont.OffspringPop(cranks(m)).sd;
                 sum_d = sum_d + cwi(m)*MA.cont.OffspringPop(cranks(m)).p;
              end
              if prt>=2, fprintf(['end of ',num2str(MA.cont.cit),'th selection\n'])
              end
              % update MA.cont.M
              MA.cont.s = (1-cc_s)*MA.cont.s + sqrt_s*sum_z;
              MA.cont.M = (1 - 0.5*cc_1 - 0.5*c_mu) * MA.cont.M + ...
                          (0.5*cc_1)*(MA.cont.M*MA.cont.s)*MA.cont.s';
              for m = 1:cmu
                 u1 = MA.cont.OffspringPop(cranks(m)).p; 
                 u2 = MA.cont.OffspringPop(cranks(m)).sd; 
                 MA.cont.M = MA.cont.M + ((0.5*c_mu*cwi(m))*u1)*u2';
              end  
              MA.cont.M=adjustY(MA.cont.M,ctune);
             % update MA.cont.sigma
             pow = norm(MA.cont.s)/MA.cont.echi - 1; 
             MA.cont.sigma = MA.cont.sigma*exp((cc_s/cd_s)*pow);
             MA.cont.sigma = max(ctune.csigmamin,...
                             min(ctune.csigmamax,MA.cont.sigma));
             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
             % perform integer recombination
             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
             MA.cont.okRec = norm(sum_d)~=0 && cmu>=3 && ~MA.cont.goodm;
             Vstate=15;
           else
              Vstate=26;
          end
       end
       if Vstate==15
         if MA.cont.okRec
              if prt>=2
                 fprintf(['start of ',num2str(MA.cont.cit),...
                          'th contRecombination\n'])
              end
             MA.cont.cxbest = xbest(info.cI);  nc = length(info.cI);
            okRecombonation=1;  MA.cont.goodr=0; cr=0; beta=1;
            while okRecombonation
               cr=cr+1; 
               if cr>1, beta = 2*rand(nc,1)-1; beta=beta/norm(beta); end
               MA.cont.drec = beta.*sum_d;
               okRecombonation = (norm(MA.cont.drec)~=0);
               if okRecombonation
                   % initialize vector alpha_max
                   ind = (MA.cont.drec > 0 & MA.cont.cxbest<info.cupp);
                   aupp = min((info.cupp(ind)-MA.cont.cxbest(ind))...
                          ./MA.cont.drec(ind));
                   if isempty(aupp), aupp=inf; end
                   ind = ( MA.cont.drec < 0 & MA.cont.cxbest<info.clow);
                   alow = min((info.clow(ind)-MA.cont.cxbest(ind))...
                          ./MA.cont.drec(ind));
                   if isempty(alow), alow=inf; end
                   MA.cont.alpmax=min(alow,aupp);
                   if isfinite(MA.cont.alpmax)&& MA.cont.alpmax>0
                       Vstate=16;
                       break;
                   end
               end
               if cr>=100, Vstate=26; break; end
            end
         else
             Vstate=26;
         end
      end
      if Vstate==16
        MA.cont.alp0 = sqrt(MA.cont.alpmax*MA.cont.sigma);
        % calculate dimension of the problem
        % Build first point for starting linesearch
        info.yworst  = xbest;
        info.yworst(info.cI) = max(info.clow,min(info.cupp,...
                         xbest(info.cI) + MA.cont.alp0 * MA.cont.drec));
        Vstate=17;
        x=info.yworst;
        return;
      end
      if Vstate==17
        info.fworst    = f; 
        info.xf(info.nf,1:dim)=info.yworst'; 
        info.xf(info.nf,dim+1)=info.fworst;
        info=UpdatePoints(info);
         
        Vstate=18;
      end
      if Vstate==18
            % cicle for updating MA.cont.alpha
            if info.fworst<fbest
                % initialize MA.cont.alpha and best point
                MA.cont.alpha = MA.cont.alp0; info.ytrial=info.yworst;
                info.ftrial=info.fworst;
                % calculate trial point
                info.yworst = xbest;
                beta = min(MA.cont.alpmax,ctune.cnu*MA.cont.alpha);
                info.yworst(info.cI) = max(info.clow,min(info.cupp,...
                                  xbest(info.cI) + beta * MA.cont.drec));
                Vstate=19;
                x=info.yworst;
                return;
            else
                Vstate=199;
            end
      end
      if Vstate==19
        info.fworst = f; 
        info.xf(info.nf,1:dim)=info.yworst'; 
        info.xf(info.nf,dim+1)=info.fworst;
        info=UpdatePoints(info);
        
        Vstate=20;
      end
       while Vstate==20||Vstate==21
            if Vstate==20
                % expansion step (increase stepsize)
                if (MA.cont.alpha<MA.cont.alpmax) && (info.fworst < fbest)
                    % MA.cont.alpha calulation and best point updating
                    MA.cont.alpha = ...
                        min(MA.cont.alpmax,ctune.cnu*MA.cont.alpha);
                    % best point updating
                    info.ytrial=info.yworst; info.ftrial=info.fworst;
                    % next point to be tested
                    if(MA.cont.alpha < MA.cont.alpmax)
                       info.yworst     = xbest;  
                       beta = min(MA.cont.alpmax,ctune.cnu*MA.cont.alpha);
                       info.yworst(info.cI) = max(info.clow,...
                                 min(info.cupp,xbest(info.cI) + ...
                                 beta * MA.cont.drec));
                       Vstate=21;
                       x=info.yworst;
                       return;
                    end
                else
                    Vstate=25; break;
                end
            end
           if Vstate==21
               info.fworst = f; 
               info.xf(info.nf,1:dim)=info.yworst'; 
               info.xf(info.nf,dim+1)=info.fworst;
               info=UpdatePoints(info);
               
               Vstate=20;
           end
       end  
       if Vstate==199
            MA.cont.drec=-MA.cont.drec;
           ind = (MA.cont.drec > 0 & MA.cont.cxbest<info.cupp);
           aupp = min((info.cupp(ind)-MA.cont.cxbest(ind))...
                  ./ MA.cont.drec(ind));
           if isempty(aupp), aupp=inf; end
           ind = ( MA.cont.drec < 0 & MA.cont.cxbest>info.clow);
           alow = min((info.clow(ind)-MA.cont.cxbest(ind))...
                  ./ MA.cont.drec(ind));
           if isempty(alow), alow=inf; end
           MA.cont.alpmax=min(alow,aupp);
           if MA.cont.alpmax~=0 && ~isinf(MA.cont.alpmax)
               MA.cont.alp0 = min(MA.cont.alpmax,MA.cont.sigma);
               info.yworst = xbest;
               info.yworst(info.cI)  = max(info.clow,min(info.cupp,...
                                   xbest(info.cI) + ...
                                   MA.cont.alp0*MA.cont.drec));
               Vstate=21;
               x=info.yworst;
               return;
           else
               Vstate=25; 
           end
           
       end
       if Vstate==21
            info.fworst    = f; 
            info.xf(info.nf,1:dim)=info.yworst'; 
            info.xf(info.nf,dim+1)=info.fworst;
            info=UpdatePoints(info);
             
            if info.fworst<fbest
                % initialize MA.cont.alpha and best point
                MA.cont.alpha = MA.cont.alp0; info.ytrial=info.yworst; 
                info.ftrial=info.fworst;
                % calculate trial point
                info.yworst     = xbest;
                beta = min(MA.cont.alpmax,ctune.cnu*MA.cont.alpha);
                info.yworst(info.cI) = max(info.clow,min( info.cupp,...
                              xbest(info.cI) + beta * MA.cont.drec));
                Vstate=22;
                x=info.yworst;
                return;
            else
                Vstate=25;
            end
       end
       if Vstate==22
            info.fworst   = f; 
            info.xf(info.nf,1:dim) = info.yworst'; 
            info.xf(info.nf,dim+1) = info.fworst;
            info=UpdatePoints(info);
            
            Vstate=23;
       end
       while Vstate==23||Vstate==24
            if Vstate==23
                % expansion step (increase stepsize)
                if (MA.cont.alpha<MA.cont.alpmax) && (info.fworst < fbest)
                    % MA.cont.alpha calulation and best point updating
                    MA.cont.alpha = ...
                         min(MA.cont.alpmax, ctune.cnu*MA.cont.alpha);
                    % best point updating
                    info.ytrial=info.yworst; info.ftrial=info.fworst;
                    % next point to be tested
                    if(MA.cont.alpha < MA.cont.alpmax)
                       info.yworst     = xbest; 
                       beta = min(MA.cont.alpmax,ctune.cnu*MA.cont.alpha);
                       info.yworst(info.cI) = max(info.clow,min(...
                                  info.cupp,xbest(info.cI) + ...
                                  beta * MA.cont.drec));
                       Vstate=24;
                       x=info.yworst;
                       return;
                    end
                else
                    Vstate=25; break;
                end
            end
           if Vstate==24
               info.fworst = f; 
               info.xf(info.nf,1:dim)=info.yworst'; 
               info.xf(info.nf,dim+1)=info.fworst;
               info=UpdatePoints(info);
               
               Vstate=23;
           end
       end  
      if Vstate==25                           
          if MA.cont.alpha>0
              if(info.ftrial < fbest)
                   info.ncLS =info.ncLS+1;
                   fbest = info.ftrial; xbest = info.ytrial; MA.cont.goodr=1;
                   if prt>=3
                       fprintf('continuous line search was successful\n')
                   end
              end
         else
              if(info.ftrial < fbest)
                  info.ncLS =info.ncLS+1;
                if prt>=3
                   fprintf('continuous line search was successful\n')
                end 
                fbest  = info.ftrial; xbest = info.ytrial; MA.cont.goodr=1;
              end
           end
           if prt>=2
              fprintf(['end of ',num2str(MA.cont.cit),'th cRecombination\n'])
           end
           MA.cont.good = MA.cont.good ||MA.cont.goodr;
           MA.cont.okTR = ~MA.cont.okRec||~MA.cont.goodr;
           Vstate=26;
       end
       if Vstate==26
           MA.cont.Del  = min(1,MA.cont.sigma);
           MA.cont.okTR = MA.cont.okTR && info.nf>=dim+1;
           if MA.cont.okTR % perform integer trust region algorithm
               if prt>=2
                 fprintf(['start of ',num2str(MA.cont.cit),'th cTRS\n'])
               end
               MA.cont.goodt=0; MA.cont.succTR=0; 
               MA.cont.unsuccTR=0;
               MA.cont.M = adjustY(MA.cont.M,ctune);
               MA.cont.G = MA.cont.M'*MA.cont.M; 
               Vstate = 27;
           else
               Vstate=29;
           end
       end
       while Vstate==27||Vstate==28
            if Vstate==27
                XX = info.xf(info.nf-dim:info.nf,1:dim)';
                FF = info.xf(info.nf-dim:info.nf,dim+1)';
                MA.cont.cxbest = xbest(info.cI);
                MA.cont.g   = fitGrad(nc,XX(info.cI,:),FF,MA.cont.cxbest,...
                         fbest,nc+1,ctune);
                warning off
                sn = -MA.cont.G\MA.cont.g;
                warning on
                MA.cont.sc=Dogleg(MA.cont.Del,sn,MA.cont.G,MA.cont.g);
                info.yworst     = xbest;
                info.yworst(info.cI) = ...
                         min(info.cupp,max(info.clow,...
                         MA.cont.cxbest+MA.cont.sc));
                Vstate=28;
                x=info.yworst;
                return;
            end
            if Vstate==28
                [info.fworst] = f; 
                info.xf(info.nf,1:dim)=info.yworst'; 
                info.xf(info.nf,dim+1)=info.fworst;
                info=UpdatePoints(info);
                
                gcsc = max(-MA.cont.g'*MA.cont.sc,...
                       eps*abs(MA.cont.g)'*abs(MA.cont.sc));
                mueff = (fbest-info.fworst)/gcsc;
                succcTR  = (mueff*abs(mueff-1) > ctune.czeta);
                % Updating iterate and trust-region radius.
                if succcTR 
                   if prt>=3
                      fprintf('continuous trust region was successful\n')
                   end 
                   fbest = info.fworst; xbest = info.yworst; 
                   MA.cont.goodt=1;
                   MA.cont.Del = max(norm(MA.cont.sc,inf),...
                                 ctune.ctheta)*MA.cont.Del;
                  MA.cont.succTR = MA.cont.succTR+1;    
                  info.ncTR =info.ncTR+1;
                else
                    MA.cont.unsuccTR = MA.cont.unsuccTR+1;  
                    MA.cont.Del = min(norm(MA.cont.sc,inf),...
                                  MA.cont.Del)/ctune.ctheta;
                    if MA.cont.Del<=ctune.cDeltamin
                        if prt>=2
                          fprintf(['end of ',num2str(MA.cont.cit),...
                                   'th cTRS\n'])
                        end
                       MA.cont.good = MA.cont.good ||MA.cont.goodt;
                        Vstate=29; 
                        break; 
                    end
                    if prt>=3
                     fprintf('continuous trust region was unsuccessful\n')
                   end 
                end
                Vstate=27;
            end
       end 
       if Vstate==29
          if norm(MA.cont.M,inf)>=5 % change
              MA.cont.s = zeros(nc, 1); MA.cont.M = eye(nc); 
              MA.cont.sigma=1;
          end
          if ni>0, Vstate=30; else, Vstate=2;end
       end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%
    %  iMATRS: integer MATRS
    %%%%%%%%%%%%%%%%%%%%%%%%%
    if ni>0  
        if Vstate==30 
             MA.int.iit = MA.int.iit+1;  
%              if cputime-info.cpu0>=info.secmax
%                 info.error='secmax reached'; info.done=1;  
%                 return;
%              end
             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
             % phase I: Perform integer mutation
             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
             if prt>=2
                 fprintf(['start of ',num2str(MA.int.iit),...
                          'th iMutation\n'])
             end
             if ~MA.int.good, MA.int.OffspringPop0=MA.int.OffspringPop;
             else, MA.int.OffspringPop0 =[];
             end
             info.xbest0i=xbest; 
             MA.int.goodm=0; MA.int.OffspringPop=[];
             Vstate=31; 
             MA.int.im=1;
        end
        while Vstate<42
            if Vstate==31 
                MA.int.Offspring.sd = MA.int.D(:,MA.int.im); 
                MA.int.Offspring.p = ...
                                round(MA.int.M*MA.int.Offspring.sd);
                MA.int.ixbest = xbest(info.iI);       
                [MA,info] = reqintLSS(MA,info);
                if ~MA.int.feasible
                     MA.int.im=MA.int.im+1;
                      if MA.int.im>size(MA.int.D,2)
                       Vstate=42;
                         break;% end of the mutation phase
                      else
                         Vstate=31; 
                         % go back to continue the mutation phase
                         continue
                      end
                else
                    info.yworst  = xbest;
                    info.yworst(info.iI)  = max(info.ilow,min(...
                                    info.iupp,xbest(info.iI)+...
                                    MA.int.alp0 * MA.int.Offspring.p));
                    Vstate=32;
                    x=info.yworst;
                    return;
                end
            end
           if Vstate==32 
                sxf  = size(info.xf,1);
                diff = ... 
                    (info.xf(:,1:dim)-repmat(info.yworst',sxf,1)).^2;
                [mn,ind]=min(sum(diff,2));
                if (mn<=10^-16)
                    fval   = info.xf(ind(1),dim+1); info.fworst = fval;
                else
                    info.fworst= f; 
                    info.xf(info.nf,1:dim)=info.yworst';
                    info.xf(info.nf,dim+1)=info.fworst;
                    info=UpdatePoints(info);
                     
                end
                % cicle for updating alpha
                if info.fworst<fbest
                    % initialize alpha and best point
                    MA.int.alpha = MA.int.alp0; info.ytrial=info.yworst;
                    info.ftrial=info.fworst;
                    % calculate trial point
                    info.yworst = xbest;
                    beta   = min(MA.int.alpmax,itune.inu*MA.int.alpha);
                    info.yworst(info.iI) = ...
                         max(info.ilow,min(info.iupp,...
                         xbest(info.iI) + beta * MA.int.Offspring.p));
                    Vstate     = 33; 
                    x=info.yworst;
                    return;
                else
                    MA.int.alpha = 0; info.ytrial = xbest;
                    info.ftrial = Inf;
                    Vstate=36;
                end
           end
           if Vstate==33 
                sxf  = size(info.xf,1);
                diff = ...
                   (info.xf(:,1:dim)-repmat(info.yworst',sxf,1)).^2;
                [mn,ind]=min(sum(diff,2));
                if (mn<=10^-16)
                    fval   = info.xf(ind(1),dim+1); info.fworst = fval;
                else
                    info.fworst= f; 
                    info.xf(info.nf,1:dim)=info.yworst';
                    info.xf(info.nf,dim+1)=info.fworst;
                    info=UpdatePoints(info);
                     
                end
                Vstate=34; 
           end
           while Vstate==34||Vstate==35 
                if Vstate==34 
                    % expansion step (increase stepsize)
                    if (MA.int.alpha<MA.int.alpmax)&&(info.fworst<fbest)
                        % MA.int.alpha calulation and best point updating
                        MA.int.alpha = ...
                           min(MA.int.alpmax,itune.inu*MA.int.alpha);
                        % best point updating
                        info.ytrial=info.yworst;
                        info.ftrial=info.fworst;
                        % next point to be tested
                        if(MA.int.alpha < MA.int.alpmax)
                           info.yworst  = xbest; 
                           beta    = ...
                              min(MA.int.alpmax,itune.inu*MA.int.alpha);
                           info.yworst(info.iI) = max(info.ilow,...
                               min(info.iupp,xbest(info.iI) ...
                               + beta * MA.int.Offspring.p));
                           Vstate=35; 
                           x=info.yworst;
                           return;
                        end
                    else
                        Vstate=41; break; 
                    end
               end
               if Vstate==35 
                    sxf  = size(info.xf,1);
                    diff = ...
                     (info.xf(:,1:dim)-repmat(info.yworst',sxf,1)).^2;
                    [mn,ind]=min(sum(diff,2));
                    if (mn<=10^-16)
                        fval = info.xf(ind(1),dim+1);
                        info.fworst = fval;
                    else
                        info.fworst= f; 
                        info.xf(info.nf,1:dim)=info.yworst';
                        info.xf(info.nf,dim+1)=info.fworst;
                        info=UpdatePoints(info);
                         
                    end
                       Vstate=34; 
               end
           end
           if Vstate==36 
                MA.int.Offspring.p = -MA.int.Offspring.p; 
                MA.int.ixbest      = xbest(info.iI);       
                [MA,info]         = reqintLSS(MA,info);
                if ~MA.int.feasible
                     MA.int.im=MA.int.im+1;
                      if MA.int.im>size(MA.int.D,2)
                         Vstate=42; 
                         break;% end of the mutation phase
                      else
                         Vstate=31; 
                         % go back to continue the mutation phase
                         continue
                      end
                else
                    info.fworst0   = info.fworst;
                    info.yworst    = xbest;
                    info.yworst(info.iI) = max(info.ilow,min(...
                                    info.iupp,xbest(info.iI) + ...
                                    MA.int.alp0 * MA.int.Offspring.p));
                    Vstate  = 37; 
                    x=info.yworst;
                    return;
                end
           end
           if Vstate==37 
                sxf  = size(info.xf,1);
                diff = (info.xf(:,1:dim)-repmat(info.yworst',sxf,1)).^2;
                [mn,ind]=min(sum(diff,2));
                if (mn<=10^-16)
                    fval   = info.xf(ind(1),dim+1); info.fworst = fval;
                else
                    info.fworst= f; 
                    info.xf(info.nf,1:dim)=info.yworst';
                    info.xf(info.nf,dim+1)=info.fworst;
                    info=UpdatePoints(info);
                     
                end
                if info.fworst<fbest
                    MA.int.alpha = MA.int.alp0; info.ytrial=info.yworst;
                    info.ftrial=info.fworst;
                    info.yworst = xbest;
                    beta   = min(MA.int.alpmax,itune.inu*MA.int.alpha);
                    info.yworst(info.iI) = ...
                        max(info.ilow,min(info.iupp,...
                        xbest(info.iI) + beta * MA.int.Offspring.p));
                    Vstate=38;
                    x=info.yworst;
                    return;
                else
                     MA.int.Offspring.sd = -MA.int.Offspring.sd;
                     MA.int.alpha = 0; info.ytrial = xbest; 
                     info.ftrial = Inf; 
                     Vstate=41;
                     if info.fworst0<info.fworst
                         info.fworst=info.fworst0; 
                     end
                end
            end
            if Vstate==38 
                sxf  = size(info.xf,1);
                diff = (info.xf(:,1:dim)-repmat(info.yworst',sxf,1)).^2;
                [mn,ind]=min(sum(diff,2));
                if (mn<=10^-16)
                    fval = info.xf(ind(1),dim+1); info.fworst = fval;
                else
                    info.fworst= f; 
                    info.xf(info.nf,1:dim)=info.yworst';
                    info.xf(info.nf,dim+1)=info.fworst;
                    info=UpdatePoints(info);
                     
                end
                Vstate=39; 
            end
            while Vstate==39||Vstate==40  
                if Vstate==39 % 
                    % expansion step (increase stepsize)
                    okstep = (MA.int.alpha<MA.int.alpmax && ...
                              info.fworst < fbest);
                    if okstep
                        % MA.int.alpha calulation and best point updating
                        MA.int.alpha = ...
                           min(MA.int.alpmax, itune.inu*MA.int.alpha);
                        % best point updating
                        info.ytrial=info.yworst; 
                        info.ftrial=info.fworst;
                        % next point to be tested
                        if(MA.int.alpha < MA.int.alpmax)
                           info.yworst = xbest;  
                           beta  = ...
                              min(MA.int.alpmax,itune.inu*MA.int.alpha);
                          
                           info.yworst(info.iI) = ...
                           max(info.ilow,min(info.iupp,...
                           xbest(info.iI) + beta * MA.int.Offspring.p));
                           Vstate=40;
                           x=info.yworst;
                           return;
                        end
                    else
                        Vstate=41; break;
                    end
                end
                if Vstate==40 
                    sxf  = size(info.xf,1);
                    diff = ...
                     (info.xf(:,1:dim)-repmat(info.yworst',sxf,1)).^2;
                    [mn,ind]=min(sum(diff,2));
                    if (mn<=10^-16)
                        fval = info.xf(ind(1),dim+1); 
                        info.fworst = fval;
                    else
                        info.fworst = f; 
                        info.xf(info.nf,1:dim)=info.yworst';
                        info.xf(info.nf,dim+1)=info.fworst;
                        info=UpdatePoints(info);
                         
                    end
                   Vstate=39; 
                end
            end
             if Vstate==41
                 if MA.int.alpha>0
                      info.niLS =info.niLS+1;
                      MA.int.Offspring.f  = info.ftrial; 
                      MA.int.OffspringPop = ...
                             [MA.int.OffspringPop MA.int.Offspring];
                      if(info.ftrial < fbest)
                         MA.int.a(MA.int.im) = MA.int.alpha;
                         fbest = info.ftrial;
                         xbest = info.ytrial; MA.int.goodm=1;
                         if prt>=3
                           fprintf(['continuous line search was',...
                                    ' successful\n'])
                         end
                      else
                          MA.int.a(MA.int.im) = MA.int.sigma;
                      end
                 else
                      MA.int.a(MA.int.im) = MA.int.sigma;
                      MA.int.Offspring.f  = info.fworst;
                      MA.int.OffspringPop = ...
                              [MA.int.OffspringPop MA.int.Offspring];
                 end
                  if MA.int.feasible
                      MA.int.im=MA.int.im+1;
                      if MA.int.im>size(MA.int.D,2)
                          Vstate=42; 
                          break;% end of the mutation phase
                      else % go back to continue the mutation phase
                          Vstate=31; 
                      end
                  end
             end
        end
       if Vstate==42 
            changDi = (norm(xbest(info.iI)-info.xbest0i(info.iI))==0);
            if ~changDi
               MA.int.pinit = info.xbest0i(info.iI)-xbest(info.iI);
               MA.int.pinit = ceil(5*MA.int.pinit/norm(MA.int.pinit,inf));
                if MA.int.dir<30 % 30,
                    MA.int.dir=MA.int.dir+1;
                else, MA.int.dir=2;
                end
                MA.int.D = iusequence(MA,itune.ilambda,ni);
                MA.int.s = zeros(ni, 1); MA.int.M = eye(ni);
            else
                MA.int.dir = 1;
               if itune.ilambda<=ni
                   MA.int.D   = eye(ni,itune.ilambda);
                   perm = randperm(itune.ilambda);
                   MA.int.D   = MA.int.D(:,perm);
               else
                   ii=0;
                   while ii<10
                       ii=ii+1;
                        MA.int.pinit = randi([-5,5],ni,1);
                        if norm(MA.int.pinit)~=0, break; end
                   end
                   if itune.ilambda==2*dim
                        MA.int.D   = [eye(ni,ni) -eye(ni,ni)];
                        perm = randperm(2*ni);
                        MA.int.D   = MA.int.D(:,perm);
                   elseif itune.ilambda>2*dim
                      MA.int.D = [eye(ni,ni) -eye(ni,ni) ...
                       iusequence(MA,itune.ilambda-2*ni,ni)];
                      perm = randperm(itune.ilambda);
                      MA.int.D   = MA.int.D(:,perm);
                   else
                      MA.int.D= [eye(ni,ni) ...
                      iusequence(MA,itune.ilambda-ni,ni)]; 
                      perm = randperm(itune.ilambda);
                      MA.int.D   = MA.int.D(:,perm);
                   end
               end
            end
            MA.int.lambda = length(MA.int.OffspringPop);
           if ~MA.int.good && MA.int.lambda<6
               MA.int.OffspringPop = ...
                   [MA.int.OffspringPop0,MA.int.OffspringPop];
           end
           MA.int.lambda = length(MA.int.OffspringPop);
           MA.int.good = MA.int.goodm;
           if prt>=2
               fprintf(['end of ',num2str(MA.int.iit),'th iMutation\n'])
           end
           MA.int.okTR = ~MA.int.goodm; MA.int.goodr=0; 
           if MA.int.lambda>=1
             imu       = ceil(MA.int.lambda/2);  
             iwi_raw   = log(MA.int.lambda/2 + 0.5) - log((1:imu));
             iwi       = iwi_raw/sum(iwi_raw);
             imu_eff   = max(1+eps,1/sum(iwi .^2));
             ic_s      = min(1.999,(imu_eff + 2)/(ni + imu_eff + 5));
             ic_1      = 2/((ni+1.3)^2 + imu_eff);
             ic_mu     = .... 
             min(1-ic_1, 2*(imu_eff-2+1/imu_eff)/((ni+2)^2+imu_eff));
             id_s      = 1 + ic_s + 2*max(0,sqrt((imu_eff-1)/(ni+1))-1);
             isqrt_s   = sqrt(ic_s*(2-ic_s)*imu_eff);
             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
             % phase II: perform selection
             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
              if prt>=2
                  fprintf(['start of ',num2str(MA.int.iit),...
                          'th selection\n'])
              end
             iranks = RankPop(MA.int.OffspringPop, ordering);
             sumzi  = zeros(ni, 1);
             sumdi  = zeros(ni, 1);
             for m = 1:imu
                 sumzi = sumzi + iwi(m)*MA.int.OffspringPop(iranks(m)).sd;
                 sumdi = sumdi + iwi(m)*MA.int.OffspringPop(iranks(m)).p;
              end
               if prt>=2
                 fprintf(['end of ',num2str(MA.int.iit),'th selection\n'])
              end
              MA.int.s    = (1-ic_s)*MA.int.s + isqrt_s*sumzi; 
              etaTR = 1 - 0.5*ic_1 - 0.5*ic_mu;
              MA.int.M    = etaTR* MA.int.M + ...
                        (0.5*ic_1)*(MA.int.M*MA.int.s)*MA.int.s';
              for m = 1:imu
                 u1 = MA.int.OffspringPop(iranks(m)).p; 
                 u2 = MA.int.OffspringPop(iranks(m)).sd; 
                 MA.int.M = MA.int.M + ((0.5*ic_mu*iwi(m))*u1)*u2';
              end  
              MA.int.M=adjustY(MA.int.M,itune);
              pow = norm(MA.int.s)/MA.int.echi - 1;
              MA.int.sigma = MA.int.sigma*exp((ic_s/id_s)*pow);
              MA.int.sigma = ...
                      round(min(itune.isigmamax,max(1,MA.int.sigma)));
              MA.int.okRec = norm(sumdi)~=0 && imu>=3&& ~MA.int.goodm; 
              Vstate=43; 
           else
              Vstate=54; 
           end
       end
       if Vstate==43
          if MA.int.okRec
              if prt>=2
                 fprintf(['start of ',num2str(MA.int.iit),...
                     'th iRecombination\n'])
              end
             MA.int.ixbest = xbest(info.iI);  ni = length(info.iI);
             okRecombonation=1; MA.int.goodr=0; ir=0; beta=1;
            while okRecombonation
               ir=ir+1; 
               if ir>1, beta = 2*rand(ni,1)-1; beta=beta/norm(beta); end
               MA.int.drec = ceil(beta.*sumdi);
               okRecombonation = (norm(MA.int.drec)~=0);
               if okRecombonation
                   % initialize vector alpha_max
                   ind = (MA.int.drec > 0 & MA.int.ixbest<info.iupp);
                   aupp = min((info.iupp(ind)-MA.int.ixbest(ind))...
                          ./MA.int.drec(ind));
                   if isempty(aupp), aupp=inf; end
                   ind = (MA.int.drec < 0 & MA.int.ixbest<info.ilow);
                   alow = min((info.ilow(ind)-MA.int.ixbest(ind))...
                         ./MA.int.drec(ind));
                   if isempty(alow), alow=inf; end
                   MA.int.alpmax = max(1,floor(min(alow,aupp)));
                   if isfinite(MA.int.alpmax)&& MA.int.alpmax>0 
                       Vstate=44; 
                       break;
                   end
               end
               if ir>=100
                   Vstate=54;
                   break;
               else
                   Vstate=44;
               end
            end
         else
             Vstate=54;
         end
      end
      if Vstate==44 
        MA.int.alp0   = max(1,round(sqrt(MA.int.alpmax*MA.int.sigma)));
        info.yworst  = xbest;
        info.yworst(info.iI)  = max(info.ilow,min(info.iupp,...
                       xbest(info.iI) + MA.int.alp0 * MA.int.drec));
        Vstate=45;
        x=info.yworst;
        return;
      end
      if Vstate==45 
        sxf  = size(info.xf,1);
        diff = (info.xf(:,1:dim)-repmat(info.yworst',sxf,1)).^2;
        [mn,ind]=min(sum(diff,2));
        if (mn<=10^-16)
            fval   = info.xf(ind(1),dim+1); info.fworst = fval;
        else
            info.fworst= f; 
            info.xf(info.nf,1:dim)=info.yworst';
            info.xf(info.nf,dim+1)=info.fworst;
            info=UpdatePoints(info);
             
        end
        Vstate=46; 
      end
      if Vstate==46 
            if info.fworst<fbest
                MA.int.alpha = MA.int.alp0; info.ytrial=info.yworst; 
                info.ftrial=info.fworst;
                info.yworst  = xbest;
                beta    = min(MA.int.alpmax,itune.inu*MA.int.alpha);
                info.yworst(info.iI) = max(info.ilow,min(info.iupp,...
                                   xbest(info.iI) + beta * MA.int.drec));
                Vstate=47; 
                x=info.yworst;
                return;
            else
                Vstate=477; 
            end
      end
      if Vstate==47 
        sxf  = size(info.xf,1);
        diff = (info.xf(:,1:dim)-repmat(info.yworst',sxf,1)).^2;
        [mn,ind]=min(sum(diff,2));
        if (mn<=10^-16)
            fval   = info.xf(ind(1),dim+1); info.fworst = fval;
        else
            info.fworst= f; 
            info.xf(info.nf,1:dim)=info.yworst';
            info.xf(info.nf,dim+1)=info.fworst;
            info=UpdatePoints(info);
             
        end
        Vstate=48; 
      end
      while Vstate==48||Vstate==49 
            if Vstate==48 
                % expansion step (increase stepsize)
                if (MA.int.alpha<MA.int.alpmax) && (info.fworst < fbest)
                    MA.int.alpha = ...
                           min(MA.int.alpmax,itune.inu*MA.int.alpha);
                    % best point updating
                    info.ytrial=info.yworst; info.ftrial=info.fworst;
                    % next point to be tested
                    if(MA.int.alpha < MA.int.alpmax)
                       info.yworst     = xbest; 
                       beta = min(MA.int.alpmax,itune.inu*MA.int.alpha);
                       info.yworst(info.iI) = max(info.ilow,min(...
                           info.iupp,xbest(info.iI)+beta*MA.int.drec));
                       Vstate=49;
                       x=info.yworst;
                       return;
                    end
                else
                    Vstate=53; 
                    
                    break; 
                end
            end
           if Vstate==49 
                sxf  = size(info.xf,1);
                diff = (info.xf(:,1:dim)-repmat(info.yworst',sxf,1)).^2;
                [mn,ind]=min(sum(diff,2));
                if (mn<=10^-16)
                    fval   = info.xf(ind(1),dim+1); info.fworst = fval;
                else
                    info.fworst= f; 
                    info.xf(info.nf,1:dim)=info.yworst';
                    info.xf(info.nf,dim+1)=info.fworst;
                    info=UpdatePoints(info);
                     
                end
               Vstate=48; 
           end
      end  
      if Vstate==477
           MA.int.drec  = -MA.int.drec;
           ind = (MA.int.drec > 0 & MA.int.ixbest<info.iupp);
           aupp = min( (info.iupp(ind)-MA.int.ixbest(ind))...
                  ./ MA.int.drec(ind));
           if isempty(aupp), aupp=inf; end
           ind = ( MA.int.drec < 0 & MA.int.ixbest>info.ilow);
           alow=min((info.ilow(ind) - MA.int.ixbest(ind))...
                 ./ MA.int.drec(ind));
           if isempty(alow), alow=inf; end
           MA.int.alpmax = max(1,floor(min(alow,aupp)));
           if MA.int.alpmax~=0 && ~isinf(MA.int.alpmax)
               MA.int.alp0 = min(MA.int.alpmax,MA.int.sigma);
               info.yworst   = xbest;
               info.yworst(info.iI)  = ...
                        max(info.ilow,min(info.iupp,...
                        xbest(info.iI) + MA.int.alp0 * MA.int.drec));
               Vstate=49; 
               x=info.yworst;
               return;
           else
               Vstate=53; 
           end
      end
      if Vstate==49 
            sxf  = size(info.xf,1);
            diff = (info.xf(:,1:dim)-repmat(info.yworst',sxf,1)).^2;
            [mn,ind]=min(sum(diff,2));
            if (mn<=10^-16)
                fval   = info.xf(ind(1),dim+1); info.fworst = fval;
            else
                info.fworst= f; 
                info.xf(info.nf,1:dim)=info.yworst';
                info.xf(info.nf,dim+1)=info.fworst;
                info=UpdatePoints(info);
                 
            end
            if info.fworst<fbest
                % initialize MA.int.alpha and best point
                MA.int.alpha = MA.int.alp0; info.ytrial=info.yworst; 
                info.ftrial=info.fworst;
                % calculate trial point
                info.yworst  = xbest;
                beta  = min(MA.int.alpmax,itune.inu*MA.int.alpha);
                info.yworst(info.iI) = ...
                           max(info.ilow,min(info.iupp,...
                           xbest(info.iI) + beta * MA.int.drec));
                Vstate     = 50; 
                x=info.yworst;
                return;
            else
                Vstate     = 52; 
            end
      end
      if Vstate==50
            sxf  = size(info.xf,1);
            diff = (info.xf(:,1:dim)-repmat(info.yworst',sxf,1)).^2;
            [mn,ind]=min(sum(diff,2));
            if (mn<=10^-16)
                fval   = info.xf(ind(1),dim+1); info.fworst = fval;
            else
                info.fworst= f; 
                info.xf(info.nf,1:dim)=info.yworst';
                info.xf(info.nf,dim+1)=info.fworst;
                info=UpdatePoints(info);
                 
            end
            Vstate  = 51;
      end
      while Vstate==51||Vstate==52 
            if Vstate==51 
                % expansion step (increase stepsize)
                if (MA.int.alpha<MA.int.alpmax) && (info.fworst < fbest)
                    MA.int.alpha = ...
                         min(MA.int.alpmax, itune.inu*MA.int.alpha);
                    info.ytrial=info.yworst; info.ftrial=info.fworst;
                    % next point to be tested
                    if(MA.int.alpha < MA.int.alpmax)
                       info.yworst  = xbest;  
                       beta  = ...
                          min(MA.int.alpmax,itune.inu*MA.int.alpha);
                       info.yworst(info.iI) = max(info.ilow,min(...
                          info.iupp,xbest(info.iI)+beta*MA.int.drec));
                       Vstate=52;
                       x=info.yworst;
                       return;
                    end
                else
                    Vstate=53;break;
                end
            end
           if Vstate==52 
                sxf  = size(info.xf,1);
                diff = (info.xf(:,1:dim)-repmat(info.yworst',sxf,1)).^2;
                [mn,ind]=min(sum(diff,2));
                if (mn<=10^-16)
                    fval = info.xf(ind(1),dim+1); info.fworst = fval;
                else
                    info.fworst= f; 
                    info.xf(info.nf,1:dim)=info.yworst';
                    info.xf(info.nf,dim+1)=info.fworst;
                    info=UpdatePoints(info);
                     
                end
                Vstate=51; 
           end
      end  
      if Vstate==53                      
          if MA.int.alpha>0
              if(info.ftrial < fbest)
                  info.niLS =info.niLS+1;
                   fbest = info.ftrial; xbest = info.ytrial; 
                   MA.int.goodr=1;
                   if prt>=3
                       fprintf('integer line search was successful\n')
                   end
              end
         else
              if(info.ftrial < fbest)
                if prt>=3
                   fprintf('integer line search was successful\n')
                end 
                fbest  = info.ftrial; xbest = info.ytrial;
                MA.int.goodr=1;
              end
           end
           if prt>=2
              fprintf(['end of ',num2str(MA.int.iit),...
                      'th iRecombination\n'])
           end
          MA.int.good   = MA.int.good ||MA.int.goodr;
          MA.int.okTR   = ~MA.int.okRec||~MA.int.goodr;
          Vstate  = 54; 
      end
      if Vstate==54 
           MA.int.Deli  = min(30,max(10,MA.int.sigma)); 
           MA.int.okTR= MA.int.okTR && info.nf>=dim+1;
           if MA.int.okTR % perform integer trust region algorithm
               if prt>=2
                 fprintf(['start of ',num2str(MA.int.iit),'th iTRS\n'])
               end
                MA.int.istuck = 0; MA.int.isuccTR=0; 
                MA.int.iunsuccTR=0; 
                MA.int.M = adjustY(MA.int.M,itune);
                if rank(MA.int.M) < size(MA.int.M,2)
                     Vstate=57; 
                else
                     Vstate = 55; 
                     warning off
                     MA.int.invM = inv(MA.int.M);
                     warning on
                     MA.int.changePTR=0;
                 end
           else
               Vstate=57; 
               
           end
     end
     while Vstate==55||Vstate==56 
            if Vstate==55

                if info.nf>=dim+1&& MA.int.changePTR
                   I = randperm(info.nf,dim+1);
                   XXi = info.xf(I,1:dim)';
                   FFi = info.xf(I,dim+1)';
               else
                 XXi = info.xf(info.nf-dim:info.nf,1:dim)';
                 FFi = info.xf(info.nf-dim:info.nf,dim+1)';
               end
               MA.int.ixbest = xbest(info.iI);
               MA.int.g = fitGrad(ni,XXi(info.iI,:),FFi,MA.int.ixbest,...
                        fbest,ni+1,itune);
               ll     = max(info.ilow-MA.int.ixbest,-MA.int.Deli);
               uu     = min(info.iupp-MA.int.ixbest,MA.int.Deli);
               igls   = 0;
                for i = 1 : ni
                    if ll(i) >= uu(i), igls=1; end
                end
                if ~igls
                   r = -MA.int.invM*MA.int.g;
                   U = qr([MA.int.M,r]);
                   R = triu(U(1:ni,1:ni));
                   y = U(1:ni,ni+1);
                   MA.int.sci = obils_search(R,y,ll,uu);
                   if isempty(MA.int.sci)||norm(MA.int.sci)==0
                       ptr = MA.int.ixbest-XXi(info.iI,1);
                       MA.int.sci = MA.int.Deli*round(ptr/norm(ptr));
                       if isempty(MA.int.sci)||norm(MA.int.sci)==0
                           Vstate=57; 
                           break
                       end
                   end
                else
                    Vstate=57;  break
                end
                info.yworst     = xbest;
                info.yworst(info.iI) = min(info.iupp,max(info.ilow,...
                              MA.int.ixbest+MA.int.sci));
                sxf  = size(info.xf,1);
                diff = (info.xf(:,1:dim)-repmat(info.yworst',sxf,1)).^2;
                [mn,~]     = min(sum(diff,2));
                if (mn>10^-16)
                   Vstate=56; MA.int.changePTR=0;
                   x=info.yworst; 
                   return;
                else
                     MA.int.istuck=MA.int.istuck+1; MA.int.changePTR=1;
                     if sign(rand-0.5)
                        MA.int.Deli = ...
                        abs(MA.int.Deli+sign(rand-0.5)*randi([1,10],1));
                     else
                         MA.int.Deli = round(MA.int.Deli/randi([1,10],1));
                     end
                     if MA.int.Deli<=0||MA.int.istuck>itune.iDeltainit
                        Vstate=57; 
                        break; 
                     end
                     Vstate=55;
                     continue
                end
            end
            if Vstate==56 
               [info.fworst] = f; 
               info.xf(info.nf,1:dim)=info.yworst';
               info.xf(info.nf,dim+1)=info.fworst;
               info=UpdatePoints(info);
                
               gisci    = max(-MA.int.g'*MA.int.sci,...
                          eps*abs(MA.int.g)'*abs(MA.int.sci));
                mueff   = (fbest-info.fworst)/gisci;
                suciTR  = (mueff*abs(mueff-1) > itune.izeta);
                if suciTR 
                   if prt>=3
                      fprintf('integer trust region was successful\n')
                   end 
                   info.niTR =info.niTR+1;
                   fbest = info.fworst; xbest = info.yworst;  
                   MA.int.goodt=1;
                   MA.int.good = MA.int.good ||MA.int.goodt;
                   Vstate=57;  
                   MA.int.isuccTR=MA.int.isuccTR+1;
                   break;
                else
                    MA.int.iunsuccTR = MA.int.iunsuccTR+1;
                    if MA.int.Deli<=itune.iDeltabar
                        MA.int.Deli=MA.int.Deli-1;
                    else
                        MA.int.Deli = ...
                            floor(min(norm(MA.int.sci,inf),...
                            MA.int.Deli)/itune.itheta);
                    end
                    if prt>=3
                       fprintf('integer trust region was unsuccessful\n')
                   end 
                end
                if MA.int.Deli<=0
                    if prt>=2
                       fprintf(['end of ',num2str(MA.int.iit),...
                                'th iTRS\n'])
                    end
                    Vstate=57; 
                    break; 
                end
                Vstate=55; 
            end
     end
     if Vstate==57
       if norm(MA.int.M,inf)>=5
         MA.int.s = zeros(ni, 1); MA.int.M = eye(ni); MA.int.sigma=1;
       end
       if nc>0 && ni==0
          Vstate=2; 
       elseif ni>0 && nc==0
          Vstate = 30;  
       elseif ni>0 && nc>0
           Vstate = 58; 
       end
       
     end
    end
   
    
    
   if Vstate == 58
        p0   =  xbest-info.xbest0;
        changDic = norm(p0)==0;
        if ~changDic 
           MA.cont.p      = p0(info.cI); 
           MA.int.p       = p0(info.iI);
           MA.mixed.p     = zeros(dim,1);
           MA.mixed.p(info.iI) = MA.int.p; 
           MA.mixed.p(info.cI) = MA.cont.p;
           if norm(MA.int.p)~=0||norm(MA.cont.p)~=0
            [MA,info]= requirMixedLSS(xbest,MA,info);
             Vstate=59;
          else
             Vstate=2;
           end
        else
            MA.int.p=[]; MA.cont.p=[];
            for kk=1:length(info.F)-3
                p0           = info.X(:,1)-info.X(:,end-kk-1);
                MA.cont.p    = p0(info.cI); 
                MA.int.p     = p0(info.iI);
                MA.mixed.p   = zeros(dim,1); 
                MA.mixed.p(info.iI) = MA.int.p; 
                MA.mixed.p(info.cI) = MA.cont.p;
                if norm(MA.mixed.p)~=0, break; end
            end
            if isempty(MA.int.p)&&isempty(MA.cont.p)
                Vstate=2;
            else
                 if norm(MA.int.p)~=0 || norm(MA.cont.p)~=0
                   [MA,info]= requirMixedLSS(xbest,MA,info);
                    Vstate=59; 
                 else
                     Vstate=2;
                 end
            end
        end
   end
   if Vstate==59
       if MA.int.feasible&&MA.cont.feasible
             info.ytrial=xbest; info.ftrial=fbest; 
             MA.cont.alpha = 0; MA.int.alpha = 0;
            info.yworst  = xbest;
            info.yworst(info.cI)  = ...
                            max(info.clow,min(info.cupp,...
                            xbest(info.cI)+MA.cont.alp0 * MA.cont.p));
            info.yworst(info.iI)  = ...
                           max(info.ilow,min(info.iupp,...
                           xbest(info.iI)+MA.int.alp0 * MA.int.p));
            Vstate=60;
            x=info.yworst;
            return;           
       elseif MA.int.feasible
             info.ytrial=xbest; info.ftrial=fbest; 
             MA.int.alpha = 0;
            info.yworst  = xbest;
            info.yworst(info.iI)  = ...
                           max(info.ilow,min(info.iupp,...
                           xbest(info.iI)+MA.int.alp0 * MA.int.p));
            Vstate=60;
            x=info.yworst;
            return;  
       elseif MA.cont.feasible
             info.ytrial=xbest; info.ftrial=fbest; 
             MA.cont.alpha = 0; 
            info.yworst  = xbest;
            info.yworst(info.cI)  = ...
                            max(info.clow,min(info.cupp,...
                            xbest(info.cI)+MA.cont.alp0 * MA.cont.p));
            Vstate=60;
            x=info.yworst;
            return;
       else
           Vstate=2;
       end
   end
   if Vstate==60
        sxf=size(info.xf,1);
        diff=(info.xf(:,1:dim)-repmat(info.yworst',sxf,1)).^2;
        [mn,ind]=min(sum(diff,2));
        if (mn<=10^-16)
            fval = info.xf(ind(1),dim+1); info.fworst = fval;
        else
            info.fworst    = f; 
            info.xf(info.nf,1:dim)=info.yworst'; 
            info.xf(info.nf,dim+1)=info.fworst;
            info=UpdatePoints(info);
        end
        Vstate=62;
   end
  if Vstate==62
    if info.fworst<fbest
        % initialize alpha and best point
        if MA.int.feasible&&MA.cont.feasible
            MA.cont.alpha = MA.cont.alp0; MA.int.alpha = MA.int.alp0;
            info.ytrial=info.yworst; info.ftrial=info.fworst;
            % calculate trial point
            info.yworst= xbest;
            betac = min(MA.cont.alpmax,ctune.cnu*MA.cont.alpha);
            info.yworst(info.cI) = max(info.clow,min(info.cupp,...
                               xbest(info.cI) + betac * MA.cont.p));

            betai      = min(MA.int.alpmax,itune.inu*MA.int.alpha);
            info.yworst(info.iI) = max(info.ilow,min(info.iupp,...
                               xbest(info.iI) + betai * MA.int.p));
           
        elseif MA.int.feasible
            MA.cont.alpha = inf; MA.int.alpha = MA.int.alp0;
            info.ytrial=info.yworst; info.ftrial=info.fworst;
            % calculate trial point
            info.yworst= xbest;
           
            betai      = min(MA.int.alpmax,itune.inu*MA.int.alpha);
            info.yworst(info.iI) = max(info.ilow,min(info.iupp,...
                               xbest(info.iI) + betai * MA.int.p));
           
            
        elseif MA.cont.feasible
            MA.cont.alpha = MA.cont.alp0; MA.int.alpha = inf;
            info.ytrial=info.yworst; info.ftrial=info.fworst;
            % calculate trial point
            info.yworst= xbest;
            betac = min(MA.cont.alpmax,ctune.cnu*MA.cont.alpha);
            info.yworst(info.cI) = max(info.clow,min(info.cupp,...
                               xbest(info.cI) + betac * MA.cont.p));
           
        end
         Vstate=63;
         x=info.yworst;
         return;
    else
        Vstate=70;
    end
  end
  if Vstate==63
    sxf=size(info.xf,1);
    diff=(info.xf(:,1:dim)-repmat(info.yworst',sxf,1)).^2;
    [mn,ind]=min(sum(diff,2));
    if (mn<=10^-16)
        fval = info.xf(ind(1),dim+1); info.fworst = fval;
    else
        info.info.fworst   = f; 
        info.xf(info.nf,1:dim)=info.yworst'; 
        info.xf(info.nf,dim+1)=info.fworst;
        info=UpdatePoints(info);
    end
    
    Vstate=64; 
  end
  while Vstate==64||Vstate==65 
        if Vstate==64
            okLS = ((MA.int.alpha<MA.int.alpmax && ...
                    MA.cont.alpha<MA.cont.alpmax )&& ...
                    info.fworst < fbest);
            % expansion step (increase stepsize)
            if okLS
                MA.cont.alpha = ...
                       min(MA.cont.alpmax, ctune.cnu*MA.cont.alpha);
                MA.int.alpha  = ...
                       min(MA.int.alpmax, itune.inu*MA.int.alpha);
                info.ytrial=info.yworst; info.ftrial=info.fworst;
                % next point to be tested
                okstep = (MA.int.alpha<MA.int.alpmax && ...
                          MA.cont.alpha<MA.cont.alpmax);
                if okstep
                   info.yworst     = xbest; 
                   betac  = min(MA.cont.alpmax,ctune.cnu*MA.cont.alpha);
                   info.yworst(info.cI) = ...
                                max(info.clow,min(info.cupp,...
                                xbest(info.cI) + betac * MA.cont.p));
                   betai   = min(MA.int.alpmax,itune.inu*MA.int.alpha);
                   info.yworst(info.iI) = ...
                               max(info.ilow,min(info.iupp,...
                               xbest(info.iI) + betai * MA.int.p));
                   Vstate = 65; 
                   x=info.yworst;
                   return;
                else
                    Vstate=70; 
                end
            else
                Vstate=66; break; 
            end
        end
       if Vstate==65
            sxf  = size(info.xf,1);
            diff = (info.xf(:,1:dim)-repmat(info.yworst',sxf,1)).^2;
            [mn,ind]=min(sum(diff,2));
            if (mn<=10^-16)
                fval   = info.xf(ind(1),dim+1); info.fworst = fval;
            else
                info.info.fworst= f; 
                info.xf(info.nf,1:dim)=info.yworst';
                info.xf(info.nf,dim+1)=info.fworst;
                info=UpdatePoints(info);
                 
            end
            Vstate=64; 
       end
  end  
  
   while Vstate==66||Vstate==67 
        if Vstate==66
            okLS = (MA.int.alpha<MA.int.alpmax && ...
                    MA.cont.alpha>=MA.cont.alpmax && ... 
                    info.fworst < fbest);
            if okLS
                 MA.int.alpha = ...
                         min(MA.int.alpmax, itune.inu*MA.int.alpha);
                 info.ytrial=info.yworst; info.ftrial=info.fworst;
                 if MA.int.alpha<MA.int.alpmax
                   info.yworst  = xbest; 
                   betai   = min(MA.int.alpmax,itune.inu*MA.int.alpha);
                   info.yworst(info.iI) = ...
                          max(info.ilow,min(info.iupp,...
                          xbest(info.iI)+betai * MA.int.p));
                   Vstate=67;
                   x=info.yworst;
                   return;
                 end
            else
                Vstate=68; break;
            end
        end
        if Vstate==67
            sxf=size(info.xf,1);
            diff=(info.xf(:,1:dim)-repmat(info.yworst',sxf,1)).^2;
            [mn,ind]=min(sum(diff,2));
            if (mn<=10^-16)
                fval = info.xf(ind(1),dim+1); info.fworst = fval;
            else
               info.fworst = f; 
               info.xf(info.nf,1:dim)=info.yworst'; 
               info.xf(info.nf,dim+1)=info.fworst;
               info=UpdatePoints(info);
            end
            
           Vstate=66;
        end
   end
   while Vstate==68||Vstate==69 
        if Vstate==68
            okLS =  (MA.int.alpha>=MA.int.alpmax && ...
                     MA.cont.alpha<MA.cont.alpmax && ...
                     info.fworst < fbest);
            if okLS
                  MA.cont.alpha = ...
                       min(MA.cont.alpmax,ctune.cnu*MA.cont.alpha);
                  info.ytrial = info.yworst; 
                  info.ftrial = info.fworst;
                 if MA.int.alpha<MA.int.alpmax
                   info.yworst = xbest;  
                   betac  = min(MA.cont.alpmax,ctune.cnu*MA.cont.alpha);
                   info.yworst(info.cI) = ...
                         max(info.clow,min(info.cupp,...
                         xbest(info.cI) + betac * MA.cont.p));
                   Vstate = 69;
                   x=info.yworst;
                   return;
                 end
            else
                Vstate=70;
                break;
            end
        end
        if Vstate==69
            sxf=size(info.xf,1);
            diff=(info.xf(:,1:dim)-repmat(info.yworst',sxf,1)).^2;
            [mn,ind]=min(sum(diff,2));
            if (mn<=10^-16)
                fval = info.xf(ind(1),dim+1); info.fworst = fval;
            else
               info.fworst = f; 
               info.xf(info.nf,1:dim)=info.yworst'; 
               info.xf(info.nf,dim+1)=info.fworst;
               info=UpdatePoints(info);
            end
            
           Vstate=68;
        end
   end
   
   
   if Vstate==70
     if MA.int.alpha>0 ||MA.cont.alpha>0
         if info.ftrial<fbest
              info.nmiLS =info.nmiLS+1;
              fbest = info.ftrial; xbest = info.ytrial;
         end
         Vstate=2;
     else
         Vstate=71;
     end
   end
   %%%%%%%%%%%%%%%%%%%%% opposite direction is tried %%%%%%%%%%%%%%%%%%
   if Vstate==71
      MA.int.p =- MA.int.p; MA.cont.p=-MA.cont.p; MA.mixed.p=-MA.mixed.p;
      [MA,info]= requirMixedLSS(xbest,MA,info);
       if MA.int.feasible&&MA.cont.feasible
            info.ytrial   = xbest; 
            info.ftrial   = fbest; 
            MA.cont.alpha  = 0; 
            MA.int.alpha   = 0;
            info.yworst   = xbest;
            info.yworst(info.cI)  = max(info.clow,min(info.cupp,...
                      xbest(info.cI)+MA.cont.alp0 * MA.cont.p));
            info.yworst(info.iI)  = max(info.ilow,min(info.iupp,...
                      xbest(info.iI)+MA.int.alp0 * MA.int.p));
            Vstate=72;
            x=info.yworst;
            return;
            
       elseif MA.int.feasible
           
             info.ytrial=xbest; info.ftrial=fbest; 
             MA.int.alpha = 0;
            info.yworst  = xbest;
            info.yworst(info.iI)  = ...
                           max(info.ilow,min(info.iupp,...
                           xbest(info.iI)+MA.int.alp0 * MA.int.p));
            Vstate=72;
            x=info.yworst;
            return;  
       elseif MA.cont.feasible
             info.ytrial=xbest; info.ftrial=fbest; 
             MA.cont.alpha = 0; 
            info.yworst  = xbest;
            info.yworst(info.cI)  = ...
                            max(info.clow,min(info.cupp,...
                            xbest(info.cI)+MA.cont.alp0 * MA.cont.p));
            Vstate=72;
            x=info.yworst;
            return;
       else
           Vstate=2;
       end
   end
   if Vstate==72
        sxf=size(info.xf,1);
        diff=(info.xf(:,1:dim)-repmat(info.yworst',sxf,1)).^2;
        [mn,ind]=min(sum(diff,2));
        if (mn<=10^-16)
            fval = info.xf(ind(1),dim+1); info.fworst = fval;
        else
            info.fworst    = f; 
            info.xf(info.nf,1:dim)=info.yworst'; 
            info.xf(info.nf,dim+1)=info.fworst;
            info=UpdatePoints(info);
        end
         
        Vstate=73;
  end
  if Vstate==73
   if info.fworst<fbest
        % initialize alpha and best point
        if MA.int.feasible&&MA.cont.feasible
            MA.cont.alpha = MA.cont.alp0; MA.int.alpha = MA.int.alp0;
            info.ytrial=info.yworst; info.ftrial=info.fworst;
            % calculate trial point
            info.yworst= xbest;
            betac = min(MA.cont.alpmax,ctune.cnu*MA.cont.alpha);
            info.yworst(info.cI) = max(info.clow,min(info.cupp,...
                               xbest(info.cI) + betac * MA.cont.p));

            betai      = min(MA.int.alpmax,itune.inu*MA.int.alpha);
            info.yworst(info.iI) = max(info.ilow,min(info.iupp,...
                               xbest(info.iI) + betai * MA.int.p));
           
        elseif MA.int.feasible
            MA.cont.alpha = inf; MA.int.alpha = MA.int.alp0;
            info.ytrial=info.yworst; info.ftrial=info.fworst;
            % calculate trial point
            info.yworst= xbest;
           
            betai      = min(MA.int.alpmax,itune.inu*MA.int.alpha);
            info.yworst(info.iI) = max(info.ilow,min(info.iupp,...
                               xbest(info.iI) + betai * MA.int.p));
           
            
        elseif MA.cont.feasible
            MA.cont.alpha = MA.cont.alp0; MA.int.alpha = inf;
            info.ytrial=info.yworst; info.ftrial=info.fworst;
            % calculate trial point
            info.yworst= xbest;
            betac = min(MA.cont.alpmax,ctune.cnu*MA.cont.alpha);
            info.yworst(info.cI) = max(info.clow,min(info.cupp,...
                               xbest(info.cI) + betac * MA.cont.p));
           
        end
         Vstate=74;
         x=info.yworst;
         return;
    else
        Vstate=2;
    end
  end
  if Vstate==74
    sxf=size(info.xf,1);
    diff=(info.xf(:,1:dim)-repmat(info.yworst',sxf,1)).^2;
    [mn,ind]=min(sum(diff,2));
    if (mn<=10^-16)
        fval = info.xf(ind(1),dim+1); info.fworst = fval;
    else
        info.fworst   = f; 
        info.xf(info.nf,1:dim)=info.yworst'; 
        info.xf(info.nf,dim+1)=info.fworst;
        info=UpdatePoints(info);
    end
    
    Vstate=75;
  end
  while Vstate==75||Vstate==76 
        if Vstate==75
            % expansion step (increase stepsize)
            okLS = (MA.int.alpha<MA.int.alpmax && ...
                    MA.cont.alpha<MA.cont.alpmax && ...
                    info.fworst < fbest);
            if okLS
                MA.cont.alpha = ...
                         min(MA.cont.alpmax, ctune.cnu*MA.cont.alpha);
                MA.int.alpha = ...
                         min(MA.int.alpmax, itune.inu*MA.int.alpha);
                info.ytrial=info.yworst; info.ftrial=info.fworst;

                okstep = (MA.int.alpha<MA.int.alpmax&&...
                          MA.cont.alpha<MA.cont.alpmax);
                if okstep
                   info.yworst = xbest; 
                   betac  = min(MA.cont.alpmax,ctune.cnu*MA.cont.alpha);
                   info.yworst(info.cI) = ...
                                  max(info.clow,min(info.cupp,...
                                  xbest(info.cI) + betac * MA.cont.p));
                   betai  = min(MA.int.alpmax,itune.inu*MA.int.alpha);
                   info.yworst(info.iI) = ...
                                  max(info.ilow,min(info.iupp,...
                                  xbest(info.iI) + betai * MA.int.p));
                   Vstate=76; 
                   x=info.yworst;
                    return;
                elseif(MA.cont.alpha < MA.cont.alpmax)
                   info.yworst = xbest;  
                   betac = min(MA.cont.alpmax,ctune.cnu*MA.cont.alpha);
                   info.yworst(info.cI) = ...
                                   max(info.clow,min(info.cupp,...
                                   xbest(info.cI) + betac * MA.cont.p));
                   Vstate=76; 
                   x=info.yworst;
                    return;
               elseif (MA.int.alpha < MA.int.alpmax)  
                   info.yworst  = xbest;  
                   betai  = min(MA.int.alpmax,itune.inu*MA.int.alpha);
                   info.yworst(info.iI) = ...
                                    max(info.ilow,min(info.iupp,...
                                    xbest(info.iI) + betai * MA.int.p));
                   Vstate=76; 
                   x=info.yworst;
                   return;
                else
                     Vstate=77; break; 
                end
            else
                Vstate=77; break; 
            end
        end
       if Vstate==76
            sxf    = size(info.xf,1);
            diff   = (info.xf(:,1:dim)-repmat(info.yworst',sxf,1)).^2;
            [mn,ind] = min(sum(diff,2));
            if (mn<=10^-16)
                fval   = info.xf(ind(1),dim+1); info.fworst = fval;
            else
                info.fworst    = f; 
                info.xf(info.nf,1:dim) = info.yworst';
                info.xf(info.nf,dim+1) = info.fworst;
                info=UpdatePoints(info);
                 
            end
            Vstate=75; 
       end
  end  
  while Vstate==77||Vstate==78
        if Vstate==77
            okLS = (MA.int.alpha<MA.int.alpmax) && ...
                   (MA.cont.alpha>=MA.cont.alpmax) && ...
                   (info.fworst < fbest);
            if okLS
                 MA.int.alpha = ...
                           min(MA.int.alpmax, itune.inu*MA.int.alpha);
                 % best point updating
                 info.ytrial = info.yworst; 
                 info.ftrial = info.fworst;
                 % next point to be tested
                 if MA.int.alpha<MA.int.alpmax
                   info.yworst  = xbest; 
                   betai  = min(MA.int.alpmax,itune.inu*MA.int.alpha);
                   info.yworst(info.iI) = ...
                                   max(info.ilow,min(info.iupp,...
                                   xbest(info.iI)+betai * MA.int.p));
                   Vstate     = 78;
                   x=info.yworst;
                   return;
                 end
            else
                Vstate = 79;
            end
        end
        if Vstate==78
            sxf  = size(info.xf,1);
            diff = (info.xf(:,1:dim)-repmat(info.yworst',sxf,1)).^2;
            [mn,ind] = min(sum(diff,2));
            if (mn<=10^-16)
                fval = info.xf(ind(1),dim+1); info.fworst = fval;
            else
               info.fworst = f; 
               info.xf(info.nf,1:dim) = info.yworst'; 
               info.xf(info.nf,dim+1) = info.fworst;
               info=UpdatePoints(info);
            end
            
           Vstate=77;
        end
   end
   while Vstate==79||Vstate==80 
        if Vstate==79
            okLS = (MA.int.alpha>=MA.int.alpmax && ...
                    MA.cont.alpha<MA.cont.alpmax && ...
                   info.fworst < fbest);
            if  okLS
                  MA.cont.alpha = ...
                         min(MA.cont.alpmax,ctune.cnu*MA.cont.alpha);
                  info.ytrial = info.yworst; 
                  info.ftrial = info.fworst;
                 if MA.int.alpha<MA.int.alpmax
                   info.yworst = xbest;  
                   betac  = min(MA.cont.alpmax,ctune.cnu*MA.cont.alpha);
                   info.yworst(info.cI) = ...
                            max(info.clow,min(info.cupp,...
                            xbest(info.cI) + betac * MA.cont.p));
                   Vstate=80;
                   x=info.yworst;
                    return;
                 end
            else
                Vstate=81; break;
            end
        end
        if Vstate==80
            sxf  = size(info.xf,1);
            diff = (info.xf(:,1:dim)-repmat(info.yworst',sxf,1)).^2;
            [mn,ind]=min(sum(diff,2));
            if (mn<=10^-16)
                fval = info.xf(ind(1),dim+1); info.fworst = fval;
            else
               info.fworst = f; 
               info.xf(info.nf,1:dim) = info.yworst'; 
               info.xf(info.nf,dim+1) = info.fworst;
               info=UpdatePoints(info);
            end
            
           Vstate=79;
        end
   end
   if Vstate==81
     if MA.int.alpha>0 ||MA.cont.alpha>0
         if info.ftrial<fbest
             info.nmiLS =info.nmiLS+1;
            fbest = info.ftrial; xbest = info.ytrial;
         end
     end
     Vstate=2;
   end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


