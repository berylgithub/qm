function p = minprob(x)
    xh = max(x);
    z = [sort(xh .- x); Inf]; % concat with Inf for if k+1 is at the boundary
    s = cumsum(z);
    k = 0; % initialize k
    for i=1:length(z)-1
        if 1 <= i*z(i+1) - s(i);
            k=i;
            break;
        end
    end
    mu = (1 + s(k))/k;
    lambda = xh - mu;
    p = x .- lambda;
    p = max(0,p); % this vectorizes

