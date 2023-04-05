% transforms actual bounds {int, oint, cat, real} to encoded bounds which only accepts {int, real}
function v = boundtransform(bounds)
    dec_size = size(bounds)(2) % number of "decoded"/actual variables 
    enc_size = sum(bounds(4,:)) % number of "encoded" variables
    v = []; % new bound matrix
    for i=1:dec_size 
        % case distinction for each type
        % 0 = real, 1 = int, 2 = cat, 3 = oint
        if (bounds(3,i) == 0) || (bounds(3,i) == 1)
            v = [v bounds(:,i)]; % take the bounds as it is
        elseif (bounds(3,i) == 2) || (bounds(3,i) == 3) 
            for i=1:bounds(4,i)
                v = [v [0; 1; 0; 1]]; % set the bounds to be between 0 and 1
            end
        end
    end
