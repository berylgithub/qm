% writes x to file
function paramwriter(x, path_param)
    uid = rand(1);
    strout = num2str(uid);
    for i=1:length(x)
        strout = strcat(strout,"\t",num2str(x(i)));
    end
    file_id = fopen(path_param, 'w');
    fputs(file_id, strout);
    fclose(file_id);