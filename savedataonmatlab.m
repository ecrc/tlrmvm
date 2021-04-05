// an example to save data on matlab
fileID = fopen('dataA.bin','w');
A = rand(20,20,'single')
fwrite(fileID,A,'single');
fclose(fileID)