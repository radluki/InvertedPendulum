function cost = python_model(K)

Kstr = ['"[',num2str(K,'%f, ')];
Kstr(end) = ']';
Kstr = [Kstr,'"'];
%Kstr(regexp(Kstr,'[,]'))=[]
command = ['! python3 model_m.py ',Kstr]
cd ..
eval(command)
cd matlab
% f = fopen('model_m.json');
% line = fgetl(f);
% fclose(f);
load xxx
cost = cost
