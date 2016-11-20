function x_dot = python_model_fun(t,x)

save('params_model.mat','t','x')
command = ['! python3 model_m_one_iter.py '];
cd ..
eval(command)
cd matlab
% f = fopen('model_m.json');
% line = fgetl(f);
% fclose(f);
load output_model
x_dot = x_dot';
