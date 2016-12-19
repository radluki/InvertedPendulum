
import scipy.io
from pendulum import InvertedPendulum
from regulators import RegulatorLQR



if __name__ == '__main__':
    M = 0.5
    m = 0.2
    b = 0.1
    I = 0.006
    l = 0.3

    pend = InvertedPendulum(M, m, b, l, I)
    reg = RegulatorLQR(pend.A, pend.B)

    da = scipy.io.loadmat('matlab/params_model.mat')
    x = da['x'].ravel()
    t = da['t'][0,0]
    #print(da)
    pend.regulator = reg
    dat = dict()
    dat['x_dot'] = pend.x_dot_with_regulator(x,t)
    scipy.io.savemat('matlab/output_model.mat',dat)
    # f = open('matlab/model_m.json','w')
    # json.dump(cost,f)
    # f.close()


