#include <iostream>
#include <fstream>
#include <boost/array.hpp>
#include <time.h>
#include <math.h>
#include <string>

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/odeint/integrate/integrate_const.hpp>

using namespace std;
using namespace boost::numeric::odeint;
using namespace boost::numeric::ublas;


typedef boost::array< double , 4 > state_type;

/* The rhs of x' = f(x) defined as a class */
class InvertedPendulum 
{
    double M;
    double m;
    double b;
    double l;
    double I;
    double g;
    double ** mat;
    double ** inverse_mat;
    double *regulator;

public:
    InvertedPendulum( double M, double m, double b,
                      double l, double I,double *reg, double g = 9.81) 
                        : M(M),m(m),b(b),l(l),I(I),g(g) 
    {
        mat = new double*[2];
        inverse_mat = new double*[2];
        regulator = new double[4];
        // #TODO regulator parameters
        setRegulator(reg);
        // init matrix with 0s
        for(int i=0; i<2; i++)
        {
            mat[i] = new double[2]; // A(x)x' = f(x), mat = A(0)
            inverse_mat[i] = new double[2]; // inverse_mat = inv(A(x))
            for(int j = 0; j<2; j++)
            {
                mat[i][j] = 0;
                inverse_mat[i][j] = 0;
            }
        }
        mat[0][0] = M+m;
        mat[1][1] = I + m*l*l;
        mat[0][1] = m*l;
        mat[1][0] = m*l;
    }
    
    void setRegulator(double * tab)
    {
        for(int i=0; i<4; i++)
        {
            regulator[i] = tab[i];
        }
    
    }
    void deallocateInvertedPendulum()
    {
        for(int i=0; i<2; i++)
        {
            delete [] mat[i];
            delete [] inverse_mat[i];
        }   
        delete [] regulator;
        delete [] mat;
        delete [] inverse_mat;
    }

    //DEBUG
    void showMat()
    {
        cout<<"mat = "<<endl;
        for(int i=0; i<2; i++)
        {   
            cout<<'[';
            for(int j=0; j<2; j++)
                cout<<mat[i][j]<<", ";
            cout<<']'<<endl;
        }
        cout<<endl;
    }
    
    //DEBUG
    void checkInverseA()
    {
        cout<<"inv(A(x))*A(0) = "<<endl;
        double x = 0;
        for(int i=0; i<2; i++)
            for(int j=0; j<2; j++)
            {
                for(int k=0; k<2; k++)
                    x+=mat[i][k]*inverse_mat[k][j];
                cout<<i<<" "<<j<<" "<<x<<endl;
                x = 0;
            }
        cout<<endl;       
    }

    private:

        double detA(double theta)
        {
            return mat[0][0]*mat[1][1] - mat[1][0]*mat[0][1]*cos(theta)*cos(theta);
        }

        void inverseA(double theta)
        {
            // it is not inv(mat) but inv(A(x))
            double det = detA(theta);
            inverse_mat[0][0] = mat[1][1]/det;
            inverse_mat[1][1] = mat[0][0]/det;
            inverse_mat[1][0] = -mat[0][1]*cos(theta)/det;
            inverse_mat[0][1] = -mat[1][0]*cos(theta)/det;
        }

public:

    void operator() ( state_type &x , state_type &dxdt , const double /* t */ )
    {
        double F = 0;
        double x1 = x[1];
        double theta0 = x[2];
        double theta1 = x[3];
        double x2=0, theta2=0;

        // controll computation
        x[2] -= M_PI;
        for(int i=0; i<4; i++)
            F += regulator[i]*x[i];
        x[2] += M_PI;
        
        // sets current value of inverse_mat
        inverseA(x[2]);
        
        // A(x)*x' = f(x)
        double f[2] = { -b*x1 + m*l*theta1*theta1*sin(theta0) + F,
                        -m*g*l*sin(theta0) };
        // x' = inv(A(x))*f(x)
        for(int i=0; i<2; i++)
        {
            x2 += f[i]*inverse_mat[0][i];
            theta2 += f[i]*inverse_mat[1][i];
        }
        // derivative, output returned by argumet dxdt
        dxdt[0] = x1;
        dxdt[1] = x2;
        dxdt[2] = theta1;
        dxdt[3] = theta2;
    }
};


fstream fs;

// Observer class is the last parameter of boost's integrate function
// it has access to state and time values and it is sent to integrate by value
class SimpleObserver
{
    double* sum;

public:

    double getSum(){return *sum;}

    SimpleObserver()
    {
        this->sum = new double;    
    }
    /*
    SimpleObserver(const SimpleObserver& so)
    {
        sum = so.sum;
    }*/

    void deallocateSimpleObserver()
    {
        delete sum;
    }

    void operator()(const state_type &x , const double t )
    {
        cout << t << '\t' << x[0] << '\t' << x[1] 
           << '\t' << x[2]<< '\t' << x[3]<< endl;
        *sum += x[0];
    }


};

int main(int argc, char **argv)
{
    double reg[4];
    for(int i=0; i<4; i++)
    {
        reg[i] = strtod(argv[i+1],NULL);
    }
    //M, m, b, l, I,reg,
    double params[5];
    for(int i=0; i<5; i++)
    {
        params[i] = strtod(argv[i+5],NULL);
    }  

    double time[5];
    for(int i=0; i<3; i++)
    {
        time[i] = strtod(argv[i+10],NULL);
    }
    double n = (time[1]-time[0])/time[2];
    
    InvertedPendulum ip = InvertedPendulum(params[0],params[1],params[2],params[3],params[4],reg);
    string filename = "test3.txt";
    SimpleObserver o1 = SimpleObserver();
    runge_kutta4< state_type > stepper;
    clock_t t;
    
    fs.open(filename,fstream::out);
    //fs << "pos\tvel\ttheta\ttheta_p\n";
    state_type x0 = { 0 , 0, M_PI*0.85, 0 }; // initial conditions
    t = clock();
    integrate_const( stepper , ip , x0 , time[0] , time[1] ,n  ,o1);
    t = clock() - t;
    //cout<<"Czas: "<<((float)t)/CLOCKS_PER_SEC<<endl;
    //cout<<"Sum of x[0] from observer: "<<o1.getSum()<<endl;
    
    ip.deallocateInvertedPendulum();
    o1.deallocateSimpleObserver();
    fs.close(); 

}
