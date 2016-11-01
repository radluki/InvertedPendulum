#include <iostream>
#include <fstream>
#include <boost/array.hpp>
#include <time.h>

#include <boost/numeric/odeint.hpp>

using namespace std;
using namespace boost::numeric::odeint;

const double sigma = 10.0;
const double R = 28.0;
const double b = 8.0 / 3.0;
fstream fs;

typedef boost::array< double , 4 > state_type;

void lorenz( const state_type &x , state_type &dxdt , double t )
{
    dxdt[0] = sigma * ( x[1] - x[0] );
    dxdt[1] = R * x[0] - x[1] - x[0] * x[2];
    dxdt[2] = -b * x[2] + x[0] * x[1];
    dxdt[3] = -b * x[1] + x[3] * x[1];
}

void write_lorenz( const state_type &x , const double t )
{
    //fs << t << '\t' << x[0] << '\t' << x[1] << '\t' << x[2] << '\t' << x[3] << endl;
}

int main(int argc, char **argv)
{
    clock_t t;
    fs.open("test.txt",fstream::out);
    state_type x = { 10.0 , 1.0 , 1.0 , 9.0 }; // initial conditions
    t = clock();
    integrate( lorenz , x , 0.0 , 1.0 , 0.1 , write_lorenz );
    t = clock() - t;
    cout<<"Czas: "<<((float)t)/CLOCKS_PER_SEC<<endl;
    fs.close();
}
