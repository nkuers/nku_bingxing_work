#include <iostream>
#include<cmath>
#include <windows.h>
using namespace std;

#define ull unsigned long long int

const ull n = 22;
const ull N = pow(2, n);
ull*a = new ull[N];
int LOOP = 1000;
int bias = 100;

void init()
{
    for (ull i = 0; i < N; i++)
        a[i] = i + bias;
}

void ordinary()
{
    long long int begin, end, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&begin);
    ull sum = 0;
    for (int l = 0; l < LOOP; l++)
    {
        // init();

        for (int i = 0; i < N; i++)
            sum += a[i];
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&end);
    cout << "ordinary:" << (end - begin) * 1000.0 / freq / LOOP << "ms" << endl;
    cout<<sum<<endl;
}

void optimize()
{
    long long int begin, end, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&begin);
     ull sum1 = 0, sum2 = 0,sum =0;
    for (int l = 0; l < LOOP; l++)
    {

        for (int i = 0; i < N - 1; i += 2)
            sum1 += a[i], sum2 += a[i + 1];
         sum = sum1 + sum2;
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&end);
    cout << "optimize:" << (end - begin) * 1000.0 / freq / LOOP << "ms" << endl;
    cout<<sum<<endl;
}

int main()
{
    init();
    ordinary();
    optimize();
}
