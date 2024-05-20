#include <iostream>
#include <pthread.h>
#include <semaphore.h>
#include <windows.h>
#include <pmmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#include <nmmintrin.h>
#include <immintrin.h> //AVX、AVX2
using namespace std;

const int n = 4000;
float A[n][n];
int NUM_THREADS = 7;

void init() {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = 0;
        }
        A[i][i] = 1.0;
        for (int j = i + 1; j < n; j++)
            A[i][j] = rand() % 100;
    }

    for (int i = 0; i < n; i++) {
        int k1 = rand() % n;
        int k2 = rand() % n;
        for (int j = 0; j < n; j++) {
            A[i][j] += A[0][j];
            A[k1][j] += A[k2][j];
        }
    }
}

void f_ordinary() {
    for (int k = 0; k < n; k++) {
        for (int j = k + 1; j < n; j++) {
            A[k][j] = A[k][j] * 1.0 / A[k][k];
        }
        A[k][k] = 1.0;

        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

struct threadParam_t {
    int t_id; //线程 id
};

//信号量定义
sem_t sem_leader;
sem_t* sem_Division;
sem_t* sem_Elimination;

//线程函数定义（穿插）
void* threadFunc_horizontal1(void* param) {
    __m256 va2, vt2, vx2, vaij2, vaik2, vakj2;

    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;

    for (int k = 0; k < n; ++k) {
        vt2 = _mm256_set_ps(A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k]);

        if (t_id == 0) {
            int j;
            for (j = k + 1; j + 8 <= n; j += 8) {
                va2 = _mm256_loadu_ps(&(A[k][j]));
                va2 = _mm256_div_ps(va2, vt2);
                _mm256_store_ps(&(A[k][j]), va2);
            }

            for (; j < n; j++) {
                A[k][j] = A[k][j] * 1.0 / A[k][k];
            }
            A[k][k] = 1.0;
        }
        else {
            sem_wait(&sem_Division[t_id - 1]); // 阻塞，等待完成除法操作
        }

        // t_id 为 0 的线程唤醒其它工作线程，进行消去操作
        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; ++i)
                sem_post(&sem_Division[i]);
        }

        //循环划分任务
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS) {
            //消去
            vaik2 = _mm256_set_ps(A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k]);
            int j;
            for (j = k + 1; j + 8 <= n; j += 8) {
                vakj2 = _mm256_loadu_ps(&(A[k][j]));
                vaij2 = _mm256_loadu_ps(&(A[i][j]));
                vx2 = _mm256_mul_ps(vakj2, vaik2);
                vaij2 = _mm256_sub_ps(vaij2, vx2);

                _mm256_store_ps(&A[i][j], vaij2);
            }
            for (; j < n; j++)
                A[i][j] = A[i][j] - A[i][k] * A[k][j];

            A[i][k] = 0;
        }

        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; ++i)
                sem_wait(&sem_leader); // 等待其它 worker 完成消去

            for (int i = 0; i < NUM_THREADS - 1; ++i)
                sem_post(&sem_Elimination[i]); // 通知其它 worker 进入下一轮
        }
        else {
            sem_post(&sem_leader);// 通知 leader, 已完成消去任务
            sem_wait(&sem_Elimination[t_id - 1]); // 等待通知，进入下一轮
        }
    }
    return 0;
    pthread_exit(NULL);
}

int main() {
    init();
    long long counter;// 记录次数
    double seconds;
    long long head, tail, freq, noww;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);//开始计时

    //初始化信号量
    sem_init(&sem_leader, 0, 0);

    sem_Division = new sem_t[NUM_THREADS - 1];
    sem_Elimination = new sem_t[NUM_THREADS - 1];
    for (int i = 0; i < NUM_THREADS - 1; ++i) {
        sem_init(&sem_Division[i], 0, 0);
        sem_init(&sem_Elimination[i], 0, 0);
    }

    //创建线程
    pthread_t* handles = new pthread_t[NUM_THREADS];// 创建对应的 Handle
    threadParam_t* param = new threadParam_t[NUM_THREADS];// 创建对应的线程数据结构
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFunc_horizontal1, (void*)&param[t_id]);
    }

    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        pthread_join(handles[t_id], NULL);

    //销毁所有信号量
    sem_destroy(&sem_leader);
    for (int i = 0; i < NUM_THREADS - 1; ++i) {
        sem_destroy(&sem_Division[i]);
        sem_destroy(&sem_Elimination[i]);
    }

    delete[] sem_Division;
    delete[] sem_Elimination;
    delete[] handles;
    delete[] param;

    QueryPerformanceCounter((LARGE_INTEGER*)&tail);//结束计时
    seconds = (tail - head) * 1000.0 / freq;//单位 ms

    cout << "pthread_jing: " << seconds << " ms" << endl;

    return 0;
}
