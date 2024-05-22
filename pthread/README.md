**pthread_neon_dynamic.cpp** 

普通高斯消去的动态线程实现（neon） 

**pthread_neon_static.cpp** 

普通高斯消去的静态线程+信号量实现，主线程做除法，其余线程消去（neon） 

**all_neon.cpp** 

普通高斯消去的静态线程+信号量+三重循环全部纳入线程函数实现，主线程只做创建和挂起销毁，其余线程做除法和消去（neon） 

**neon_barrier.cpp** 

普通高斯消去的静态线程+barrier实现（neon） 

**dynamic_avx.cpp**

普通高斯消去的动态线程与avx指令集相结合的实现

**static_avx.cpp**

普通高斯消去的静态线程与avx指令集相结合的实现

**barrier_avx.cpp**

普通高斯消去的静态线程+barrier与avx指令集相结合的实现

**pthread_super.cpp**

特殊高斯消去实现
