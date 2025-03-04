#include <mpi.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <omp.h>
#include <immintrin.h>
using namespace std;

int main(int argc, char** argv) {
  int size, rank;
//MPI communicatorの初期化
  MPI_Init(&argc, &argv);
//MPI プロセス数の取得
  MPI_Comm_size(MPI_COMM_WORLD, &size);
//MPI プロセス番号の取得
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int N = 1024;
  vector<float> A(N*N);
  vector<float> B(N*N);
  vector<float> C(N*N, 0);
  vector<float> subA(N*N/size);
  vector<float> subB(N*N/size);
  vector<float> subC(N*N/size, 0);
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[N*i+j] = drand48();
      B[N*i+j] = drand48();
    }
  }
  int offset = N/size*rank;
  for (int i=0; i<N/size; i++)
    for (int j=0; j<N; j++)
      subA[N*i+j] = A[N*(i+offset)+j];
  for (int i=0; i<N; i++)
    for (int j=0; j<N/size; j++)
      subB[N/size*i+j] = B[N*i+j+offset];
  int recv_from = (rank + 1) % size;
  int send_to = (rank - 1 + size) % size;

  double comp_time = 0, comm_time = 0;
  for(int irank=0; irank<size; irank++) {
    auto tic = chrono::steady_clock::now();
    offset = N/size*((rank+irank) % size);
  omp_set_num_threads(16);
  #pragma omp parallel num_threads(16)
　//OpenMP for文　並列化　3重
　#pragma omp parallel for collapse(3)
    for (int i=0; i<N/size; i++)
      for (int j=0; j<N/size; j++)
        for (int k=0; k<N; k++)
          subC[N*i+j+offset] += subA[N*i+k] * subB[N/size*k+j];
    auto toc = chrono::steady_clock::now();
    comp_time += chrono::duration<double>(toc - tic).count();
    MPI_Send(&subB[0], N*N/size, MPI_FLOAT, send_to, 0, MPI_COMM_WORLD);
    MPI_Recv(&subB[0], N*N/size, MPI_FLOAT, recv_from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    tic = chrono::steady_clock::now();
    comm_time += chrono::duration<double>(tic - toc).count();
  }
  MPI_Allgather(&subC[0], N*N/size, MPI_FLOAT, &C[0], N*N/size, MPI_FLOAT, MPI_COMM_WORLD);
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      for (int k=0; k<N; k++)
      // SIMD　計算の並列化
          __m256 vA = _mm256_load_ps(&(A[N*i+k]));
          __m256 vB = _mm256_load_ps(&(B[N*i+k]));
          __m256 vC = _mm256_load_ps(&(C[N*i+j]));
          __m256 vC -= __m256_mul_ps(vA,vB);
          _mm256_store_ps(C,vC);
//        C[N*i+j] -= A[N*i+k] * B[N*k+j];
  double err = 0;
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      err += fabs(C[N*i+j]);
  if(rank==0) {
    double time = comp_time+comm_time;
    printf("N    : %d\n",N);
    printf("comp : %lf s\n", comp_time);
    printf("comm : %lf s\n", comm_time);
    printf("total: %lf s (%lf GFlops)\n",time,2.*N*N*N/time/1e9);
    printf("error: %lf\n",err/N/N);
  }
  MPI_Finalize();
}
