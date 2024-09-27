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
//MPI プロセス番号の取得i//
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int N = 256;
  vector<float> A(N*N);
  vector<float> B(N*N);
  vector<float> C(N*N, 0);
  vector<float> subA(N*N/size);
  vector<float> subB(N*N/size);
  vector<float> subC(N*N/size, 0);
  vector<float> recv(N*N/size);  
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
//OpenMP for文　並列化　
#pragma omp parallel for
    for (int i=0; i<N/size; i++)
//ループ入れ替え
      for (int k=0; k<N/size; k++)
        for (int j=0; j<N; j++){
          //SIMD動かない
          //__m256 Avec = _mm256_load_ps(subA+N*i+k);
          //__m256 Bvec = _mm256_load_ps(subB+N/size*k+j);
          //__m256 Cvec = _mm256_load_ps(subC+N*i+j+offset);
          //Cvec = _mm256_fmadd_ps(Avec,Bvec,Cvec);
          //_mm256_store_ps(subC+N*i+j+offset,Cvec);
          
          subC[N*i+k+offset] += subA[N*i+j] * subB[N/size*j+k];

          //ループを開いて高速化
          //subC[N*i+j+offset] += subA[N*i+k] * subB[N/size*k+j];
          //subC[N*i+j1+offset] += subA[N*i+k] * subB[N/size*k+j1];
          //subC[N*i+j2+offset] += subA[N*i+k] * subB[N/size*k+j2];
          //subC[N*i+j3+offset] += subA[N*i+k] * subB[N/size*k+j3];

          //subC[N*i1+j+offset] += subA[N*i1+k] * subB[N/size*k+j];
          //subC[N*i1+j1+offset] += subA[N*i1+k] * subB[N/size*k+j1];
          //subC[N*i1+j2+offset] += subA[N*i1+k] * subB[N/size*k+j2];
          //subC[N*i1+j3+offset] += subA[N*i1+k] * subB[N/size*k+j3];

          //subC[N*i2+j+offset] += subA[N*i2+k] * subB[N/size*k+j];
          //subC[N*i2+j1+offset] += subA[N*i2+k] * subB[N/size*k+j1];
          //subC[N*i2+j2+offset] += subA[N*i2+k] * subB[N/size*k+j2];
          //subC[N*i2+j3+offset] += subA[N*i2+k] * subB[N/size*k+j3];

          //subC[N*i3+j+offset] += subA[N*i3+k] * subB[N/size*k+j];
          //subC[N*i3+j1+offset] += subA[N*i3+k] * subB[N/size*k+j1];
          //subC[N*i3+j2+offset] += subA[N*i3+k] * subB[N/size*k+j2];
          //subC[N*i3+j3+offset] += subA[N*i3+k] * subB[N/size*k+j3];
        }
    auto toc = chrono::steady_clock::now();
    comp_time += chrono::duration<double>(toc - tic).count();
    MPI_Request request[2];
    MPI_Isend(&subB[0], N*N/size, MPI_FLOAT, send_to, 0, MPI_COMM_WORLD, &request[0]);
    MPI_Irecv(&recv[0], N*N/size, MPI_FLOAT, recv_from, 0, MPI_COMM_WORLD, &request[1]);
    MPI_Waitall(2, request, MPI_STATUS_IGNORE);
//OpenMP導入
#pragma omp for
    for (int i=0; i<N*N/size; i++)
      subB[i] = recv[i];
    tic = chrono::steady_clock::now();
    comm_time += chrono::duration<double>(tic - toc).count();
  }
  MPI_Allgather(&subC[0], N*N/size, MPI_FLOAT, &C[0], N*N/size, MPI_FLOAT, MPI_COMM_WORLD);

//答えあわせ
  for (int i=0; i<N; i++)
    //ループ順序入れ替え
    for (int k=0; k<N; k++)
      for (int j=0; j<N; j++)
        C[N*i+j] -= A[N*i+k] * B[N*k+j];
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
