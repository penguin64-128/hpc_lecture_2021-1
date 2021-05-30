include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <omp.h>
#include <mpi.h>
#include <immintrin.h>
using namespace std;

//20M58851　finalreport
/*
実行コマンド
qrsh -g tga-hpc-lecture -l s_gpu=1 -l h_rt=1:10:00
module load intel-mpi gcc intel

mpicxx finalreport_20M58851.cpp -O3 -fopenmp
mpirun -np 4 ./a.out

結果
N    : 1024
comp : 0.086229 s
comm : 0.005121 s
total: 0.091350 s (23.508393 GFlops)
error: 0.000089

（参考）
変更後のサンプルプログラムの性能(example.cpp)
N    : 1024
comp : 1.232392 s
comm : 0.003374 s
total: 1.235765 s (1.737776 GFlops)
error: 0.000129

試行錯誤の詳細は下記参照
finalreport_20M58851_result.txt
*/

//12_profiler 00matmulを参考に改編
void matmult(vector<float> &subA, vector<float> &subB, vector<float> &subC, int N, int size,int offset) {
  const int m = N, n = N, k = N;
  const int kc = N/8; //N÷8
  const int nc = N/64; //N÷64
  const int mc = N/16; //N÷16
  const int nr = N/64; //N÷64
  const int mr = N/128;  //N÷128
//OpenMP導入
#pragma omp parallel for collapse(2)
  for (int jc=0; jc<n/size; jc+=nc) {
    for (int pc=0; pc<k; pc+=kc) {
      float Bc[kc*nc];
      for (int p=0; p<kc; p++) {
        for (int j=0; j<nc; j++) {
            // Pack into Bc
            Bc[p*nc+j] = subB[N/size*(p+pc)+j+jc];
        }
      }
      for (int ic=0; ic<m/size; ic+=mc) {
        float Ac[mc*kc],Cc[mc*nc];
        for (int i=0; i<mc; i++) {
          for (int p=0; p<kc; p++) {
            // Pack into Ac
            Ac[i*kc+p] = subA[N*(i+ic)+p+pc];
          }
          for (int j=0; j<nc; j++) {
            // Initialize Cc
            Cc[i*nc+j] = 0;
          }
        }
        for (int jr=0; jr<nc; jr+=nr) {
          for (int ir=0; ir<mc; ir+=mr) {
            for (int kr=0; kr<kc; kr++) {
              for (int i=ir; i<ir+mr; i++) {
                __m256 Avec = _mm256_broadcast_ss(Ac+i*kc+kr);
                for (int j=jr; j<jr+nr; j+=8) {
                  __m256 Bvec = _mm256_load_ps(Bc+kr*nc+j);
                  __m256 Cvec = _mm256_load_ps(Cc+i*nc+j);
                  Cvec = _mm256_fmadd_ps(Avec, Bvec, Cvec);
                  _mm256_store_ps(Cc+i*nc+j, Cvec);
                }
              }
            }
          }
        }
        for (int i=0; i<mc; i++) {
          for (int j=0; j<nc; j++) {
            // Unpack form Cc
            subC[N*(i+ic)+j+jc+offset] += Cc[i*nc+j];
          }
        }
      }
    }
  }
}


int main(int argc, char** argv) {
  int size, rank;
//MPI communicatorの初期化
  MPI_Init(&argc, &argv);
//MPI プロセス数の取得
  MPI_Comm_size(MPI_COMM_WORLD, &size);
//MPI プロセス番号の取得i//
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int N = 1024;
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
      C[N*i+j] = drand48();
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

    matmult(subA,subB,subC,N,size,offset);

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
#pragma omp parallel for
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      for (int k=0; k<N; k++)
        C[N*i+j] -= A[N*i+k] * B[N*k+j];
  double err = 0;
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      err += fabs(C[N*i+j]);
  if(rank==0){
    double time = comp_time+comm_time;
    printf("N    : %d\n",N);
    printf("comp : %lf s\n", comp_time);
    printf("comm : %lf s\n", comm_time);
    printf("total: %lf s (%lf GFlops)\n",time,2.*N*N*N/time/1e9);
    printf("error: %lf\n",err/N/N);
  }
  MPI_Finalize();
}