/*
A timing and basic testing suite for the Tensor class.

!VERY IMPORTANT NOTE: Matrix inversion as is can be numerically unstable and may raise floating point
!exceptions. This is a known issue and if so, you will need to run again.

Ata Guvendi
Northwestern University
Spring 2025
CS 358
*/


#include <iostream>
#include "linalg.h"
#include <cstdlib>
#include <chrono>
#include <thread>
#include <vector>

using namespace std;

//Some functions for apply:
double randomunif(double x){ //used to generate random matrices.
    return (double)rand()/RAND_MAX;
}
void simulate_work(Tensor& A){
    this_thread::sleep_for(chrono::seconds(1));
}

Tensor* MM_layer_other = nullptr;
size_t N = 500;
//Kernel function that runs matrix mult with a fixed other tensor.
void apply_fixed_matmul(Tensor& A) {
    size_t shape[2] = {N,N};
    size_t shape2[3] = {1,N,N};
    A.reshape(shape,2);
    for(int i =0; i<50; i++)
        A = Tensor::matmul(A, *MM_layer_other);
    A.reshape(shape2,3);
}

//Runs matrix mult with sheet apply vectorization
void run_parallel_apply_test_with_MM(){
    size_t shape[2] = {N,N};
    Tensor us(shape,2,1);
    Tensor other(shape,2,1);
    us.apply(randomunif);
    other.apply(randomunif);
    MM_layer_other = &other;

    Tensor tensors[8] = {us,us,us,us,us,us,us,us};
    //One multiplication:
    auto start = chrono::high_resolution_clock::now();
    apply_fixed_matmul(us);
    auto end = chrono::high_resolution_clock::now();

    double duration_ms = chrono::duration<double, milli>(end - start).count();
    cout << "Time to multiply " << N << "x" << N << " matrix: " << duration_ms << " ms\n";

    us.T_ct = 8;
    auto start_par = chrono::high_resolution_clock::now();
    apply_fixed_matmul(us);
    auto end_par = chrono::high_resolution_clock::now();

    double par_duration_ms = chrono::duration<double, milli>(end_par - start_par).count();
    cout << "Time to multiply " << N << "x" << N << " matrix (equipped with 8 threads): " << par_duration_ms << " ms\n";
    us.T_ct = 1;

    //layer, multiply in parallel, and then unlayer:
    start = chrono::high_resolution_clock::now();
    Tensor layered = Tensor::layer(tensors, 8);
    end = chrono::high_resolution_clock::now();
    double layer_duration = chrono::duration<double, milli>(end - start).count();;

    layered.T_ct = 8;
    start = chrono::high_resolution_clock::now();
    layered.apply(apply_fixed_matmul);
    end = chrono::high_resolution_clock::now();
    double mm_duration = chrono::duration<double, milli>(end - start).count();

    start = chrono::high_resolution_clock::now();
    Tensor* unlayered = Tensor::unlayer(layered);
    end = chrono::high_resolution_clock::now();
    double unlayer_duration = chrono::duration<double, milli>(end - start).count();

    cout << "Time to layer 8 " <<  N << "x" << N << " matrices: " << layer_duration << " ms\n";
    cout << "Time to apply 8 matrix mults on " << N << "x" << N << " matrices: " << mm_duration << " ms\n";
    cout << "Time to unlayer 8 " <<  N << "x" << N << " matrices: " << unlayer_duration << " ms\n";
    delete[] unlayered;
}

// Tests perfectly parallel work.
// "Work" is sleeping for a second.
void run_parallel_apply_test_perfect_independence(){
    size_t shape[2] = {N,N};
    Tensor us(shape,2,1);;
    us.apply(randomunif);
    Tensor tensors[8] = {us,us,us,us,us,us,us,us};

    //One instance:
    auto start = chrono::high_resolution_clock::now();
    simulate_work(us);
    auto end = chrono::high_resolution_clock::now();

    double duration_ms = chrono::duration<double, milli>(end - start).count();
    cout << "Simulated parallel work on " << N << "x" << N << " matrix: " << duration_ms << " ms\n";
    //layer, apply, and then unlayer:
    start = chrono::high_resolution_clock::now();
    Tensor layered = Tensor::layer(tensors, 8);
    end = chrono::high_resolution_clock::now();
    double layer_duration = chrono::duration<double, milli>(end - start).count();;

    layered.T_ct = 8;
    start = chrono::high_resolution_clock::now();
    layered.apply(simulate_work);
    end = chrono::high_resolution_clock::now();
    double mm_duration = chrono::duration<double, milli>(end - start).count();

    start = chrono::high_resolution_clock::now();
    Tensor* unlayered = Tensor::unlayer(layered);
    end = chrono::high_resolution_clock::now();
    double unlayer_duration = chrono::duration<double, milli>(end - start).count();

    cout << "Time to layer 8 " <<  N << "x" << N << " matrices: " << layer_duration << " ms\n";
    cout << "Time to apply 8 simulated parallel work on " << N << "x" << N << " matrices: " << mm_duration << " ms\n";
    cout << "Time to unlayer 8 " <<  N << "x" << N << " matrices: " << unlayer_duration << " ms\n";
    delete[] unlayered;
}

// Tests linear regression runtimes.
void linear_regression(size_t num_data, size_t num_features, int numT){
    
    size_t colvecshape[2] = {num_data,1};
    Tensor X = Tensor::ones(colvecshape, 2, numT);
    Tensor y = X;
    //i*X_col + noise
    for(size_t i=1; i<num_data; i++){
        Tensor feature(colvecshape,2,numT);
        feature.apply(randomunif);
        feature = feature * 10.0;
        X = Tensor::hBlock(X, feature);
        y = y+(feature*i);
    }
    auto start = chrono::high_resolution_clock::now();
    Tensor XtX = Tensor::matmul(X.T(),X);
    Tensor Xty = Tensor::matmul(X.T(),y);
    Tensor XtXinv = XtX.inv();

    Tensor beta = Tensor::matmul(XtXinv, Xty);
    Tensor yhat = Tensor::matmul(X,beta);

    //You can verify that the residuals are indeed very small here.

    //cout<<"Residuals:" << endl;
    //(y-yhat).print();
    
    auto end = chrono::high_resolution_clock::now();
    double duration_ms = chrono::duration<double, milli>(end - start).count();
    cout << "Time to regress " <<  num_data << "x" << num_features << " design matrix with " <<numT << " threads = "<< duration_ms << " ms\n";
}

//Function passed into apply that performs the linear regression across sheets
void linear_regression_fn(Tensor& X){
    size_t num_data = X.shape[1];
    size_t num_features = X.shape[2];
    size_t Xshape[2] = {num_data,num_features};
    size_t shape2[3] = {1,num_data,num_features};
    X.reshape(Xshape, 2);
    Tensor y = X.isolate(1,num_features-1);
    //Didn't y leak into X here?
    //yes it did.
    //Does it matter?
    //No. Linear algebra ops will still run at the same runtime as dimensions are consistent.
    Tensor XtX = Tensor::matmul(X.T(),X);
    Tensor Xty = Tensor::matmul(X.T(),y);
    Tensor XtXinv = XtX.inv();
    Tensor beta = Tensor::matmul(XtXinv, Xty);
    Tensor yhat = Tensor::matmul(X,beta);
    X.reshape(shape2, 3);
}

//Function tests runtimes of linear regression using sheet vectorization.
void linear_regression_layered(size_t num_data, size_t num_features, int numT){
    
    size_t colvecshape[2] = {num_data,1};
    Tensor XYs[numT];
    for (int i = 0; i<numT; i++){
        Tensor X = Tensor::ones(colvecshape, 2, numT);
        Tensor y = X;
        //i*X_col + noise
        for(size_t i=1; i<num_data; i++){
            Tensor feature(colvecshape,2,numT);
            feature.apply(randomunif);
            feature = feature * 10.0;
            if (i!=((size_t)numT-1))
                X = Tensor::hBlock(X, feature);
            y = y+(feature*i);
        }
        XYs[i] = Tensor::hBlock(X,y);
    }
    //Standard layer, apply, unlayer pattern:

    auto start_layer = chrono::high_resolution_clock::now();
    Tensor layered = Tensor::layer(XYs, numT);
    auto end_layer = chrono::high_resolution_clock::now();
    double duration_ms = chrono::duration<double, milli>(end_layer - start_layer).count();
    cout << "Time to layer regression in 8 sheets " <<  num_data << "x" << num_features << " design matrix = "<< duration_ms << " ms\n";

    layered.T_ct=8;
    auto start_apply = chrono::high_resolution_clock::now();
    layered.apply(linear_regression_fn);
    auto end_apply = chrono::high_resolution_clock::now();
    double duration_ms_apply = chrono::duration<double, milli>(end_apply - start_apply).count();
    cout << "Time to apply regression in 8 sheets " <<  num_data << "x" << num_features << " design matrix = "<< duration_ms_apply << " ms\n";

    auto start = chrono::high_resolution_clock::now();
    Tensor* unlayered = Tensor::unlayer(layered);
    auto end = chrono::high_resolution_clock::now();
    duration_ms = chrono::duration<double, milli>(end - start).count();


    cout << "Time to unlayer regression in 8 sheets " << num_data << "x" << num_features << " design matrix = "<< duration_ms << " ms\n";
    delete[] unlayered;


}

//Runs the various benchmarks.
int main() {
    run_parallel_apply_test_with_MM();
    run_parallel_apply_test_perfect_independence();
    linear_regression(400, 8, 1);
    linear_regression(400, 8, 2);
    linear_regression(400, 8, 4);
    linear_regression(400, 8, 8);
    linear_regression_layered(400,8,8);

    return 0;
}