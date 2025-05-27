/*
A library that implements the tensor class and basic linear algebra operations.alignas
Apply method offers an alternative to rapidly parallelizing kernels in sequential programs.


Ata Guvendi
Northwestern University
Spring 2025
CS 358
*/

#include <cstddef>
#include <cassert>
#include <omp.h>
#include <iostream>
#include <cmath>
#include <vector>

using namespace std;


class Tensor {
private:
    size_t* strides;//internal -> data is kept flat. In order to index (5,3) for example, we have to keep how large a dimension is so we can calculate a 'leaq' like pattern.
   
public:
    int T_ct; //number of omp threads this object has access to
    double* data; //kept as a flat array for parallelization benefits.
    size_t* shape; //ordered array of shape, i.e. a column vector will be [5,1]
    size_t n_dims; //the length of shape
    size_t total_size; //the length of data
    
    /// @brief Default constructor will produce a 1x1 tensor with the value 0.
    Tensor(){
        data = new double[1];
        data[0] = 0.0;
        shape = new size_t[2];
        shape[0] = 1;
        shape[1] = 1;

        strides = new size_t[2];
        strides[0] = 1;
        strides[1] = 1;
        T_ct=1;
        n_dims = 2;
        total_size = 1;
    }
    /// @brief Constructor default initializes every entry to 0.
    /// @param shape_input Array of sizes. E.g. {5,1} is a column vector 5 items long.
    /// @param n_dims Number of dimensions, should be the length of the shape array.
    /// Caution: Unlike numpy, it is disallowed to define vectors as {5}. Bad practice, no consideration made
    /// for implementation. Operations run on same n_dims objects.
    Tensor(const size_t* shape_input, size_t n_dims, int Tin) : T_ct(Tin), n_dims(n_dims), total_size(1 /*placeholder-> recalculated in body*/) {
        assert(n_dims>=2);
        shape = new size_t[n_dims];
        #pragma omp parallel for num_threads(T_ct)
        for (size_t i = 0; i < n_dims; i++){
            shape[i] = shape_input[i];
            #pragma omp atomic
            total_size *= shape[i];
        }
        strides = new size_t[n_dims];
        strides[n_dims - 1] = 1;
        #pragma omp parallel for num_threads(T_ct)
        for (int i = (int)n_dims - 2; i >= 0; i--){
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        data = new double[total_size];
        #pragma omp parallel for num_threads(T_ct)
        for (size_t i = 0; i < total_size; i++){
            data[i] = 0.0;
        }
    }

    /// @brief Constructor initializes every entry from the input data.
    /// @param data_input Initialization array. Must be in flat format.
    /// @param shape_input Array of sizes. E.g. {5,1} is a column vector 5 items long.
    /// @param n_dims Number of dimensions, should be the length of the shape array.
    /// Caution: Unlike numpy, it is disallowed to define vectors as {5}. Bad practice, no consideration made
    /// for implementation. Operations run on same n_dims objects.
    Tensor(const double* data_input, const size_t* shape_input, size_t n_dims, int Tin) : T_ct(Tin), n_dims(n_dims), total_size(1) {
        assert(n_dims>=2);
        shape = new size_t[n_dims];
        #pragma omp parallel for num_threads(T_ct)
        for (size_t i = 0; i < n_dims; i++){
            shape[i] = shape_input[i];
            #pragma omp atomic
            total_size *= shape[i];
        }
        strides = new size_t[n_dims];
        strides[n_dims - 1] = 1;
        #pragma omp parallel for num_threads(T_ct)
        for (int i = (int)n_dims - 2; i >= 0; i--){
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        data = new double[total_size];
        #pragma omp parallel for num_threads(T_ct)
        for (size_t i = 0; i < total_size; i++){
            data[i] = data_input[i];
        }
    }

    /// @brief Copy constructor from other tensor.
    /// @param other tensor.
    Tensor(const Tensor& other) : T_ct(other.T_ct), n_dims(other.n_dims), total_size(other.total_size) {
        shape = new size_t[n_dims];
        strides = new size_t[n_dims];
        #pragma omp parallel for num_threads(T_ct)
        for (size_t i = 0; i < n_dims; i++){
            shape[i] = other.shape[i];
            strides[i] = other.strides[i];
        }
        data = new double[total_size];
        #pragma omp parallel for num_threads(T_ct)
        for (size_t i = 0; i < total_size; i++){
            data[i] = other.data[i];
        }
    }

    /// @brief Assignment operator. Very close to copy constructor - but memory safety matters here.
    /// @param other tensor
    /// @return itself.
    Tensor& operator=(const Tensor& other){
        if (this != &other) {
            delete[] data;
            delete[] shape;
            delete[] strides; //overwriting - need to delete.
            n_dims = other.n_dims;
            total_size = other.total_size;
            T_ct = other.T_ct;
            shape = new size_t[n_dims];
            strides = new size_t[n_dims];
            #pragma omp parallel for num_threads(T_ct)
            for (size_t i = 0; i < n_dims; i++){
                shape[i] = other.shape[i];
                strides[i] = other.strides[i];
            }
            data = new double[total_size];
            for (size_t i = 0; i < total_size; i++){
                data[i] = other.data[i];
            }
        }
        return *this;
    }

    // Destructor
    ~Tensor() {
        delete[] data;
        delete[] shape;
        delete[] strides;
    }

    /// @brief Fills the tensor with the value specified 
    /// @param value value to fill with. 
    void fill(double value){
        #pragma omp parallel for num_threads(T_ct)
        for (size_t i = 0; i < total_size; i++){
            data[i] = value;
        }
    }

    /// @brief Elementwise addition (add every element). Shapes must be compatible (identical)
    /// @param other tensor
    /// @return resultant tensor (new)
    Tensor operator+(const Tensor& other) const{
        assert(n_dims == other.n_dims);
        #pragma omp parallel for num_threads(T_ct)
        for (size_t i = 0; i < n_dims; i++){
            assert(shape[i] == other.shape[i]);
        }
        Tensor result(shape, n_dims, other.T_ct);
        #pragma omp parallel for num_threads(T_ct)
        for (size_t i = 0; i < total_size; i++){
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    /// @brief Elementwise addition (every element) with broadcasting
    /// @param factor value to broadcast
    /// @return resultant tensor (new)
    Tensor operator+(double factor) const{
        Tensor result(shape, n_dims, T_ct);
        #pragma omp parallel for num_threads(T_ct)
        for (size_t i = 0; i < total_size; i++){
            result.data[i] = data[i] + factor;
        }
        return result;
    }

    /// @brief Elementwise subtraction (subtract every element). Shapes must be compatible (identical)
    /// @param other tensor
    /// @return resultant tensor (new)
    Tensor operator-(const Tensor& other) const{
        assert(n_dims == other.n_dims);
        #pragma omp parallel for num_threads(T_ct)
        for (size_t i = 0; i < n_dims; i++){
            assert(shape[i] == other.shape[i]);
        }
        Tensor result(shape, n_dims, other.T_ct);
        #pragma omp parallel for num_threads(T_ct)
        for (size_t i = 0; i < total_size; i++){
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }

    /// @brief Elementwise subtraction (every element) with broadcasting
    /// @param factor value to broadcast
    /// @return resultant tensor (new)
    Tensor operator-(double factor) const{
        Tensor result(shape, n_dims, T_ct);
        #pragma omp parallel for num_threads(T_ct)
        for (size_t i = 0; i < total_size; i++){
            result.data[i] = data[i] - factor;
        }
        return result;
    }

    /// @brief Elementwise multiplication (mult every element). Shapes must be compatible (identical)
    /// @param other tensor
    /// @return resultant tensor (new)
    Tensor operator*(const Tensor& other) const{
        assert(n_dims == other.n_dims);
        #pragma omp parallel for num_threads(T_ct)
        for (size_t i = 0; i < n_dims; i++){
            assert(shape[i] == other.shape[i]);
        }
        Tensor result(shape, n_dims, T_ct);
        #pragma omp parallel for num_threads(T_ct)
        for (size_t i = 0; i < total_size; i++){
            result.data[i] = data[i] * other.data[i];
        }
        return result;
    }

    /// @brief Elementwise mult (every element) with broadcasting
    /// @param factor value to broadcast
    /// @return resultant tensor (new)
    Tensor operator*(double factor) const{
        Tensor result(shape, n_dims, T_ct);
        #pragma omp parallel for num_threads(T_ct)
        for (size_t i = 0; i < total_size; i++){
            result.data[i] = data[i] * factor;
        }
        return result;
    }

    /// @brief Elementwise division (mult every element). Shapes must be compatible (identical)
    /// @param other tensor
    /// @return resultant tensor (new)
    Tensor operator/(const Tensor& other) const{
        assert(n_dims == other.n_dims);
        #pragma omp parallel for num_threads(T_ct)
        for (size_t i = 0; i < n_dims; i++){
            assert(shape[i] == other.shape[i]);
        }
        Tensor result(shape, n_dims, T_ct);
        #pragma omp parallel for num_threads(T_ct)
        for (size_t i = 0; i < total_size; i++){
            result.data[i] = data[i] / other.data[i];
        }
        return result;
    }

    /// @brief Elementwise division (every element) with broadcasting
    /// @param factor value to broadcast
    /// @return resultant tensor (new)
    Tensor operator/(double factor) const{
        Tensor result(shape, n_dims, T_ct);
        #pragma omp parallel for num_threads(T_ct)
        for (size_t i = 0; i < total_size; i++){
            result.data[i] = data[i] / factor;
        }
        return result;
    }

    /// @brief Accessor. Reads double at index. Can only do one item at a time.
    /// @param indices must be an array of n_dims length specifying which item to look up.
    /// @return double at the index.
    double operator()(const size_t* indices) const{
        size_t flat_index = 0;
        #pragma omp parallel for num_threads(T_ct)
        for (size_t i = 0; i < n_dims; i++){
            assert(indices[i] < shape[i]); // bounds check
            flat_index += indices[i] * strides[i];
        }
        return data[flat_index];
    }

    /// @brief Setter. Sets double at index. Can only do one item at a time.
    /// @param indices must be an array of n_dims length specifying which item to look up.
    /// @param value double at the index.
    void set(const size_t* indices, double value) {
        size_t flat_index = 0;
        #pragma omp parallel for num_threads(T_ct)
        for (size_t i = 0; i < n_dims; i++){
            assert(indices[i] < shape[i]); // bounds check
            flat_index += indices[i] * strides[i];
        }
        data[flat_index] = value;
    }

    /// @brief Standard matrix multiplication. WARNING: No support for matrix-vector. Define your vector as a matrix with 1 dimension 1.
    /// @param A Matrix 1
    /// @param B Matrix 2
    /// @return Resultant matrix
    static Tensor matmul(const Tensor& A, const Tensor& B){
        assert(A.n_dims == 2 and B.n_dims == 2);
        assert(A.shape[1] == B.shape[0]);
        size_t indexer[2] = {A.shape[0], B.shape[1]};
        int maxT = A.T_ct>B.T_ct? A.T_ct : B.T_ct;
        Tensor result(indexer, 2, maxT);
        #pragma omp parallel for num_threads(maxT)
        for (size_t i = 0; i<A.shape[0]; i++){
            for (size_t k = 0; k<A.shape[1]; k++){
                for (size_t j=0; j<B.shape[1]; j++){
                    result.data[i*B.shape[1]+j] += A.data[i*A.shape[1]+k] * B.data[k*B.shape[1]+j];
                }

            }
        }
        return result;
    }

    /// @brief Applies the given function in place to the tensor.
    /// @param func double->double to be applied to each element.
    void apply(double (*func)(double)){
        #pragma omp parallel for num_threads(T_ct) schedule(dynamic) //no guarantees about f being a predictible workload.
        for (size_t i = 0; i < total_size; i++){
            data[i] = func(data[i]);
        }
    }

    /// @brief Applies a function to a tensor across all entities along dimension 0. To parallelize tasks using this, make sure
    /// to make full use of layer(), as it can build the correct structure required.
    /// @param func  void->Tensor, modifying the tensor in place. Note that the tensor argument must
    /// be the same dimension as this object, and you must use reshape yourself if you want to focus on that sheet exclusively.
    /// Thus you can count on each slice being of dimensions {1,b,c,d,e...} if this tensor is {a,b,c,d,e,...}
    void apply(void (*func)(Tensor&)){
        size_t num_layers = shape[0];
        #pragma omp parallel for num_threads(T_ct) schedule(dynamic) //no guarantees about f being a predictible workload.
        for (size_t i = 0; i < num_layers; i++){
            Tensor slice = this->isolate(0,i);
            slice.T_ct = 1; //no overlaunching threads!
            func(slice);
            size_t sheet_start_idx = i*strides[0];
            for (size_t j = 0; j < slice.total_size; j++)
                data[j+sheet_start_idx] = slice.data[j%strides[0]];
        }
    }

    /// @brief Layers the input sheets by creating a +1 dimensional Tensor and placing each sheet along dimension 0.
    /// @param sheets array of sheets to layer
    /// @param num_sheets length of sheets
    /// @return Layered tensor
    static Tensor layer(const Tensor* sheets, size_t num_sheets){
        assert (num_sheets > 0);

        size_t n_dims_per_sheet = sheets[0].n_dims;
        //normally one should check dimensional compatibility here, but this is to construct a HPC
        // application. If an error occurs because the dimensions in sheets is not uniform, thats on you.
        size_t total_dims_required = n_dims_per_sheet+1;
        size_t shape[total_dims_required];
        shape[0] = num_sheets; //always layer among first row -> increases locality within sheets.
        #pragma omp parallel for
        for (size_t d = 1; d<total_dims_required; d++){
            shape[d] = sheets[0].shape[d-1];
        }
        //This can be a massively parallel operation that we are building, no need to find the max among all sheets.
        //You the user, need to make sure you manage your threads well.
        Tensor layered(shape, total_dims_required, sheets[0].T_ct);

        //!Idea: If you imagine every sheet as a 1x1, the below makes so much more sense
        //!as you are now copying data for a matrix, not a tensor.
        #pragma omp parallel for collapse(2)
        for(size_t i = 0; i<num_sheets; i++){
            for (size_t j = 0; j<sheets[0].total_size; j++){
                layered.data[i*sheets[0].total_size+j] = sheets[i].data[j];
            }
        }

        return layered;
    }

    /// @brief Inverse operation of layer, with the caveat of no repairs done to collapse the layering dimension.
    /// @param layered tensor to unlayer
    /// @return An array of unlayered tensors. Note: Caller takes ownership of heap array.
    static Tensor* unlayer(Tensor& layered){
        assert (layered.n_dims >= 3);
        size_t num_sheets = layered.shape[0];
        Tensor* layers = new Tensor[num_sheets];
        //isolate is one of the least parallel friendly functions, so why not use
        //the parallelism on the outside?
        int T_ct = layered.T_ct;
        layered.T_ct = 1;
        #pragma omp parallel for
        for (size_t i=0; i<num_sheets; i++){
            layers[i] = layered.isolate(0,i);
        }
        layered.T_ct = T_ct;

        return layers;
    }

    /// @brief Attempts to cast the tensor into new specified dimensions. New dimensions must perfectly divide the data.
    /// @param new_shape array which must divide the tensor into the desired shape.
    /// @param new_n_dims length of the new_shape array.
    void reshape(const size_t* new_shape, size_t new_n_dims){
        //Checking if division of the array is possible
        size_t new_total_size = 1;
        #pragma omp parallel for num_threads(T_ct)
        for (size_t i = 0; i < new_n_dims; i++){
            #pragma omp atomic
            new_total_size *= new_shape[i];
        }
        assert(new_total_size == total_size);
        delete[] shape;
        delete[] strides;
        //allocate and copy
        shape = new size_t[new_n_dims];
        #pragma omp parallel for num_threads(T_ct)
        for (size_t i = 0; i < new_n_dims; i++){
            shape[i] = new_shape[i];
        }
        strides = new size_t[new_n_dims];
        strides[new_n_dims - 1] = 1;
        #pragma omp parallel for num_threads(T_ct)
        for (int i = (int)new_n_dims - 2; i >= 0; i--){
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        n_dims = new_n_dims;
    }
    
    /// @brief Transposes the matrix
    /// @return  Returns a new trasposed matrix. Original is unaffected
    Tensor T() const{
        assert (n_dims == 2); //above 2 dimensions, transposition is ambiguous as any two dimensions can be reordered.
        size_t transpose_shape[2] = {shape[1], shape[0]};
        Tensor transposed(transpose_shape, 2, T_ct);

        #pragma omp parallel for num_threads(T_ct)
        for (size_t i = 0; i<shape[0]; i++){
            for (size_t j = 0; j<shape[1]; j++){
                transposed.data[j*transposed.strides[0] + i] = data[i*strides[0] + j]; 
            }
        }
        return transposed;

    }

    /// @brief Vertically stacks the two tensors of dimension 2. Col count must be the same
    /// @param A tensor 1 (axc)
    /// @param B tensor 2 (bxc)
    /// @return Vertically concatenated tensor ((a+b)xc)
    static Tensor vBlock(const Tensor& A, const Tensor& B){
        assert(A.n_dims == 2);
        assert(B.n_dims == 2);
        assert(A.shape[1] == B.shape[1]);

        size_t new_shape[2] = {A.shape[0]+B.shape[0], A.shape[1]};
        int maxT = A.T_ct>B.T_ct? A.T_ct : B.T_ct;
        Tensor result(new_shape, 2, maxT);

        #pragma omp parallel for num_threads(maxT)
        for (size_t i = 0; i<A.shape[0]; i++){
            for (size_t j = 0; j<A.shape[1]; j++){
                result.data[i*result.strides[0] + j] = A.data[i*A.strides[0] + j]; 
            }
        }

        #pragma omp parallel for num_threads(maxT)
        for (size_t i = 0; i<B.shape[0]; i++){
            for (size_t j = 0; j<B.shape[1]; j++){
                result.data[(i+A.shape[0])*result.strides[0] + j] = B.data[i*B.strides[0] + j]; 
            }
        }

        return result;
    }

    /// @brief Horizontally stacks the two tensors of dimension 2. Row count must be the same
    /// @param A tensor 1 (axb)
    /// @param B tensor 2 (axc)
    /// @return Vertically concatenated tensor (ax(b+c))
    static Tensor hBlock(const Tensor& A, const Tensor& B){
        assert(A.n_dims == 2);
        assert(B.n_dims == 2);
        assert(A.shape[0] == B.shape[0]);

        size_t new_shape[2] = {A.shape[0], A.shape[1]+B.shape[1]};
        int maxT = A.T_ct>B.T_ct? A.T_ct : B.T_ct;
        Tensor result(new_shape, 2, maxT);

        #pragma omp parallel for num_threads(maxT)
        for (size_t i = 0; i<A.shape[0]; i++){
            for (size_t j = 0; j<A.shape[1]; j++){
                result.data[i*result.strides[0] + j] = A.data[i*A.strides[0] + j]; 
            }
        }

        #pragma omp parallel for num_threads(maxT)
        for (size_t i = 0; i<B.shape[0]; i++){
            for (size_t j = 0; j<B.shape[1]; j++){
                result.data[i*result.strides[0] + (j+A.shape[1])] = B.data[i*B.strides[0] + j]; 
            }
        }

        return result;
    }

    /// @brief Isolates the tensor along the dimension specified at index. Tensor will NOT go down in ndims, but will have 1 in the shape entry
    /// at the relevant place.
    /// @param dim The dimension number to isolate
    /// @param index The index to focus on in the given dimension
    /// @return Tensor of the same dimension but with the specified dimension isolated.
    Tensor isolate(size_t dim, size_t index) const{
        assert(dim<n_dims);
        assert(index<shape[dim]);

        size_t new_shape[n_dims];
        #pragma omp parallel for num_threads(T_ct)
        for (size_t i = 0; i<n_dims; i++)
            new_shape[i] = shape[i]*(dim != i) + (dim==i);
        Tensor isolated(new_shape, n_dims, T_ct);

        int isolated_idx = 0;
        for (size_t i=0; i<total_size; i++){
            if (((i / strides[dim]) % shape[dim]) == index){
                isolated.data[isolated_idx] = data[i];
                isolated_idx++; //TODO: omp races, got to find a better way to index isolated_idx without setting a variable.
            }
        }
        return isolated;
    }

    /// @brief Reduced row eachelon form of the Tensor. Must be 2 dimensional.
    /// @param tol Values under tol are accepted to be 0.
    /// @return RREF'd tensor.
    Tensor rref(double tol) const{
        assert (n_dims == 2);

        vector<Tensor> rows;
        size_t min_dim = (shape[0]<shape[1])?shape[0]:shape[1];
        for (size_t i = 0; i<shape[0]; i++){
            rows.emplace_back(this->isolate(0,i)); 
        }

        for (size_t pivot_num = 0; pivot_num < min_dim; pivot_num++){ 
            //search for a non-zero row at that column.
            size_t non_zero_row = pivot_num;
            for (size_t row=pivot_num+1; row<shape[0]; row++){
                if (fabs(rows[row].data[pivot_num]) > fabs(rows[non_zero_row].data[pivot_num])){
                    non_zero_row = row;
                }
            }
            //dont row reduce if nothing is above tolerance.
            if (fabs(rows[non_zero_row].data[pivot_num]) < tol) continue;

            double pivot = rows[non_zero_row].data[pivot_num];
            Tensor temp = rows[non_zero_row];
            rows[non_zero_row] = rows[pivot_num];
            rows[pivot_num] = temp/pivot;

            
            for (size_t row = 0; row<shape[0]; row++){
                double factor = rows[row].data[pivot_num];
                if (row != pivot_num)
                    rows[row] = rows[row] - (rows[pivot_num] * factor);
            }

        }

        Tensor result = rows[0];
        for (size_t i = 1; i<shape[0]; i++){
            result = Tensor::vBlock(result, rows[i]);
        }

        #pragma omp parallel for num_threads(result.T_ct)
        for (size_t i=0; i<result.total_size; i++){
            if (fabs(result.data[i]< tol)) result.data[i] = 0.0;
        }

        return result;
    }

    /// @brief Creates the identity matrix
    /// @param size Will create dimension {size,size}
    /// @param T_ct Maximum number of threads the tensor object has access to
    /// @return Identity matrix
    static Tensor eye(size_t size, int T_ct){
        size_t shape[2] = {size, size};
        Tensor result(shape,2, T_ct);

        #pragma omp parallel for num_threads(T_ct)
        for (size_t i=0; i<size; i++){
            result.data[i*result.strides[0] + i] = 1.0;
        }
        return result;
    }


    /// @brief Will create a tensor filled with ones.
    /// @param shape_input Shape array of the desired size
    /// @param n_dims Length of shape input
    /// @param Tin Number of threads the Tensor has access to
    /// @return Tensor of ones
    static Tensor ones(const size_t* shape_input, size_t n_dims, int Tin){
        Tensor result(shape_input, 2, Tin);

        #pragma omp parallel for num_threads(Tin)
        for (size_t i=0; i<result.total_size; i++){
            result.data[i] = 1.0;
        }
        return result;
    }

    /// @brief Inverts the tensor (must be 2D). Uses [A I] ~ [I A^-1] implementation
    /// In future implementations, consider a more parallel algorithm,
    /// this algorithm is iterative and does not parallelize well at all.
    /// @return Inverse of the tensor.
    Tensor inv() const{
        assert (n_dims == 2);
        assert (shape[0] == shape[1]);
        Tensor augmented = hBlock(*this, eye(shape[0], T_ct));
        Tensor reduced = augmented.rref(1e-7);
        size_t inverse_shape[2] = {shape[0], shape[1]};
        Tensor inverse(inverse_shape, 2, T_ct);
        #pragma omp parallel for num_threads(T_ct)
        for (size_t i = 0; i<shape[0]; i++){
            for (size_t j = 0; j<shape[1]; j++){
                inverse.data[i*strides[0] + j] = reduced.data[i*reduced.strides[0] + j + shape[0] /*first n cols should be I*/]; 
            }
        }
        return inverse;
    }

    /// @brief Prints the tensor to console.
    void print() const{
        if (n_dims == 2){
            cout<<"Matrix of shape (" << shape[0] << ", " << shape[1] << ")"<< endl;
            for (size_t i = 0; i<shape[0]; i++){
                for (size_t j = 0; j<shape[1]-1; j++){
                    cout << data[i*strides[0] + j] << ", ";
                }
                cout << data[i*strides[0] + (shape[1]-1)] << endl;;
            }
        }
        else{
            cout << "Printing in higher orders is not implemented yet. Sorry!" << endl;
        }
    }
};
