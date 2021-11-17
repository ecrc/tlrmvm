#ifndef MATRIX_H
#define MATRIX_H

#include <unistd.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <complex>
using namespace std;

namespace tlrmat{



/**
 * @brief A dense matrix class with basic feature.
 * To keep simplicity, we didn't use any optimization techniques for performance.
 * 
 * @tparam T 
 */
template<typename T>
class Matrix{
public:

    Matrix(); //!< Empty constructor, donothing.

    Matrix(T *inptr, size_t M, size_t N); //!< Accept an initialized pointer as constructor.
    
    Matrix(vector<T> invec, size_t M, size_t N); //!< Accept an initialized pointer as constructor.

    Matrix(size_t M, size_t N); //!< Random number constructor.
    
    Matrix(const Matrix&); //!< Copy constructor.

    ~Matrix(); //!< Deconstructor.

    T GetElem(size_t row, size_t col) const {return dataptr[row+col*M];}  //!< Get element at Index(i,j)
    
    void SetElem(size_t row, size_t col, T val) {dataptr[row+col*M] = val;} // set element at Index(i,j)
    
    void Fill(T val); //!< Fill the data pointer with value val.

    T Sum(); //!< Sum of matrix.

    vector<T> RowSum(); //!< Sum along the row.
    
    vector<T> ColSum(); //!< Sum along the column.



    /**
     * @brief Transpose Matrix.
     * 
     * @return Matrix 
     */
    Matrix<T> Transpose();

    size_t Row() const {return M;} //!< return Row.

    Matrix<T> Row(size_t idx);

    size_t Col() const {return N;} //!< return Column.

    Matrix<T> Col(size_t idx);

    Matrix<T> Block(vector<size_t> rowidx, vector<size_t> colidx);

    Matrix<T> Block(vector<size_t> rowidx);

    size_t Lda() const {return lda;} //!< return leading dimension.

    vector<size_t> Shape() const { return {M, N}; }

    // deprecated
    T * RawPtr() const {return (T*)dataptr.data();} //!< return raw pointer.
    
    vector<T>& Datavec() const { return (vector<T>&)dataptr; }
    /**
     * @brief Copy data of matrix from a source ptr.
     * 
     * @param srcptr 
     */
    // void CopyFromPointer(T * srcptr);

    double allclose(Matrix<T> & rhs);

    /**
     * @brief Write Matrix to files.
     * 
     * @param abspath 
     */
    void Tofile(string abspath);
    /**
     * @brief Load Matrix From files.
     * 
     * @param abspath 
     */
    static Matrix<T> Fromfile(string abspath, size_t M, size_t N);

    Matrix<T> ApplyMask(Matrix<int> maskmat);

    // operator overloading
    Matrix<T>& operator=(const Matrix<T>& rhs);
    Matrix<T> operator+(const Matrix<T>& rhs);
    Matrix<T>& operator+=(const Matrix<T>& rhs);
    Matrix<T> operator-(const Matrix<T>& rhs);
    Matrix<T>& operator-=(const Matrix<T>& rhs);
    Matrix<T> operator*(const Matrix<T>& rhs);
    Matrix<T>& operator*=(const Matrix<T>& rhs);

    template<typename U> // https://stackoverflow.com/questions/4660123/overloading-friend-operator-for-template-class
    friend ostream& operator<<(ostream& os, const Matrix<U>& mat);


protected:
    vector<T> dataptr;
    size_t M;
    size_t N;
    size_t lda;
};

template<typename T>
ostream& operator<<(ostream& os, const Matrix<T>& mat);


template <typename T>
void GetRealPart(Matrix<complex<T>> & complexmat, T ** realptr);
template <typename T>
void GetImagPart(Matrix<complex<T>> & complexmat, T ** imagptr);



} // namespace tlrmat


#endif // MATRIX_H


