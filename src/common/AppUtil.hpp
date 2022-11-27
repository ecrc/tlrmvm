//  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                      All rights reserved.

#ifndef APPUTIL_H
#define APPUTIL_H

#include <string>
#include <complex>
using namespace std;


/******************************************************************************//**
 * AppUtil.h:
 * This file contains utility functions for different application. It is not 
 * part of tlrmat function. The utility functions such as read in data from 
 * binary will appear here.
 *******************************************************************************/


/**
 * @brief Read binary file into memory buffer.
 * 
 * @tparam T 
 * @param absfilepath Binary file absolution path.
 * @param outbuffer Output memory buffer.
 * @param length Number of elements read into outbuffer.
 */
template<typename T>
void ReadBinary(string absfilepath, T * outbuffer, size_t length);
/**
 * @brief Read Binary Astronomy int file.
 * 
 * @param prefix 
 * @param outbuffer 
 * @param length 
 * @param acc 
 * @param nb 
 * @param id 
 */
void ReadAstronomyBinary(string prefix, int ** outbuffer, size_t length, 
string acc, int nb, string id);

/**
 * @brief Read Binary Astronomy float file.
 * 
 * @param prefix 
 * @param outbuffer 
 * @param length 
 * @param acc 
 * @param nb 
 * @param id 
 */
void ReadAstronomyBinary(string prefix, float ** outbuffer, size_t length, 
string acc, int nb, string id);

/**
 * @brief Read Binary Seismic int file.
 * 
 * @param prefix 
 * @param outbuffer 
 * @param length 
 * @param acc 
 * @param nb 
 * @param id 
 */
void ReadSeismicBinary(string prefix, int ** outbuffer, size_t length, 
string acc, int nb, int id);

/**
 * @brief Read Binary Seismic float file.
 * 
 * @param prefix 
 * @param outU 
 * @param outV 
 * @param length 
 * @param acc 
 * @param nb 
 * @param id 
 */
void ReadSeismicBinary(string prefix, complex<float> ** outptr, 
size_t length, string acc, int nb, int id);

void ReadSeismicBinaryX(string prefix, complex<float> ** outX, 
size_t length, string acc, int nb, int id);


// get a random x vector of length
void RandomX(float * xvector, int length);
void RandomX(double * xvector, int length);

void RandomX(complex<float> *xvector, int length);
void RandomX(complex<double> *xvector, int length);



#endif // APPUTIL_H




