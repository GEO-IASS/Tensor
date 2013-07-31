#pragma once

#include <vector>
#include <string>
#include <stdio.h>
#include <iostream>
#include <algorithm>


#include <cv.h>
#include <highgui.h>
#include <cxcore.h>


using namespace std;
using namespace cv;

class Tensor    // Feature Map
{
protected:
    float * _data;  // data pointer
    int _size[3];   // size of tensor
    int _ownData;
public:
    Tensor();
    Tensor(bool ownData);
    Tensor(int h, int w, int c);
    Tensor(int h, int w, int c, float * data, bool copy = false);
    Tensor(int size[3], float * data, bool copy = false);
    Tensor(const Tensor & src);

    
    ~Tensor();
    bool isValid();
    bool isMatrix();
    int zero();                             // set all elements to zero
    int init(int h, int w, int c, float * data, bool copy = false);
    int init(int dims[3], float * data, bool copy = false);
    int memSet(float x);

    int reshape(Tensor src);
    int reshape(int h, int w, int c);
    int reshape(int size[3]);

    float * getData() const;            // get data pointer
    float & at(int i, int j, int k);    // get pixel of tensor, return the  of the tensor
    const int * getSize();  // get size array
    int getSize(int size[]) const;  // get size array
    int getSize(int axis);  // get size of axis

    int getHeight();    // get heigh, axis 0
    int getWidth();     // get width, axis 1
    int getChannel();   // get channel or depth, axis 2
    int getNumOfElem() const; // get total number of elements
    int getArea();      // get area size: size[0] * size[1]

    int copyData(const float * data, int n, int bias = 0);  // copy data from a pointer
    int copyTo(Tensor & dst);                   // copy tensor to a new tensor
    int copyFrom(Tensor src);                   // copy data from src

    int crop(Tensor & dst, int y, int x, int z, int h, int w, int c);  // crop a sub-tensor, start from y, x, z, size h, w, c
    int crop(Tensor & dst, int y1, int x1, int y2, int x2);  // crop a sub-tensor, start from y, x, z, size h, w, c
    //Tensor & slice(int axis, int start, int n); // get slice along axis, not used any more

    int padding(int py, int px, int pz, float val =  0);    // padding the tensor
    //int paddingPost(int py, int px, int pz, float val);  //padding post with val

    int shuffle(int axis, Tensor & target, vector<int> map);// shuffle tensor with map alone axis
    int shuffle(int axis, Tensor & target);                 // random shuffle tensor along axis
    int flip(int axis, Tensor & target);                    // flip tensor alone axis
    
    int permute(Tensor & target, int order[3]);
    int permute(int order[3]);
    int permute(int order_0, int order_1, int order_2);

    int features(Tensor & dst, int sbin);
    void resize(Tensor & dst, float scale);
    void resize(float scale);

    int FloatMatToTensor(Mat &img);
    int MatToTensor(Mat &img);
    int TensorToMat(Mat &img, int type);
    /*
     *
     * Tensor functions
     *
     */
    int sum();
    int sum(Tensor & dst); // only support sum along channels
    int sum(int axis);

    int add(Tensor B, Tensor & dst, float scaleA = 1, float scaleB = 1, float scaleC = 0);
    int add(Tensor B);
    //int add(Tensor & dst, float scaleA = 1, float scaleB = 1, float scaleC = 0);
    //int add(float scaleA = 1, float scaleB = 1, float scaleC = 0);
    int add(float scaleA, float bias);
    int add(Tensor & dst, float scaleA, float bias);
    int add(float val);
    int minus(Tensor B);
    int div(Tensor B);
    int dot(Tensor B, Tensor & target);
    int dot(Tensor B);

    //find(GT, thresh, tmpI, tmpY, tmpX, tmpS);
    int find(bool GT, float thresh, vector<int> & tmpI, vector<int> & tmpY, vector<int> & tmpX, vector<float> & tmpS);



    
    /*
     *
     * Matrix functions
     *
     */
    
    int transpose(Tensor & dst);
    int transpose();
    
    int mul(Tensor B, Tensor & target, int transA, int transB);
    int mul(Tensor B, Tensor & target);
    int mul(Tensor B, int transB);
    int mul(Tensor B);

    enum CMP{GE, GT, LE, LT, EQ};

    int find(Tensor::CMP cmp, float s, vector<int> & Y, vector<int> & X, vector<int> & I, vector<float> & S);

    
    int copyToVector(vector<int> & v);
    int copyToVector(vector<float> & v);
    int copyToVector(vector<Tensor> & v);
    
    int copyFromVector(vector<int> v);
    int copyFromVector(vector<float> v);
    int copyFromVector(vector<Tensor> v);

    
    //Tensor & conv(Tensor filter);       // convolution;
    //Pyramid & conv(Pyramid filters);    // convolution with filters
    int print();
    int release(); // release memory
    int write(FILE * file);
    int read(FILE * file);
    int write(char * fname);
    int read(char * fname);
};
