#include <vector>
#include <string>
#include <iostream>
#include <assert.h>
#include <string.h>
#include <math.h>
extern "C" {
#include <cblas.h>
}
#include "Tensor.h"

using namespace std;

Tensor::Tensor()
{
    _data = NULL;
    _size[0] = _size[1] = _size[2] = 0;
    _ownData = true;
}

Tensor::Tensor(bool ownData)
{
    _data = NULL;
    _size[0] = _size[1] = _size[2] = 0;
    _ownData = ownData;
}

Tensor::Tensor(int h, int w, int c)
{
    _data = new float[h*w*c];
    _size[0] = h;
    _size[1] = w;
    _size[2] = c;
    memset(_data, 0, sizeof(float)*this->getNumOfElem());
    _ownData = true;
}

Tensor::Tensor(int h, int w, int c, float * data, bool copy)
{
    _size[0] = h;
    _size[1] = w;
    _size[2] = c;
    if (copy)
    {
        _data = new float[h*w*c];
        memcpy(_data, data, sizeof(float)*this->getNumOfElem());
    }
    else
    {
        _data = data;
    }
    _ownData = true;
}

Tensor::Tensor(int size[3],  float * data, bool copy)
{
    new (this)Tensor(size[0], size[1], size[2], data, copy);
}

Tensor::Tensor(const Tensor & src)
{
    src.getSize(_size);
    //int n = src.getNumOfElem();
    _data = src.getData();
    //_data = new float[n];
    //memcpy(_data, src.getData(), sizeof(float)*n);
    _ownData = false;
}

Tensor::~Tensor()
{
    if (_ownData)
    {
        if (_data)
        {
            delete [] _data;
            _data = NULL;
        }
        _size[0] = _size[1] = _size[2] = 0;
    }
}

int Tensor::release()
{
    if (_data)
    {
        delete [] _data;
        _data = NULL;
    }
    _size[0] = _size[1] = _size[2] = 0;
    return 0;
}

bool Tensor::isValid()
{
    return (_data != NULL) & (_size[0] > 0) & (_size[1] > 0) & (_size[2] > 0);
}

int Tensor::zero()
{
    if (isValid())
    {
        memset(_data, 0, sizeof(float)*this->getNumOfElem());
    }
    return 0;
}

int Tensor::init(int h, int w, int c, float * data, bool copy)
{
    int n = this->getNumOfElem();
    _size[0] = h;
    _size[1] = w;
    _size[2] = c;
    if (copy)
    {
        if (n != h * w * c)
        {
            if (_data)
                delete [] _data;
            _data = new float[h*w*c];
        }
        memcpy(_data, data, sizeof(float)*this->getNumOfElem());
    }
    else
    {
        if (_data)
            delete [] _data;
        _data = data;
    }
    return 0;
}

int Tensor::init(int dims[3], float * data, bool copy)
{
    this->init(dims[0], dims[1], dims[2], data, copy);
    return 0;
}

int Tensor::memSet(float x)
{
	int dim = _size[0] * _size[1] * _size[2];
	//for(int i = 0; i < dim; i ++) _data[i] = x;
    float * p = _data;
    while(dim--)
        *(p++) = x;
    return 0;
}

int Tensor::reshape(Tensor src)
{
    if (this->getNumOfElem() == src.getNumOfElem())
    {
        src.getSize(_size);
    }
    else
    {
        if (_data)
            delete [] _data;
        src.getSize(_size);
        _data = new float[src.getNumOfElem()];
    }
    return 0;
}

int Tensor::reshape(int h, int w, int c)
{
    if (this->getNumOfElem() == w * h * c)
    {
        _size[0] = h;
        _size[1] = w;
        _size[2] = c;
    }
    else
    {
        if (_data)
            delete [] _data;
        _size[0] = h;
        _size[1] = w;
        _size[2] = c;
        _data = new float[h * w * c];
    }
    return 0;
}

int Tensor::reshape(int dims[3])
{
    this->reshape(dims[0], dims[1], dims[2]);
    return 0;
}

float * Tensor::getData() const
{
    return _data;
}

float & Tensor::at(int i, int j, int k)
{
    assert((i >= 0) & (j >= 0) & (k >= 0) );
    assert( (i < _size[0]) & (j < _size[1]) & (k < _size[2]) );
    return _data[i + j * _size[0] + k * _size[0] * _size[1]];
}

const int * Tensor::getSize()
{
    return _size;
}

int Tensor::getSize(int size[3]) const
{
    size[0] = _size[0];
    size[1] = _size[1];
    size[2] = _size[2];
    return 0;
}

int Tensor::getSize(int i)
{
    assert(i < 3);
    return _size[i];
}

int Tensor::getHeight()
{
    return _size[0];
}

int Tensor::getWidth()
{
    return _size[1];
}

int Tensor::getChannel()
{
    return _size[2];
}

int Tensor::getNumOfElem() const
{
    return _size[0] * _size[1] * _size[2];
}

int Tensor::getArea()
{
    return _size[0] * _size[1];
}

int Tensor::copyData(const float * data, int n, int bias)
{
    assert(bias + n <= this->getNumOfElem());
    memcpy(&_data[bias], data, sizeof(float)*n);
    return 0;
}

int Tensor::copyTo(Tensor & dst)
{
    dst.reshape(*this);
    dst.copyData(_data, getNumOfElem());
    return 0;
}

int Tensor::copyFrom( Tensor src)
{
    assert(src.isValid());
    this->reshape(src);
    int n = getNumOfElem();
    this->copyData(src.getData(), n);
    return 0;
}

int Tensor::crop(Tensor & dst, int y1, int x1, int y2, int x2)
{
    return crop(dst, y1, x1, 0, y2-y1, x2-x1, _size[2]);
}

int Tensor::crop(Tensor & dst, int y, int x, int z, int h, int w, int c)
{
    assert(y >= 0 & y + h <= _size[0] & \
            x >= 0 & x + w <= _size[1] & \
            z >= 0 & z + c <= _size[2]);
    dst.reshape(h,w,c);
    for (int k = 0; k < c; k++)
    {
        for (int i = 0; i < w; i++)
        {
            float * p = &_data[y + (x + i) * _size[0] + (k + z) * _size[0] * _size[1]];
            int c_bias = i * h + k * h * w;
            dst.copyData(p, h, c_bias);
        }
    }
    return 0;
}
/*//replace by a new crop function
  Tensor & slice(int start, int nChannels)
  {
  assert(start >= 0 & start + nChannels <= _size[2]);
  Tensor * s = new Tensor(_size[0], _size[1], nChannels);
  for (int k = 0; k < nChannels; k++)
  {
  int mapSize = _size[0] * _size[1];
  s->copyData(&_data[(start + k) * mapSize], k * mapSize,mapSize);
  }
  return *s;
  }
  */
int Tensor::padding(int py, int px, int pz, float val)
{
    assert( (py >= 0)  & (px >= 0)  & ( pz >= 0) );
    Tensor c(_size[0]+2*py, _size[1]+2*px, _size[2]+2*pz);
    float * p = c.getData();
    int n = c.getNumOfElem();
    for(int i = 0; i < n; i ++)
    {
    	*p++ = val;
    }
    int h = c.getSize(0);
    int w = c.getSize(1);
    for (int k = 0; k < _size[2]; k++)
    {
        for (int i = 0; i < _size[1]; i++)
        {
            float * p = &_data[i * _size[0] + k * _size[0] * _size[1]];
            int c_bias = py + (px + i) * h + (pz + k) * h * w;
            c.copyData(p, _size[0], c_bias);
        }
    }
    this->copyFrom(c);
    return 0;
}


int Tensor::shuffle(int axis, Tensor & target, vector<int> map)
{
    assert(this != &target);
    assert((int)map.size() == _size[axis]);
    assert( (axis >= 0) & (axis < 3) );
    target.reshape(*this);
    float * srcdata = _data;
    if (axis == 2) // high efficiency
    {
        int len = this->getArea();
        for (int k = 0; k < _size[2]; k++)
        {
            target.copyData(&srcdata[map[k]*len], len, k*len);
        }
    }
    if (axis == 1) // middle efficiency
    {
        int len = this->getSize(0);
        int step = this->getArea();
        for (int k = 0; k < _size[2]; k++)
        {
            for (int j = 0; j < _size[1]; j++)
            {
                target.copyData(&srcdata[map[j]*len+ k * step],\
                        len,\
                        j * len + k * step);
            }
        }
    }
    if (axis == 0)
    {
        for (int k = 0; k < _size[2]; k++)
        {
            for (int j = 0; j < _size[1] ; j++)
            {
                for (int i = 0; i < _size[0]; i++)
                {
                    target.at(i,j,k) = this->at(map[i],j,k);
                }
            }
        }
    }
    return 0;
}

int Tensor::shuffle(int axis, Tensor & target)
{
    vector<int> map;
    int n = this->getSize(axis);
    for (int i = 0; i < n; i++)
    {
        map.push_back(i);
    }
    for (int i = 0; i < n; i++)
    {
        int j = rand()%n;
        int tmp = map[j];
        map[j] = map[i];
        map[i] = tmp;
    }
    this->shuffle(axis, target, map);
    return 0;
}

int Tensor::flip(int axis, Tensor & target)
{
    vector<int> map;
    int n = this->getSize(axis);
    for (int i = 1; i <= n; i++)
    {
        map.push_back(n-i);
    }
    this->shuffle(axis, target, map);
    return 0;
}

int Tensor::permute(Tensor & target, int order[3])
{
    if (order[0] == 0 && order[1] == 1 && order[2] == 2)
    {
        this->copyTo(target);
    }
    else
    {
        target.reshape(_size[order[0]], _size[order[1]], _size[order[2]]);
        float * src = _data;
        float * dst = target.getData();
        int h = _size[0];
        int w = _size[1];
        int c = _size[2];
        int dim[3];
        for (dim[2] = 0; dim[2] < c; dim[2]++)
        {
            for (dim[1] = 0; dim[1] < w; dim[1]++)
            {
                for (dim[0] = 0; dim[0] < h; dim[0]++)
                {
                    target.at(dim[order[0]], dim[order[1]], dim[order[2]]) = \
                                                                             this->at(dim[0], dim[1], dim[2]);
                }
            }
        }
    }
}

int Tensor::permute(int order[3])
{
    Tensor dst;
    this->permute(dst, order);
    this->copyFrom(dst);
}

int Tensor::permute(int order_0, int order_1, int order_2)
{
    Tensor dst;
    int order[] = {order_0, order_1, order_2};
    this->permute(dst, order);
    this->copyFrom(dst);
}

bool Tensor::isMatrix()
{
    return isValid() & (_size[2] == 1);
}

int Tensor::FloatMatToTensor(Mat &img)
{
	reshape(img.channels(), img.cols, img.rows);
	int dim = img.cols * img.rows * img.channels();
	for(int i = 0; i < dim; i ++)
	{
		_data[i] = ((float*)img.data)[i];
	}
	int ord[3] = {2, 1, 0};
	permute(ord);
	return 0;
}

int Tensor::MatToTensor(Mat &img)
{
	reshape(img.channels(), img.cols, img.rows);
	int dim = img.cols * img.rows * img.channels();
	for(int i = 0; i < dim; i ++)
	{
		_data[i] = img.data[i];
	}
	int ord[3] = {2, 1, 0};
	permute(ord);
	return 0;
}


int Tensor::TensorToMat(Mat &img, int type)
{

	int ord[3] = {2, 1, 0};
	permute(ord);
	img.release();
	img = Mat(getChannel(), getWidth(), type);
	int dim = img.cols * img.rows * img.channels();
	for(int i = 0; i < dim; i ++)
	{
		(img.data)[i] = _data[i];
	}
	permute(ord);
	return 0;
}

int Tensor::transpose(Tensor & dst)
{
    assert(isMatrix());
    dst.reshape(_size[1], _size[0], 1);
    float * b = dst.getData();
    for (int i = 0; i < _size[0]; i++)
    {
        for (int j = 0; j < _size[1]; j++)
            b[j + i * _size[1]] = _data[i + j * _size[0]];    
    }
    return 0;
}

int Tensor::transpose()
{
    Tensor dst;
    transpose(dst);
    this->copyFrom(dst);
}

int Tensor::mul(Tensor B, Tensor & target, int transA, int transB)
{
    ///*
    assert(isMatrix() & B.isMatrix());
    target.reshape(transA ? _size[1] : _size[0], transB ? B.getHeight() : B.getWidth(), 1);
    target.zero();
    CBLAS_TRANSPOSE _transA = transA == 1 ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE _transB = transB == 1 ? CblasTrans : CblasNoTrans;
    int M = transA ? _size[1] : _size[0];
    int N = transB ? B.getHeight() : B.getWidth();
    int K = transB ? B.getWidth() : B.getHeight();
    //int lda = transA ? _size[0] : _size[1];
    //int ldb = transB ? _size[1] : _size[1];
    
    cblas_sgemm(CblasColMajor, _transA, _transB, M, N, K, 1.0, _data,
    		_size[0], B.getData(), B.getHeight(), 1.0, target.getData(), target.getHeight());
    //*/
    /*
    assert(isMatrix() & B.isMatrix());
    target.reshape(transA ? _size[1] : _size[0], transB ? B.getHeight() : B.getWidth(), 1);
    target.zero();
    float * b = B.getData();
    float * t = target.getData();
    for (int i = 0; i < _size[1] ; i++)
    {
        for (int j = 0; j < B.getWidth(); j++)
        {
            for (int k = 0; k < _size[0] ; k++)
            {
                t[i + j * _size[1]] += _data[k + i * _size[0]] * b[k + j * B.getHeight()];
                if (isnan(_data[k + i * _size[0]]))
                    printf("A isnan\n");
                if (isnan(b[k + j * B.getHeight()]))
                    printf("B isnan\n");
                //if (i == 4)
                //    printf("%f %f\n", _data[k + i * _size[0]], b[k + j * B.getHeight()]);
            }
        }
    }
    //*/
}



int Tensor::mul(Tensor B, Tensor & target)
{
    mul(B, target, 0, 0);
    return 0;
}
int Tensor::mul(Tensor B, int transB)
{
    Tensor target;
    mul(B, target, 0, transB);
    copyFrom(target);
    return 0;
}

int Tensor::mul(Tensor B)
{
    mul(B, 0);
    return 0;
}

int Tensor::print()
{
    if (isMatrix() && isValid())
    {
        for (int i = 0; i < _size[0]; i++)
        {
            for (int j = 0; j < _size[1]; j++)
                printf("%f ", _data[i + j*_size[0]]);
            printf("\n");
        }
    }
    else if (isValid())
    {
        for (int k = 0; k < _size[2] ; k++)
        {
            for (int i = 0; i < _size[0]; i++)
            {
                for (int j = 0; j < _size[1]; j++)
                    printf("%f ", _data[i + j * _size[0] + k * _size[0] * _size[1]]);
                printf("\n");
            }
            printf("\n");
        }
    }
    else
    {
        printf("Invalid Tensor\n");
    }
}

#define eps 0.0001

// unit vectors used to compute gradient orientation
float uu[9] = {1.0000,
    0.9397,
    0.7660,
    0.500,
    0.1736,
    -0.1736,
    -0.5000,
    -0.7660,
    -0.9397};
float vv[9] = {0.0000,
    0.3420,
    0.6428,
    0.8660,
    0.9848,
    0.9848,
    0.8660,
    0.6428,
    0.3420};


int Tensor::features(Tensor & tfeat, int nowSbin)
{
    int dims[3];
    this->getSize(dims);
    if(dims[2] != 3) printf("Invalid input");

    float * im = _data;

    // memory for caching orientation histograms & their norms
    int blocks[2];
    blocks[0] = (int)round((float)dims[0]/(float)nowSbin);
    blocks[1] = (int)round((float)dims[1]/(float)nowSbin);
    float *hist = new float[blocks[0]*blocks[1]*18];
    float *norm = new float[blocks[0]*blocks[1]];
    memset(hist, 0, sizeof(float)*blocks[0]*blocks[1]*18);
    memset(norm, 0, sizeof(float)*blocks[0]*blocks[1]);

    // memory for HOG features
    int out[3];
    out[0] = max(blocks[0]-2, 0);
    out[1] = max(blocks[1]-2, 0);
    out[2] = 27+4+1;

    //outDims.clear();
    //outDims.push_back(out[0]);
    //outDims.push_back(out[1]);
    //outDims.push_back(out[2]);
    float * feat = new float[out[0] * out[1] * out[2]];
    tfeat.init(out, feat);
    memset(feat, 0, sizeof(float)*out[0] * out[1] * out[2]);


    int visible[2];
    visible[0] = blocks[0]*nowSbin;
    visible[1] = blocks[1]*nowSbin;

    for (int x = 1; x < visible[1]-1; x++) {
        for (int y = 1; y < visible[0]-1; y++) {
            // first color channel
            float *s = im + min(x, dims[1]-2)*dims[0] + min(y, dims[0]-2);
            float dy = *(s+1) - *(s-1);
            float dx = *(s+dims[0]) - *(s-dims[0]);
            float v = dx*dx + dy*dy;

            // second color channel
            s += dims[0]*dims[1];
            float dy2 = *(s+1) - *(s-1);
            float dx2 = *(s+dims[0]) - *(s-dims[0]);
            float v2 = dx2*dx2 + dy2*dy2;

            // third color channel
            s += dims[0]*dims[1];
            float dy3 = *(s+1) - *(s-1);
            float dx3 = *(s+dims[0]) - *(s-dims[0]);
            float v3 = dx3*dx3 + dy3*dy3;

            // pick channel with strongest gradient
            if (v2 > v) {
                v = v2;
                dx = dx2;
                dy = dy2;
            }
            if (v3 > v) {
                v = v3;
                dx = dx3;
                dy = dy3;
            }

            // snap to one of 18 orientations
            float best_dot = 0;
            int best_o = 0;
            for (int o = 0; o < 9; o++) {
                float dot = uu[o]*dx + vv[o]*dy;
                if (dot > best_dot) {
                    best_dot = dot;
                    best_o = o;
                } else if (-dot > best_dot) {
                    best_dot = -dot;
                    best_o = o+9;
                }
            }

            // add to 4 histograms around pixel using linear interpolation
            float xp = ((float)x+0.5)/(float)nowSbin - 0.5;
            float yp = ((float)y+0.5)/(float)nowSbin - 0.5;
            int ixp = (int)floor(xp);
            int iyp = (int)floor(yp);
            float vx0 = xp-ixp;
            float vy0 = yp-iyp;
            float vx1 = 1.0-vx0;
            float vy1 = 1.0-vy0;
            v = sqrt(v);

            if (ixp >= 0 && iyp >= 0) {
                *(hist + ixp*blocks[0] + iyp + best_o*blocks[0]*blocks[1]) +=
                    vx1*vy1*v;
            }

            if (ixp+1 < blocks[1] && iyp >= 0) {
                *(hist + (ixp+1)*blocks[0] + iyp + best_o*blocks[0]*blocks[1]) +=
                    vx0*vy1*v;
            }

            if (ixp >= 0 && iyp+1 < blocks[0]) {
                *(hist + ixp*blocks[0] + (iyp+1) + best_o*blocks[0]*blocks[1]) +=
                    vx1*vy0*v;
            }

            if (ixp+1 < blocks[1] && iyp+1 < blocks[0]) {
                *(hist + (ixp+1)*blocks[0] + (iyp+1) + best_o*blocks[0]*blocks[1]) +=
                    vx0*vy0*v;
            }
        }
    }

    // compute energy in each block by summing over orientations
    for (int o = 0; o < 9; o++) {
        float *src1 = hist + o*blocks[0]*blocks[1];
        float *src2 = hist + (o+9)*blocks[0]*blocks[1];
        float *dst = norm;
        float *end = norm + blocks[1]*blocks[0];
        while (dst < end) {
            *(dst++) += (*src1 + *src2) * (*src1 + *src2);
            src1++;
            src2++;
        }
    }

    // compute features
    for (int x = 0; x < out[1]; x++) {
        for (int y = 0; y < out[0]; y++) {
            float *dst = feat + x*out[0] + y;
            float *src, *p, n1, n2, n3, n4;

            p = norm + (x+1)*blocks[0] + y+1;
            n1 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
            //	fprintf(stderr, "%f\n", n1);
            p = norm + (x+1)*blocks[0] + y;
            n2 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
            p = norm + x*blocks[0] + y+1;
            n3 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
            p = norm + x*blocks[0] + y;
            n4 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);

            float t1 = 0;
            float t2 = 0;
            float t3 = 0;
            float t4 = 0;

            // contrast-sensitive features
            src = hist + (x+1)*blocks[0] + (y+1);
            for (int o = 0; o < 18; o++) {
                float h1 = min(*src * n1, (float)0.2);
                float h2 = min(*src * n2, (float)0.2);
                float h3 = min(*src * n3, (float)0.2);
                float h4 = min(*src * n4, (float)0.2);
                *dst = 0.5 * (h1 + h2 + h3 + h4);
                t1 += h1;
                t2 += h2;
                t3 += h3;
                t4 += h4;
                dst += out[0]*out[1];
                src += blocks[0]*blocks[1];
            }

            // contrast-insensitive features
            src = hist + (x+1)*blocks[0] + (y+1);
            for (int o = 0; o < 9; o++) {
                float sum = *src + *(src + 9*blocks[0]*blocks[1]);
                float h1 = min(sum * n1, (float)0.2);
                float h2 = min(sum * n2, (float)0.2);
                float h3 = min(sum * n3, (float)0.2);
                float h4 = min(sum * n4, (float)0.2);
                *dst = 0.5 * (h1 + h2 + h3 + h4);
                dst += out[0]*out[1];
                src += blocks[0]*blocks[1];
            }

            // texture features
            *dst = 0.2357 * t1;
            dst += out[0]*out[1];
            *dst = 0.2357 * t2;
            dst += out[0]*out[1];
            *dst = 0.2357 * t3;
            dst += out[0]*out[1];
            *dst = 0.2357 * t4;

            // truncation feature
            dst += out[0]*out[1];
            *dst = 0;
        }
    }

    if(hist != NULL)
    {
        delete hist;
        hist = NULL;
    }

    if(norm != NULL)
    {
        delete norm;
        norm = NULL;
    }
    return 0;
    //return *tfeat;
}
//
//
//
// mex funtion resize
//
struct alphainfo {
    int si, di;
    float alpha;
}; //local funtion

void alphacopy(float *src, float *dst, struct alphainfo *ofs, int n) {
    struct alphainfo *end = ofs + n;
    while (ofs != end) {
        dst[ofs->di] += ofs->alpha * src[ofs->si];
        ofs++;
    }
} //local funtion

// resize along each column
// result is transposed, so we can apply it twice for a complete resize
void resize1dtran(float *src, int sheight, float *dst, int dheight, 
        int width, int chan) {
    float scale = (float)dheight/(float)sheight;
    float invscale = (float)sheight/(float)dheight;

    // we cache the interpolation values since they can be 
    // shared among different columns
    int len = (int)ceil(dheight*invscale) + 2*dheight;
    alphainfo ofs[len];
    int k = 0;
    for (int dy = 0; dy < dheight; dy++) {
        float fsy1 = dy * invscale;
        float fsy2 = fsy1 + invscale;
        int sy1 = (int)ceil(fsy1);
        int sy2 = (int)floor(fsy2);       

        if (sy1 - fsy1 > 1e-3) {
            assert(k < len);
            assert(sy1-1 >= 0);
            ofs[k].di = dy*width;
            ofs[k].si = sy1-1;
            ofs[k++].alpha = (sy1 - fsy1) * scale;
        }

        for (int sy = sy1; sy < sy2; sy++) {
            assert(k < len);
            assert(sy < sheight);
            ofs[k].di = dy*width;
            ofs[k].si = sy;
            ofs[k++].alpha = scale;
        }

        if (fsy2 - sy2 > 1e-3) {
            assert(k < len);
            assert(sy2 < sheight);
            ofs[k].di = dy*width;
            ofs[k].si = sy2;
            ofs[k++].alpha = (fsy2 - sy2) * scale;
        }
    }

    // resize each column of each color channel
    bzero(dst, chan*width*dheight*sizeof(float));
    for (int c = 0; c < chan; c++) {
        for (int x = 0; x < width; x++) {
            float *s = src + c*width*sheight + x*sheight;
            float *d = dst + c*width*dheight + x;
            alphacopy(s, d, ofs, k);
        }
    }
} //local function


// main function
// takes a float color image and a scaling factor
// returns resized image

void Tensor::resize(float scale)
{
    Tensor dst;
    this->resize(dst, scale);
    this->copyFrom(dst);
    dst.release();
}

void Tensor::resize(Tensor & tfeat, float scale)
{
    float * src = (float *)_data;
    int sdims[3];
    this->getSize(sdims);
    assert(sdims[2] == 3);
    assert(scale <= 1);

    int ddims[3];
    ddims[0] = (int)round(sdims[0]*scale);
    ddims[1] = (int)round(sdims[1]*scale);
    ddims[2] = sdims[2];
    tfeat.reshape(ddims);
    tfeat.zero();
    float *dst = (float *)tfeat.getData();

    float *tmp = (float *)new float[ddims[0]*sdims[1]*sdims[2]];
    memset(tmp, 0, ddims[0]*ddims[1]*ddims[2]);
    resize1dtran(src, sdims[0], tmp, ddims[0], sdims[1], sdims[2]);
    resize1dtran(tmp, sdims[1], dst, ddims[1], ddims[0], sdims[2]);
    delete [] tmp;
}


int Tensor::sum()
{
    Tensor dst;
    this->sum(dst);
    this->copyFrom(dst);
    return 0;
}

int Tensor::sum(Tensor & dst)
{
    // only support sum along channels
    dst.reshape(_size[0], _size[1], 1);
    dst.zero();
    float * p = dst.getData();
    float * src = _data;
    int n = _size[0] * _size[1];
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < _size[2]; j++)
        {
            *p += src[j*n];
            p ++;
        }
    }

}

int Tensor::add(Tensor B)
{
    assert(_size[0] == B.getHeight() && _size[1] == B.getWidth() && _size[2] == B.getChannel());
    int n = getNumOfElem();
    float * p = _data;
    float * b = B.getData();
    for (int i = 0; i < n; i++)
    {
        *p += *b;
        p ++;
        b ++;
    }
    return 0;
}

int Tensor::add(Tensor & dst, float scaleA, float bias)
{
    Tensor B;
    this->add(B, dst, scaleA, 0, bias); 
    return 0;
}
int Tensor::add(float scaleA, float bias)
{
    Tensor dst, B;
    this->add(B, dst, scaleA, 0, bias);
    this->copyFrom(dst);
    return 0;
}
    
int Tensor::add(Tensor B, Tensor & dst, float scaleA, float scaleB, float bias)
{
    dst.reshape(_size);
    if (B.isValid())
    {
        float * p = dst.getData();
        float * src = _data;
        int n = getNumOfElem();
        for (int i = 0; i < n; i++, p++, src++)
        {
            *p = (*src) * scaleA + bias;
        }
    }
    else
    {
        assert(_size[0] == B.getHeight() && _size[1] == B.getWidth() && _size[2] == B.getChannel());
        float * p = dst.getData();
        float * src = _data;
        float * b = B.getData();
        int n = getNumOfElem();
        for (int i = 0; i < n; i++, p++, src++)
        {
            *p = (*src) * scaleA + (*b) * scaleB + bias;
        }
    }
}


int Tensor::add(float val)
{
	int n = getNumOfElem();
	float * p = _data;
	for (int i = 0; i < n; i++)
	{
		*p += val;
		p ++;
	}
}

int Tensor::minus(Tensor B)
{
    assert(_size[0] == B.getHeight() && _size[1] == B.getWidth() && _size[2] == B.getChannel());
    int n = getNumOfElem();
    float * p = _data;
    float * b = B.getData();
    for (int i = 0; i < n; i++)
    {
        *p -= *b;
        p ++;
        b ++;
    }
    return 0;
}

int Tensor::div(Tensor B)
{
    assert(_size[0] == B.getHeight() && _size[1] == B.getWidth() && _size[2] == B.getChannel());
    int n = getNumOfElem();
    float * p = _data;
    float * b = B.getData();
    for (int i = 0; i < n; i++)
    {
        *p /= *b;
        p ++;
        b ++;
    }
    return 0;
}


int Tensor::find(Tensor::CMP cmp, float thresh, vector<int> & tmpI, vector<int> & tmpY, vector<int> & tmpX, vector<float> & tmpS)
{
    tmpI.clear();
    tmpX.clear();
    tmpY.clear();
    tmpS.clear();
    int cnt = getNumOfElem();
    int count = 0;
    if (cmp == GT)
    {
        for (int i = 0; i < cnt; i++)
        {
            if (_data[i] > thresh)
            {
                tmpI.push_back(i);
                tmpX.push_back(i/_size[0]);
                tmpY.push_back(i%_size[0]);
                tmpS.push_back(_data[i]);
                count++;
            }
        }
    }
    else if (cmp == GE)
    {
        for (int i = 0; i < cnt; i++)
        {
            if (_data[i] >= thresh) 
            {
                tmpI.push_back(i);
                tmpX.push_back(i/_size[0]);
                tmpY.push_back(i%_size[0]);
                tmpS.push_back(_data[i]);
                count++;
            }
        }
    }
    else if (cmp == LT)
    {
        for (int i = 0; i < cnt; i++)
        {
            if (_data[i] < thresh)
            {
                tmpI.push_back(i);
                tmpX.push_back(i/_size[0]);
                tmpY.push_back(i%_size[0]);
                tmpS.push_back(_data[i]);
                count++;
            }
        }
    }
    else if (cmp == LE)
    {
        for (int i = 0; i < cnt; i++)
        {
            if (_data[i] <= thresh) 
            {
                tmpI.push_back(i);
                tmpX.push_back(i/_size[0]);
                tmpY.push_back(i%_size[0]);
                tmpS.push_back(_data[i]);
                count++;
            }
        }

    }
    else if (cmp == EQ)
    {
        for (int i = 0; i < cnt; i++)
        {
            if (_data[i] == thresh) // maybe fabs(_data[i] - thresh) < 1e-6 be better
            {
                tmpI.push_back(i);
                tmpX.push_back(i/_size[0]);
                tmpY.push_back(i%_size[0]);
                tmpS.push_back(_data[i]);
                count++;
            }
        }
    }
    else
    {
        fprintf(stderr, "cmp error");
    }
    return count; 
}

int Tensor::copyToVector(vector<int> & v)
{
    int n = getNumOfElem();
    v.resize(n);
    for (int i = 0; i < n; i++)
    {
        v[i] = (int)_data[i];
    }
    return 0;
}
int Tensor::copyToVector(vector<float> & v)
{
    int n = getNumOfElem();
    v.resize(n);
    for (int i = 0; i < n; i++)
    {
        v[i] = _data[i];
    }
    return 0;
}

int Tensor::copyToVector(vector<Tensor> & v)
{
    assert(isMatrix() && isValid());
    for (int i = 0; i < (int)v.size(); i++)
    {
        v[i].release();
    }
    v.resize(_size[0], Tensor());
    for (int i = 0; i < _size[0]; i++)
    {
        v[i].reshape(_size[1], _size[2], 1);
        float * p = v[i].getData();
        for (int j = 0; j < _size[1]; j++)
        {    
            for (int k = 0; k < _size[2]; k++)
            {
                p[j + k * _size[2]] = _data[i + j * _size[0] + k * _size[1] * _size[2]];
            }
        }
    }
    return 0;
}
    
int Tensor::copyFromVector(vector<int> v)
{
    int n = v.size();
    reshape(n, 1, 1);
    for (int i = 0; i < n; i++)
    {
        _data[i] = (float)v[i];
    }
    return 0;
}
int Tensor::copyFromVector(vector<float> v)
{
    int n = v.size();
    reshape(n, 1, 1);
    for (int i = 0; i < n; i++)
    {
        _data[i] = v[i];
    }
    return 0;
}
int Tensor::copyFromVector(vector<Tensor> v)
{
    int nv = v.size();
    assert(nv > 0);
    int h = v[0].getHeight();
    int w = v[0].getWidth();
    for (int i = 1; i < nv; i++)
    {
        assert(v[i].isMatrix());
        assert(h == v[i].getHeight());
        assert(w == v[i].getWidth());
    }
    reshape(nv, h, w);
    for (int i = 0; i < nv; i++)
    {
        for (int j = 0; j < h; j++)
        {
            for (int k = 0; k < w; k++)
            {
                at(i,j,k) = v[i].at(j,k,1);
            }
        }
    }
    return 0;
}

int Tensor::write(char * fname)
{
    FILE * fp = fopen(fname, "wb");
    write(fp);
    fclose(fp);
}

int Tensor::read(char * fname)
{
    FILE * fp = fopen(fname, "rb");
    read(fp);
    fclose(fp);
}

int Tensor::write(FILE * fp)
{
    fwrite(_size, sizeof(int), 3, fp);
    fwrite(_data, sizeof(float), getNumOfElem(), fp);
    return 0;
}

int Tensor::read(FILE * fp)
{
	int tsize[3];
    int n = fread(tsize, sizeof(int), 3, fp);
    if (n < 3)
        return -1;
    reshape(tsize);
    n = fread(_data, sizeof(float), getNumOfElem(), fp);
    if (n < getNumOfElem())
        return -1;
    return 0;
}

int Tensor::dot(Tensor B, Tensor & target)
{
	assert(_size[0] == B.getHeight() && _size[1] == B.getWidth() && _size[2] == B.getChannel());
	target.reshape(B);
	int n = getNumOfElem();
	float * x = _data;
	float * y = B.getData();
	float * z = target.getData();
	for (int i = 0; i < n; i++)
	{
		*z++ = (*x++) * (*y++);
	}
	return 0;
}

int Tensor::dot(Tensor B)
{
	assert(_size[0] == B.getHeight() && _size[1] == B.getWidth() && _size[2] == B.getChannel());
	int n = getNumOfElem();
	float * x = _data;
	float * y = B.getData();
	for (int i = 0; i < n; i++)
	{
		*x++ *= (*y++);
	}
	return 0;
}
