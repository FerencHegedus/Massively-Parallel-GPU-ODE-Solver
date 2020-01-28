#ifndef SINGLESYSTEM_PERTHREAD_INDEXINGMACROENABLED_H
#define SINGLESYSTEM_PERTHREAD_INDEXINGMACROENABLED_H


#define F(i)     F[tid + i*NT]
#define X(i)     X[tid + i*NT]
#define cPAR(i)  cPAR[tid + i*NT]
#define sPAR(i)  sPAR[i]
#define sPARi(i) sPARi[i]
#define ACC(i)   ACC[tid + i*NT]
#define ACCi(i)  ACCi[tid + i*NT]
#define EF(i)    EF[tid + i*NT]
#define TD(i)    TD[tid + i*NT]


#endif