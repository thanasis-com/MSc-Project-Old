#include "mex.h" 

// keyponts function is used to find the local maximum points in different scales

double* keypoints(int sigL, int rows, int cols, double *DOGResponse, int *num)
{
 int i,j,k,m,n,p,numRes,newrows,newcols;
 int signal;
 double max;
 double *res;
 double *oldres;

 newrows = rows+2;
 newcols = cols+2;
 numRes = 0;
 oldres = new double[4];
 res = new double[4];
 for (i=1;i<=sigL-2;i++)
 {
     for (j=1;j<=rows;j++)
	 {
         for (k=1;k<=cols;k++)
		 {
			 signal = 0;
			 max = DOGResponse[i*newrows*newcols+k*newrows+j];

             // check whether this point is a local maximum
 		     for (m=j-1;m<=j+1;m++)
			 {
				 for(n=k-1;n<=k+1;n++)
				 {
					 if ( DOGResponse[i*newrows*newcols+n*newrows+m]>=max && (m!=j || n!=k) )
					 {
						 signal = 1;
						 break;
					 }
					 if (DOGResponse[(i+1)*newrows*newcols+n*newrows+m]>=max)
					 {
						 signal = 1;
						 break;
					 }
					 if (DOGResponse[(i-1)*newrows*newcols+n*newrows+m]>=max)
					 {
						 signal = 1;
						 break;
					 }
				 }
				 if (signal == 1)
					 break;
			 }


			 if (signal == 1)
				 continue;
             
			 delete []res;
			 numRes = numRes+1;
			 res = new double[numRes*4];
			 for (p=0;p<(numRes-1)*4;p++)
			 {
				 res[p] = oldres[p];
			 }
			 delete []oldres;
			 oldres = new double[numRes*4];
		     res[p] = k;
			 res[p+1] = j;
			 res[p+2] = max;
			 res[p+3] = i+1;
			 for (m=0;m<numRes*4;m++)
			 {
				 oldres[m] = res[m];
			 }
			 *num = *num+1;
		 }
	 }
 }

 delete []oldres;
 return res;

}

// MEX function

void mexFunction(int nlhs,mxArray *plhs[], int nrhs,const mxArray *prhs[])

{
	int sigL, rows, cols;
	int i,j;
	int count;
	double *DOGResponse, *res;
	double *data;

	sigL = *(mxGetPr(prhs[0]));
	rows = *(mxGetPr(prhs[1]));
	cols = *(mxGetPr(prhs[2]));    
	
	DOGResponse = (mxGetPr(prhs[3]));

	count = 0;
	res = keypoints( sigL, rows, cols, DOGResponse, &count);

	plhs[0] = mxCreateDoubleMatrix(1,count*4,mxREAL);

	data = mxGetPr(plhs[0]); 

	for (i=0;i<count*4;i++)
	{
		data[i] = res[i];
	}

	delete []res;

}