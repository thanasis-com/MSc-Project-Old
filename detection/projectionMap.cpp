#include "mex.h" 

// Method used to get magnitude projection image and orientation projection image

void projection(int rows, int cols, double *mag, double beta, double *negx, double *negy, 
        double *O, double *M)
{
 int r,c,nx,ny;
 int i;
 int signal;
 double gold;
 
 // for each pixel
 for (r=0;r<rows;r++)
 {
     for (c=0;c<cols;c++)
     {
         signal = 0;
         // compute projection position
         nx = negx[c*rows+r];
         ny = negy[c*rows+r];
         if ( (mag[c*rows+r]>beta) && (ny<=rows) && (ny>=1) &&
                 (nx<=cols) && (nx>=1) )
         {
                 // compute orientation projection
                 O[(nx-1)*rows+(ny-1)] = O[(nx-1)*rows+(ny-1)]-1; 
                 // compute magnitude projection
                 M[(nx-1)*rows+(ny-1)] = M[(nx-1)*rows+(ny-1)]-mag[c*rows+r];          
             
         }
     }
 }
 

}

// MEX function
void mexFunction(int nlhs,mxArray *plhs[], int nrhs,const mxArray *prhs[])

{
    
	int rows, cols;
	double *mag, *negx, *negy, *O, *M;
	double beta;
    int i, j;
    


	rows = *(mxGetPr(prhs[0]));
	cols = *(mxGetPr(prhs[1]));
    mag = (mxGetPr(prhs[2]));
    beta = *(mxGetPr(prhs[3]));
    negx = (mxGetPr(prhs[4]));
    negy = (mxGetPr(prhs[5]));
    
    
    plhs[0] = mxCreateDoubleMatrix(rows,cols,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(rows,cols,mxREAL);

	O = mxGetPr(plhs[0]); 
    M = mxGetPr(plhs[1]);
    
  	for (i=0;i<rows;i++)
	{
        for (j=0;j<cols;j++)
            O[j*rows+i] = 0;
	}  
    
    for (i=0;i<rows;i++)
	{
        for (j=0;j<cols;j++)
            M[j*rows+i] = 0;
	}       
   
    
    

	projection( rows, cols, mag, beta, negx, negy, O, M);
    
    
    
}