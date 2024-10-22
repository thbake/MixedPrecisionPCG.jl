using MATLAB, MatrixDepot

# Get some sparse SPD matrix
A = sparse(matrixdepot("HB/bcsstk01")) 

# Construct preconditioner with incomplete Cholesky factorization.
# Later I could try implementing own incomplete Cholesky factorizations.
L = mxcall(:ichol, 1, A) 
L2 = mat"ichol($A, struct('type', 'ict', 'droptol', 1e-2))"

