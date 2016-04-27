function a = ichol(a)
	n = size(a,1);

	for k=1:n
		a(k,k) = sqrt(a(k,k));
		for i=(k+1):n
		    if (a(i,k)!=0)
		        a(i,k) = a(i,k)/a(k,k);            
		    endif
		endfor
		for j=(k+1):n
		    for i=j:n
		        if (a(i,j)!=0)
		            a(i,j) = a(i,j)-a(i,k)*a(j,k);  
		        endif
		    endfor
		endfor
	endfor

    for i=1:n
        for j=i+1:n
            a(i,j) = 0;
        endfor
    endfor            
endfunction