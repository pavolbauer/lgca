function [ixR iyR] = tlgc(X,m,n,p)
% Transforms cartesian coordinates onto the voronoi mesh
% p is position in the cell, e.g. 0=center, 1..6 are border positions
% FIX: this is a low-performance version using dynamics memory allocation for a small input-size m
ixR=[];
iyR=[];
inputL = length(m);
for i=1:inputL
    if(mod(m(i),2)==0)
        iy=n(i)+0.5;
    else
        iy=n(i);
    end
    ix=X(m(i));

    factor=0.25;
    if(p==2)
        iy=iy-factor;
        ix=ix+factor;
    elseif(p==1)
        iy=iy+factor;
        ix=ix+factor;
    elseif(p==6)
        iy=iy+factor+0.1;
    elseif(p==5)
        iy=iy+factor;
        ix=ix-factor;    
    elseif(p==4)
        iy=iy-factor;
        ix=ix-factor;
    elseif(p==3)
        iy=iy-factor-0.1;
    end
    ixR=[ixR; ix];
    iyR=[iyR; iy];
end
end
