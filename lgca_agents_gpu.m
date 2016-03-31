function [ output_args ] = lgca_agents_gpu( NumBlocks, NumThreads, LGCASize, its )
%lgca_agents_gpu Lattice Gas Cellular Automaton on GPU.
% This script calls one of the GPU kernels and visualizes the iterative propagation of agents.
% Matlab Parallel Computing Toolbox v6.0 or higher is required due to CUDA support.
% Parameters:
% * NumBlocks: Number of CUDA Blocks (nxm)
% * NumThreads: Number of CUDA threads per block (nxm)
% * GridSize: Size of the LGCA grid
% * its: Number of iterations
%
% P. Bauer, 2011

%default arguments
switch nargin
    case 0
        NumBlocks = [20 20];
        NumThreads = [20 20];
        LGCASize = [400 400];
	its=100;
    case 1
        NumThreads = [20 20];
        LGCASize = [400 400];
	its=100;
    case 2
        LGCASize = [400 400];
	its=100;
    case 3
	its=100;
end


reset(parallel.gpu.GPUDevice.current())
kernel=parallel.gpu.CUDAKernel('lgca_monotone_collisions.ptx','lgca_monotone_collisions.cu','runLGCA');
kernel.ThreadBlockSize = NumBlocks;
kernel.GridSize = NumThreads;

m=int16(LGCASize(1));
n=int16(LGCASize(2));

caGrid{1} = uint32(zeros(m,n));
caGrid{2} = uint32(zeros(m,n));
caGrid{3} = uint32(zeros(m,n));
caGrid{4} = uint32(zeros(m,n));
caGrid{5} = uint32(zeros(m,n));
caGrid{6} = uint32(zeros(m,n));

%randomly distributed initial population
%     as=60000;
%     for (j=1:as)
%        ax=round(rand()*3998)+1;
%        ay=round(rand()*3998)+1;
%        grid2=round(rand*5)+1;
%        caGrid{grid2}(ax,ay)=j;
%     end

%uniformly distributed initial population
    agent=1;
    for (g=1:6)
        for (i=1:m)
            for (j=1:n)
                caGrid{g}(i,j)=agent;
                agent=agent+1;
            end
        end
    end
    

% Prepare hexagonal grid for plot
dimsize = sqrt(3) / 2;
[K L] = meshgrid(0:1:13);
npl = size(K,1);
K = dimsize * K;
L = L + repmat([0 0.5],[npl,npl/2]);
[XV YV] = voronoi(K(:),L(:)); 

Db1 = parallel.gpu.GPUArray( caGrid{1} );
Db2 = parallel.gpu.GPUArray( caGrid{2} );
Db3 = parallel.gpu.GPUArray( caGrid{3} );
Db4 = parallel.gpu.GPUArray( caGrid{4} );
Db5 = parallel.gpu.GPUArray( caGrid{5} );
Db6 = parallel.gpu.GPUArray( caGrid{6} );

for iteration=1:its
    plotLGCA(XV,YV,K,caGrid);
    drawnow(); 

    tic
    [Db1, Db2, Db3, Db4, Db5, Db6]=feval( kernel, Db1, Db2, Db3, Db4, Db5, Db6, m, n );
    
    %Display the board
    caGrid{1}=gather(Db1);
    caGrid{2}=gather(Db2);
    caGrid{3}=gather(Db3);
    caGrid{4}=gather(Db4);
    caGrid{5}=gather(Db5);
    caGrid{6}=gather(Db6); 
    toc
end
end

function plotLGCA(XV,YV,X,caGrid)
clf
plot(XV,YV,'b-')
axis equal, axis([0 10 0 12]), zoom on
hold on
ix=X';

for(grid=1:6)
    [i j] = find(caGrid{grid});
    matrix=zeros(size(i,1),4);
    if(i)
        for(count=1:size(i))
            matrix(count,1)=caGrid{grid}(i(count),j(count));
            matrix(count,2)=i(count);
            matrix(count,3)=j(count);
            matrix(count,4)=grid;
        end

    [x y]=tlgc(ix,i+1,j,grid);
    plot(x,y,'LineStyle','none',...
                    'Marker','s',...
                    'MarkerEdgeColor','k',...
                    'MarkerFaceColor','g',...
                    'MarkerSize',5)
    end
    matrix=sortrows(matrix);
    for(c=1:size(matrix,1))
        fprintf ('( %i: %i | %i | %i )  ',matrix(c,1),matrix(c,2),matrix(c,3),matrix(c,4));
    end
end
fprintf('\n');
end
