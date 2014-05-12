%% fig = nutilsquadmesh( x, y, val )
% Plot results of a 2D Nutils quad mesh which can be exported using the Bézier
% scheme.
%
%   x       X-coordinates of the elements
%   y       Y-coordinates of the elements
%   vals    Values at the points
%
% EXAMPLE:
% In Nutils code we can store the .mat file as follows:
%
% >>> points, colors = domain.elem_eval( [ geom + disp, stress(disp,geom)[0,1] ], ischeme='bezier4', separate=True )
% >>> with export.MatFile( 'solution' ) as mat:
% >>>   dat = {'points':points, 'colors': colors}
% >>>   mat.data(dat)
%
% Where we can plot it in MATLAB using:
%
% >>> load 'path/to/the/file.mat'
% >>> x = points(:,1), y = points(:,2);
% >>> nutilsquadmesh( x, y, colors );
function fig = nutilsquadmesh( x, y, val )
	if length(x) ~= length(y) || length(x) ~= length(val)
        error('The vectors x, y and colors should have the same length');
    end
    warning('This script can only be used to plot quad meshes with equal number of data-points per element. Furthermore the number of data-points in x and y direction of the element should match.');

    idxnan      = find(isnan(x));
    idxnotnan   = find(~isnan(x));

    pntperelem  = max(diff(idxnan))-1;
    elemcoord   = [x(idxnotnan), y(idxnotnan)];
    elemval     = val(idxnotnan);

    nelem       = length(idxnotnan)/pntperelem;
    npoint      = length(elemcoord);
    nscheme     = sqrt(pntperelem);

    if ceil(nelem) ~= floor(nelem)
        error('Number of elements should be an integer, probably the number of data points varies per element');
    end

    if ceil(nscheme) ~= floor(nscheme)
        error('The number of integration points in x and y direction do not match, you are probably using triangles');
    end

    fig = figure;
    hold on;
    for i = 1:pntperelem:npoint
        elemidx = i:i+pntperelem-1;
        elemcoord(elemidx,1);
        xx = reshape(elemcoord(elemidx,1), nscheme, nscheme);
        yy = reshape(elemcoord(elemidx,2), nscheme, nscheme);
        zz = reshape(elemval(elemidx), nscheme, nscheme);
        surf(xx,yy,zz);
    end
end
