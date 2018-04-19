function H=montageplot(X,chanlocs,y1,y2,ch,TU,TL)
% Plots a montage of the 3-way array X with y1 number of rows and y2
% number of columns
%
% Written by Morten Mørup
%
% Usage:
%   H=montageplot(X,chanlocs,y1,y2,ch,TU,TL)
%
% Input:
%   X           3-way array
%   chanlocs    The channel locations as defined by the EEG structure in
%               EEGLAB (optional)
%   y1          Number of rows in montage (default: ceil(sqrt(size(X,1))))
%   y2          Number of columns in montage (default: ceil(sqrt(size(X,1))))
%   ch          what are the currently investigated channel in the plot (optional)
%   TU          Upper significance limit (optional)
%   TL          Lower significance limit (optional)
%
% Output:
%   H           Handle to the plot, type get(H) to access handle parameters
%
% Copyright (C) Morten Mørup and Technical University of Denmark, 
% September 2006
%                                          
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 2 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

if nargin<2
    chanlocs=[];
end
if nargin<3
    y1=ceil(sqrt(size(X,1)));
end
if nargin<4
    y2=ceil(sqrt(size(X,1)));
end
if nargin<5
    ch=1;
end
if nargin<6
    TU=[];
end
if nargin<7
    TL=[];
end
gcf;

% Create montage
maxch=size(X,1);
N2=size(X,2);
N3=size(X,3);
Y=zeros(y1*N2,y2*N3);
if ~isempty(TU)
    TUt=zeros(y1*N2,y2*N3);
end
if ~isempty(TL)
    TLt=zeros(y1*N2,y2*N3);
end
for i=1:y1
     for j=1:y2
         if (i-1)*y2+j+ch-1<=maxch
             x((i-1)*y2+j+ch-1)=((j-1)*N3+1);
             y((i-1)*y2+j+ch-1)=(i-1)*N2+1;
             T=[squeeze(X((i-1)*y2+j+ch-1,:,:))];
             Y(((i-1)*N2+1):(i*N2),((j-1)*N3+1):(j*N3))=T;
             if ~isempty(TU)
                 T=[squeeze(TU((i-1)*y2+j+ch-1,:,:))];
                 TUt(((i-1)*N2+1):(i*N2),((j-1)*N3+1):(j*N3))=T;
             end
             if ~isempty(TL)
                 T=[squeeze(TL((i-1)*y2+j+ch-1,:,:))];
                 TLt(((i-1)*N2+1):(i*N2),((j-1)*N3+1):(j*N3))=T;
             end
         end
     end
end


% Plot the montage
H=imagesc(Y);
hold on;

% Plot two tailed significant region
if exist('TUt','var')
    T=zeros(size(TUt));
    I=find(TUt>0);
    T(I)=1;
    contour(T,1,'LineColor','Red','linewidth',2)
end
if exist('TLt','var')
    T=zeros(size(TLt));
    I=find(TLt<0);
    T(I)=1;
    contour(T,1,'LineColor','Green','linewidth',2)
end


% add time-frequency edges and display channel names
for i=1:y1+1
      plot([0.5 size(Y,2)+0.5],[(i-1)*N2+0.5 (i-1)*N2+0.5], 'k-')
end
for j=1:y2+1
      plot([((j-1)*N3)+0.5 ((j-1)*N3)+0.5], [0.5 size(Y,1)+0.5], 'k-')
end
if ~isempty(chanlocs)
   for k=ch:size(X,1)
        if k<=y1*y2+ch-1
            text(x(k),y(k),chanlocs(k).labels,'FontSize',8,'FontWeight','bold');
        end
   end
end


hold off;
