% Copyright (c) 2016, Engr Sharjeel Abid Butt, PhD Scholar, Department of Electronic Engineering, 
% Faculty of Engineering and Technology, International Islamic University, Islamabad, Pakistan.
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without 
% modification, are permitted provided that the following conditions are 
% met:
% 
%     * Redistributions of source code must retain the above copyright 
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright 
%       notice, this list of conditions and the following disclaimer in 
%       the documentation and/or other materials provided with the distribution
%       
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
% POSSIBILITY OF SUCH DAMAGE.


function [ Data ] = specWindowOperation(spect, windowSize, overlapInterval, winPrint)
%specWindowOperation The goal of this function is to implement window operation on the spectrogram
%   Detailed explanation goes here

    switch nargin 
    
        case 1
        windowSize = 20;
        overlapInterval = 10;
        winPrint = false;
  
        case 2
        overlapInterval = 10;
        winPrint = false;
    
        case 3
        winPrint = false;
    end
    
    
    
    start = 0;
    stop  = windowSize;
    total = size(spect, 2);
    
    while(1)
        if(stop > total)
            Data = Data';
            return
        end
        if winPrint
            disp([start+1 stop]);
        end
        win = spect(:, start+1:stop);
        
        if start == 0
            Data = win(:);
        else
            Data = [Data win(:)];
        end
        start = start + overlapInterval;
        stop  = start + windowSize;
    end
        
end

