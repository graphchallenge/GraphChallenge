function Y = inferenceReLUvec(W,bias,Y0);
% Performs ReLU inference  using input feature vector(s) Y0,
% DNN weights W, and constant bias.

  % Initialized feature vectors.
  Y = Y0;
  
  % Loop through each weight layer W{i}
  for i=1:length(W)
  
     % Propagate through layer.
     % Note: using graph convention of A(i,j) means connection from i *to* j,
     % that requires *left* multiplication feature *row* vectors.
     Z = Y*W{i};
     b = bias{i};

     % Apply bias to non-zero entries.
     Y = Z + (double(logical(Z)) .* b);
     
     % Threshold negative values.
     Y(Y < 0) = 0;
     
  end
  
return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Software Engineer: Dr. Jeremy Kepner
% MIT 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (c) <2019> Massachusetts Institute of Technology
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

