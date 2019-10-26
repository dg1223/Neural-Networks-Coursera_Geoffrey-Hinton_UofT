function G = configuration_goodness(rbm_w, visible_state, hidden_state)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
% <hidden_state> is a binary matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
% This returns a scalar: the mean over cases of the goodness (negative energy) of the described configurations.
    
  %%without loop: ( H' * W )' .* V, then take sum and mean  
    
  weights = rbm_w;                 % 100 x 256
  hidden = hidden_state;            % 100 x 10
  visible = visible_state';         % 10  x 256 
  
  goodness = 0;
  goodness_total = 0;
  num_configurations = size(hidden)(2);
  
  for cases = 1:num_configurations
    state_matrix = hidden(:,cases) * visible(cases,:);  % 100 x 256  
    [row,col] = find(state_matrix);   % gives you the row and col numbers of the non-zero values
  
    for state_index = 1:numel(row)
      goodness = goodness + weights(row(state_index),col(state_index));
    end
    
    goodness_total = goodness_total + goodness;
    goodness = 0;
  
  end

  G = goodness_total / num_configurations;
  
    %error('not yet implemented');
end
