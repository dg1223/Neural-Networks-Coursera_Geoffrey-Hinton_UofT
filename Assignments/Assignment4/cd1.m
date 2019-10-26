function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
    
    visible_data = sample_bernoulli(visible_data);
    
    update_hidden_1  = sample_bernoulli(visible_state_to_hidden_probabilities(rbm_w, visible_data));
    reconstruction   = sample_bernoulli(hidden_state_to_visible_probabilities(rbm_w, update_hidden_1));
    %update_hidden_2  = sample_bernoulli(visible_state_to_hidden_probabilities(rbm_w, reconstruction));
    update_hidden_2  = visible_state_to_hidden_probabilities(rbm_w, reconstruction);
    
    % CD3
    reconstruction_2 = sample_bernoulli(hidden_state_to_visible_probabilities(rbm_w, update_hidden_2));
    update_hidden_3  = visible_state_to_hidden_probabilities(rbm_w, reconstruction_2);
    reconstruction_3 = sample_bernoulli(hidden_state_to_visible_probabilities(rbm_w, update_hidden_3));
    update_hidden_4  = visible_state_to_hidden_probabilities(rbm_w, reconstruction_3);
    
    ret = configuration_goodness_gradient(visible_data, update_hidden_1) - configuration_goodness_gradient(reconstruction, update_hidden_2);
    %error('not yet implemented');
end
