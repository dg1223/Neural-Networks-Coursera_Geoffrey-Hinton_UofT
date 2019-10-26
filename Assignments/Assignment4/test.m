%a = test_rbm_w;                % weights
%b = test_hidden_state_1_case;   % hidden
%c = data_1_case';               % visible
%
%d = b * c;           % state_matrix
%[e,f] = find(d);     % row,col
%
%weights(row(state_index),col(state_index))
%
%    a = test_hidden_state_1_case;   % 100 x 1
%    b = data_1_case';               % 1   x 256
%    
%    num_configurations = size(b)(1);
%    
%    for cases = 1:num_configurations
%      derivative = a * b ;
%    end
%    
%    %d_G_by_rbm_w = derivative / num_configurations;

    
    update_hidden_1 = sample_bernoulli(visible_state_to_hidden_probabilities(test_rbm_w, data_10_cases));
    reconstruction   = sample_bernoulli(hidden_state_to_visible_probabilities(test_rbm_w, update_hidden_1));
    update_hidden_2  = sample_bernoulli(visible_state_to_hidden_probabilities(test_rbm_w, reconstruction));
    %update_hidden_2  = visible_state_to_hidden_probabilities(test_rbm_w, reconstruction);
    
    reconstruction_2 = sample_bernoulli(hidden_state_to_visible_probabilities(test_rbm_w, update_hidden_2));
    update_hidden_3  = visible_state_to_hidden_probabilities(test_rbm_w, reconstruction_2);
    reconstruction_3 = sample_bernoulli(hidden_state_to_visible_probabilities(test_rbm_w, update_hidden_3));
    update_hidden_4  = visible_state_to_hidden_probabilities(test_rbm_w, reconstruction_3);
    
    reconstruction_4 = sample_bernoulli(hidden_state_to_visible_probabilities(test_rbm_w, update_hidden_4));
    update_hidden_5  = visible_state_to_hidden_probabilities(test_rbm_w, reconstruction_4);
    
    reconstruction_5 = sample_bernoulli(hidden_state_to_visible_probabilities(test_rbm_w, update_hidden_5));
    update_hidden_6  = visible_state_to_hidden_probabilities(test_rbm_w, reconstruction_5);
    
    visible_data = sample_bernoulli(data_10_cases);
    
    ret_test = configuration_goodness_gradient(reconstruction, update_hidden_2);
    
    z1 = test_rbm_w' * test_hidden_state_10_cases;
    
    ret_cd5 = configuration_goodness_gradient(visible_data, update_hidden_1) - configuration_goodness_gradient(reconstruction, update_hidden_6);