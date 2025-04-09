% 01 AUG 2023

function [output_lower, output_upper, output_mean] = T2_multioutput_defuzzification_layer(x,lower_firing_strength,upper_firing_strength, learnable_parameters,number_outputs, output_type,type_reduction_method,mbs,number_mf,u)
% v0.2 compatible with minibatch
% !!! not compatible with multiple outputs !!! will be written when needed
%
%
% calculating the weighted sum with firts calcutating the weighted
% elements then adding them
%
% @param output -> output
%
%       (1,1,mbs) tensor
%       mbs = mini-batch size
%       (:,:,1) -> defuzzified output of the first element of the batch
%
% @param input 1 -> normalized_firing_strength
%
%       (rc,1,mbs) tensor
%       rc = number of rules
%       mbs = mini-batch size
%       (1,1,1) -> normalized firing strength of the first rule of the
%       first element of the batch
%
% @param input 2 -> output_mf
%
%       (rc,1) vector
%       rc = number of rules
%       (1,1) -> constant or value of the first output membership function

if output_type == "singleton"

    if type_reduction_method == "SM"

        normalized_lower_firing_strength = lower_firing_strength./sum(lower_firing_strength, 1);
        normalized_upper_firing_strength = upper_firing_strength./sum(upper_firing_strength, 1);

        normalized_lower_firing_strength = repmat(normalized_lower_firing_strength,1,number_outputs);
        normalized_upper_firing_strength = repmat(normalized_upper_firing_strength,1,number_outputs);

        output_lower = normalized_lower_firing_strength.* learnable_parameters.singleton.c;% first we multiply elementwise our firing strengths with output memberships
        output_upper = normalized_upper_firing_strength.* learnable_parameters.singleton.c;

        output_lower = sum(output_lower, 1); %then we sum with respect to dimension 1
        output_upper = sum(output_upper, 1); %then we sum with respect to dimension 1

        output_mean = (output_lower + output_upper)./2;

    elseif type_reduction_method == "BMM"

        normalized_lower_firing_strength = lower_firing_strength./sum(lower_firing_strength, 1);
        normalized_upper_firing_strength = upper_firing_strength./sum(upper_firing_strength, 1);

        normalized_lower_firing_strength = repmat(normalized_lower_firing_strength,1,number_outputs);
        normalized_upper_firing_strength = repmat(normalized_upper_firing_strength,1,number_outputs);

        output_lower = normalized_lower_firing_strength.* learnable_parameters.singleton.c;% first we multiply elementwise our firing strengths with output memberships
        output_upper = normalized_upper_firing_strength.* learnable_parameters.singleton.c;

        output_lower = sum(output_lower, 1); %then we sum with respect to dimension 1
        output_upper = sum(output_upper, 1); %then we sum with respect to dimension 1

        alpha = sigmoid(learnable_parameters.singleton.alpha);

        output_mean = (output_lower .* alpha) + (output_upper .* (1 - alpha));

    elseif type_reduction_method == "NT"

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);

        numerator_lower = lower_firing_strength_temp.* learnable_parameters.singleton.c;
        numerator_upper = upper_firing_strength_temp.* learnable_parameters.singleton.c;

        numerator_lower = 2*sum(numerator_lower, 1);
        numerator_upper = 2*sum(numerator_upper, 1);

        denominator = sum(lower_firing_strength, 1) + sum(upper_firing_strength, 1);

        output_lower = numerator_lower./denominator;
        output_upper = numerator_upper./denominator;

        output_mean = (output_lower + output_upper)./2;

    elseif type_reduction_method == "NT_alpha" || type_reduction_method == "NT_multi_alpha" 

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);

        %         alpha = sigmoid(learnable_parameters.linear.alpha)^2;
        alpha = sigmoid(learnable_parameters.singleton.alpha);

        numerator_lower = (lower_firing_strength_temp.* learnable_parameters.singleton.c) .* alpha;
        numerator_upper = (upper_firing_strength_temp.* learnable_parameters.singleton.c) .* (1 - alpha);

        numerator_lower = 2*sum(numerator_lower, 1);
        numerator_upper = 2*sum(numerator_upper, 1);

        denominator = sum(lower_firing_strength .* alpha, 1) + sum(upper_firing_strength .* (1 - alpha), 1);

        output_lower = numerator_lower./denominator;
        output_upper = numerator_upper./denominator;

        output_mean = (output_lower + output_upper)./2;


    elseif type_reduction_method == "NT_alpha2"

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);

        alpha = sigmoid(learnable_parameters.singleton.alpha).^2;

        numerator_lower = (lower_firing_strength_temp.* learnable_parameters.singleton.c) .* alpha;
        numerator_upper = (upper_firing_strength_temp.* learnable_parameters.singleton.c) .* (1 - alpha);

        numerator_lower = 2*sum(numerator_lower, 1);
        numerator_upper = 2*sum(numerator_upper, 1);

        denominator = sum(lower_firing_strength .* alpha, 1) + sum(upper_firing_strength .* (1 - alpha), 1);

        output_lower = numerator_lower./denominator;
        output_upper = numerator_upper./denominator;

        output_mean = (output_lower + output_upper)./2;

    elseif type_reduction_method == "NTv2" || type_reduction_method == "NTv2_multi_alpha" || type_reduction_method == "NTv3" || type_reduction_method == "NTv3_multi_alpha"

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);

        alpha = sigmoid(learnable_parameters.singleton.alpha);
        beta = sigmoid(learnable_parameters.singleton.beta);


        numerator_lower = sum((lower_firing_strength_temp.* learnable_parameters.singleton.c) .* alpha,1) + sum((upper_firing_strength_temp.* learnable_parameters.singleton.c) .* (1 - alpha),1);
        numerator_upper = sum((lower_firing_strength_temp.* learnable_parameters.singleton.c) .* (1 - beta),1) + sum((upper_firing_strength_temp.* learnable_parameters.singleton.c) .* beta,1);

        denominator_lower = sum(lower_firing_strength .* alpha, 1) + sum(upper_firing_strength .* (1 - alpha), 1);
        denominator_upper = sum(lower_firing_strength .* (1 - beta), 1) + sum(upper_firing_strength .* beta, 1);

        output_lower = numerator_lower./denominator_lower;
        output_upper = numerator_upper./denominator_upper;

        output_mean = (output_lower + output_upper)./2;

    elseif type_reduction_method == "NTv4"

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);

        alpha = sigmoid(10*learnable_parameters.singleton.alpha);
        beta = sigmoid(10*learnable_parameters.singleton.beta);

        numerator_lower = sum((lower_firing_strength_temp.* learnable_parameters.singleton.c) .* alpha,1) + sum((upper_firing_strength_temp.* learnable_parameters.singleton.c) .* (1 - alpha),1);
        numerator_upper = sum((lower_firing_strength_temp.* learnable_parameters.singleton.c) .* beta,1) + sum((upper_firing_strength_temp.* learnable_parameters.singleton.c) .* (1 - beta),1);

        denominator_lower = sum(lower_firing_strength .* alpha, 1) + sum(upper_firing_strength .* (1 - alpha), 1);
        denominator_upper = sum(lower_firing_strength .* beta, 1) + sum(upper_firing_strength .* (1 - beta), 1);

        output_lower = numerator_lower./denominator_lower;
        output_upper = numerator_upper./denominator_upper;

        output_mean = (output_lower + output_upper)./2;



    elseif type_reduction_method == "NTv2_alpha2" || type_reduction_method == "NTv3_alpha2"

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);

        alpha = sigmoid(learnable_parameters.singleton.alpha).^2;
        beta = sigmoid(learnable_parameters.singleton.beta).^2;


        numerator_lower = sum((lower_firing_strength_temp.* learnable_parameters.singleton.c) .* alpha,1) + sum((upper_firing_strength_temp.* learnable_parameters.singleton.c) .* (1 - alpha),1);
        numerator_upper = sum((lower_firing_strength_temp.* learnable_parameters.singleton.c) .* (1 - beta),1) + sum((upper_firing_strength_temp.* learnable_parameters.singleton.c) .* beta,1);

        denominator_lower = sum(lower_firing_strength .* alpha, 1) + sum(upper_firing_strength .* (1 - alpha), 1);
        denominator_upper = sum(lower_firing_strength .* (1 - beta), 1) + sum(upper_firing_strength .* beta, 1);

        output_lower = numerator_lower./denominator_lower;
        output_upper = numerator_upper./denominator_upper;

        output_mean = (output_lower + output_upper)./2;

    elseif type_reduction_method == "KM"


        delta_f = upper_firing_strength - lower_firing_strength;
        delta_f = permute(delta_f,[3 1 2]);

        payda2 = delta_f*u;

%         pay2 = reshape((permute(learnable_parameters.singleton.c,[3 2 1]).*permute(delta_f,[1 3 2])),[mbs*number_outputs,number_mf])*u;
        pay2 = pagemtimes((permute(learnable_parameters.singleton.c,[3 1 2]).*delta_f),u);

%         pay2 = reshape(pay2,mbs, number_outputs,[]);
        pay2 = permute(pay2,[2,3,1]);
        pay1 = sum(learnable_parameters.singleton.c.* lower_firing_strength,1);

        pay = pay1 + pay2;

        %         clear pay1_upper pay2_upper
        %         clear delta_f u

        payda2 = permute(payda2,[3,2,1]);
        payda1 = sum(lower_firing_strength,1);

        payda = payda1 + payda2;

        %         clear payda1 payda2

        pay = permute(pay,[2 1 3]);

        output = pay./payda;

        %         clear pay_lower pay_upper payda

        output_lower = permute(min(output,[],2),[2 1 3]);
        output_upper = permute(max(output,[],2),[2 1 3]);

        %         clear output_lower_temp output_upper_temp

        output_mean = (output_lower + output_upper)./2;

    

    elseif type_reduction_method == "KM_BMM"


        delta_f = upper_firing_strength - lower_firing_strength;
        delta_f = permute(delta_f,[3 1 2]);

        payda2 = delta_f*u;

%         pay2 = reshape((permute(learnable_parameters.singleton.c,[3 2 1]).*permute(delta_f,[1 3 2])),[mbs*number_outputs,number_mf])*u;
        pay2 = pagemtimes((permute(learnable_parameters.singleton.c,[3 1 2]).*delta_f),u);

%         pay2 = reshape(pay2,mbs, number_outputs,[]);
        pay2 = permute(pay2,[2,3,1]);
        pay1 = sum(learnable_parameters.singleton.c.* lower_firing_strength,1);

        pay = pay1 + pay2;

        %         clear pay1_upper pay2_upper
        %         clear delta_f u

        payda2 = permute(payda2,[3,2,1]);
        payda1 = sum(lower_firing_strength,1);

        payda = payda1 + payda2;

        %         clear payda1 payda2

        pay = permute(pay,[2 1 3]);

        output = pay./payda;

        %         clear pay_lower pay_upper payda

        output_lower = permute(min(output,[],2),[2 1 3]);
        output_upper = permute(max(output,[],2),[2 1 3]);

        %         clear output_lower_temp output_upper_temp

        alpha = sigmoid(learnable_parameters.singleton.alpha);

        output_mean = (output_lower .* alpha) + (output_upper .* (1 - alpha));


    elseif type_reduction_method == "CQTR" || type_reduction_method == "CQTR_multi_alpha" 

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);

        alpha = sigmoid(learnable_parameters.singleton.alpha);

        numerator_lower_1 = sum((lower_firing_strength_temp.* learnable_parameters.singleton.c) .* alpha,1);
        numerator_upper_1 = sum((upper_firing_strength_temp.* learnable_parameters.singleton.c) .* (1 - alpha),1);

        numerator_lower_2 = sum(((lower_firing_strength_temp.^2).* learnable_parameters.singleton.c) .* (1 - alpha),1);
        numerator_upper_2 = sum(((upper_firing_strength_temp.^2).* learnable_parameters.singleton.c) .* (1 - alpha),1);

        numerator_3 = sum(((lower_firing_strength_temp .* upper_firing_strength_temp).* learnable_parameters.singleton.c) .* (1 - alpha),1);
        
        numerator_lower = 2.*(numerator_lower_1 + numerator_lower_2 - numerator_3);
        numerator_upper = 2.*(numerator_upper_1 + numerator_upper_2 - numerator_3);

        denominator_1 = sum(lower_firing_strength_temp.* alpha,1);
        denominator_2 = sum(upper_firing_strength_temp.* (1 - alpha),1);
        denominator_3 = sum((lower_firing_strength_temp.^2).* (1 - alpha),1);
        denominator_4 = sum((upper_firing_strength_temp.^2).* (1 - alpha),1);
        denominator_5 = sum((lower_firing_strength_temp .* upper_firing_strength_temp).* (1 - alpha),1);

        denominator = denominator_1 + denominator_2 + denominator_3 + denominator_4 - (2 .* denominator_5);
  
        output_lower = numerator_lower./denominator;
        output_upper = numerator_upper./denominator;

        output_mean = (output_lower + output_upper)./2;

    end


elseif output_type == "linear"


    if type_reduction_method == "SM"

        normalized_lower_firing_strength = lower_firing_strength./sum(lower_firing_strength, 1);
        normalized_upper_firing_strength = upper_firing_strength./sum(upper_firing_strength, 1);

        temp_mf = [learnable_parameters.linear.a,learnable_parameters.linear.b];
        x = permute(x,[2 1 3]);
        temp_input = [x; ones(1, size(x, 2), size(x, 3))];
        temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation


        c = temp_mf*temp_input;
        c= reshape(c, [size(normalized_lower_firing_strength, 1), number_outputs, size(normalized_lower_firing_strength, 3)]);


        normalized_lower_firing_strength = repmat(normalized_lower_firing_strength,1,number_outputs);
        normalized_upper_firing_strength = repmat(normalized_upper_firing_strength,1,number_outputs);


        output_lower = normalized_lower_firing_strength.* c;
        output_upper = normalized_upper_firing_strength.* c;

        output_lower = sum(output_lower, 1); %then we sum with respect to dimension 1
        output_upper = sum(output_upper, 1); %then we sum with respect to dimension 1

        output_mean = (output_lower + output_upper)./2;

        

    elseif type_reduction_method == "BMM"

        normalized_lower_firing_strength = lower_firing_strength./sum(lower_firing_strength, 1);
        normalized_upper_firing_strength = upper_firing_strength./sum(upper_firing_strength, 1);

        temp_mf = [learnable_parameters.linear.a,learnable_parameters.linear.b];
        x = permute(x,[2 1 3]);
        temp_input = [x; ones(1, size(x, 2), size(x, 3))];
        temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation


        c = temp_mf*temp_input;
        c= reshape(c, [size(normalized_lower_firing_strength, 1), number_outputs, size(normalized_lower_firing_strength, 3)]);


        normalized_lower_firing_strength = repmat(normalized_lower_firing_strength,1,number_outputs);
        normalized_upper_firing_strength = repmat(normalized_upper_firing_strength,1,number_outputs);


        output_lower = normalized_lower_firing_strength.* c;
        output_upper = normalized_upper_firing_strength.* c;

        output_lower = sum(output_lower, 1); %then we sum with respect to dimension 1
        output_upper = sum(output_upper, 1); %then we sum with respect to dimension 1


        alpha = sigmoid(learnable_parameters.linear.alpha);

        output_mean = (output_lower .* alpha) + (output_upper .* (1 - alpha));


    elseif type_reduction_method == "NT"

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);

        temp_mf = [learnable_parameters.linear.a,learnable_parameters.linear.b];
        x = permute(x,[2 1 3]);
        temp_input = [x; ones(1, size(x, 2), size(x, 3))];
        temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation

        c = temp_mf*temp_input;
        c = reshape(c, [size(lower_firing_strength, 1), number_outputs, size(lower_firing_strength, 3)]);

        numerator_lower = lower_firing_strength_temp.* c;
        numerator_upper = upper_firing_strength_temp.* c;

        numerator_lower = 2*sum(numerator_lower, 1);
        numerator_upper = 2*sum(numerator_upper, 1);

        denominator = sum(lower_firing_strength, 1) + sum(upper_firing_strength, 1);

        output_lower = numerator_lower./denominator;
        output_upper = numerator_upper./denominator;

        output_mean = (output_lower + output_upper)./2;


    elseif type_reduction_method == "NT_alpha"  || type_reduction_method == "NT_multi_alpha" 

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);

        temp_mf = [learnable_parameters.linear.a,learnable_parameters.linear.b];
        x = permute(x,[2 1 3]);
        temp_input = [x; ones(1, size(x, 2), size(x, 3))];
        temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation

        c = temp_mf*temp_input;
        c = reshape(c, [size(lower_firing_strength, 1), number_outputs, size(lower_firing_strength, 3)]);

%         alpha = sigmoid(learnable_parameters.linear.alpha)^2;
        alpha = sigmoid(learnable_parameters.linear.alpha);


        numerator_lower = (lower_firing_strength_temp.* c).* alpha;
        numerator_upper = (upper_firing_strength_temp.* c).* (1 - alpha);

        numerator_lower = 2*sum(numerator_lower, 1);
        numerator_upper = 2*sum(numerator_upper, 1);

        denominator = (sum(lower_firing_strength .* alpha, 1) ) + (sum(upper_firing_strength .* (1 - alpha), 1) );

        output_lower = numerator_lower./denominator;
        output_upper = numerator_upper./denominator;

        output_mean = (output_lower + output_upper)./2;

    
    elseif type_reduction_method == "NT_alpha2"

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);

        temp_mf = [learnable_parameters.linear.a,learnable_parameters.linear.b];
        x = permute(x,[2 1 3]);
        temp_input = [x; ones(1, size(x, 2), size(x, 3))];
        temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation

        c = temp_mf*temp_input;
        c = reshape(c, [size(lower_firing_strength, 1), number_outputs, size(lower_firing_strength, 3)]);

        alpha = sigmoid(learnable_parameters.linear.alpha).^2;

        numerator_lower = (lower_firing_strength_temp.* c).* alpha;
        numerator_upper = (upper_firing_strength_temp.* c).* (1 - alpha);

        numerator_lower = 2*sum(numerator_lower, 1);
        numerator_upper = 2*sum(numerator_upper, 1);

        denominator = (sum(lower_firing_strength .* alpha, 1) ) + (sum(upper_firing_strength .* (1 - alpha), 1) );

        output_lower = numerator_lower./denominator;
        output_upper = numerator_upper./denominator;

        output_mean = (output_lower + output_upper)./2;

    elseif type_reduction_method == "NTv2" || type_reduction_method == "NTv2_multi_alpha" || type_reduction_method == "NTv3" || type_reduction_method == "NTv3_multi_alpha"

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);

        temp_mf = [learnable_parameters.linear.a,learnable_parameters.linear.b];
        x = permute(x,[2 1 3]);
        temp_input = [x; ones(1, size(x, 2), size(x, 3))];
        temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation

        c = temp_mf*temp_input;
        c = reshape(c, [size(lower_firing_strength, 1), number_outputs, size(lower_firing_strength, 3)]);

        alpha = sigmoid(learnable_parameters.linear.alpha);
        beta = sigmoid(learnable_parameters.linear.beta);

        numerator_lower = sum((lower_firing_strength_temp.* c) .* alpha,1) + sum((upper_firing_strength_temp.* c) .* (1 - alpha),1);
        numerator_upper = sum((lower_firing_strength_temp.* c) .* (1 - beta),1) + sum((upper_firing_strength_temp.* c) .* beta,1);

        denominator_lower = sum(lower_firing_strength .* alpha, 1) + sum(upper_firing_strength .* (1 - alpha), 1);
        denominator_upper = sum(lower_firing_strength .* (1 - beta), 1) + sum(upper_firing_strength .* beta, 1);

        output_lower = numerator_lower./denominator_lower;
        output_upper = numerator_upper./denominator_upper;

        output_mean = (output_lower + output_upper)./2;


    elseif type_reduction_method == "NTv4" 

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);

        temp_mf = [learnable_parameters.linear.a,learnable_parameters.linear.b];
        x = permute(x,[2 1 3]);
        temp_input = [x; ones(1, size(x, 2), size(x, 3))];
        temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation

        c = temp_mf*temp_input;
        c = reshape(c, [size(lower_firing_strength, 1), number_outputs, size(lower_firing_strength, 3)]);

        alpha = sigmoid(10*learnable_parameters.linear.alpha);
        beta = sigmoid(10*learnable_parameters.linear.beta);

        numerator_lower = sum((lower_firing_strength_temp.* c) .* alpha,1) + sum((upper_firing_strength_temp.* c) .* (1 - alpha),1);
        numerator_upper = sum((lower_firing_strength_temp.* c) .* beta,1) + sum((upper_firing_strength_temp.* c) .* (1 - beta),1);

        denominator_lower = sum(lower_firing_strength .* alpha, 1) + sum(upper_firing_strength .* (1 - alpha), 1);
        denominator_upper = sum(lower_firing_strength .* beta, 1) + sum(upper_firing_strength .* (1 - beta), 1);

        output_lower = numerator_lower./denominator_lower;
        output_upper = numerator_upper./denominator_upper;

        output_mean = (output_lower + output_upper)./2;


    elseif type_reduction_method == "NTv2_alpha2" || type_reduction_method == "NTv3_alpha2"

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);

        temp_mf = [learnable_parameters.linear.a,learnable_parameters.linear.b];
        x = permute(x,[2 1 3]);
        temp_input = [x; ones(1, size(x, 2), size(x, 3))];
        temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation

        c = temp_mf*temp_input;
        c = reshape(c, [size(lower_firing_strength, 1), number_outputs, size(lower_firing_strength, 3)]);

        alpha = sigmoid(learnable_parameters.linear.alpha).^2;
        beta = sigmoid(learnable_parameters.linear.beta).^2;

        numerator_lower = sum((lower_firing_strength_temp.* c) .* alpha,1) + sum((upper_firing_strength_temp.* c) .* (1 - alpha),1);
        numerator_upper = sum((lower_firing_strength_temp.* c) .* (1 - beta),1) + sum((upper_firing_strength_temp.* c) .* beta,1);

        denominator_lower = sum(lower_firing_strength .* alpha, 1) + sum(upper_firing_strength .* (1 - alpha), 1);
        denominator_upper = sum(lower_firing_strength .* (1 - beta), 1) + sum(upper_firing_strength .* beta, 1);

        output_lower = numerator_lower./denominator_lower;
        output_upper = numerator_upper./denominator_upper;

        output_mean = (output_lower + output_upper)./2;


    elseif type_reduction_method == "KM"

        temp_mf = [learnable_parameters.linear.a,learnable_parameters.linear.b];
        x = permute(x,[2 1 3]);
        temp_input = [x; ones(1, size(x, 2), size(x, 3))];
        temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation

        c = temp_mf*temp_input;
        c = reshape(c, [size(lower_firing_strength, 1), number_outputs, size(lower_firing_strength, 3)]);

        delta_f = upper_firing_strength - lower_firing_strength;
        delta_f = permute(delta_f,[3 1 2]);

        payda2 = delta_f*u;

        %         pay2 = pagemtimes((permute(c,[3 1 2]).*repmat(delta_f,1,1,number_outputs)),u);
        pay2 = pagemtimes((permute(c,[3 1 2]).*delta_f),u);

        pay2 = permute(pay2,[3,2,1]);
        pay1 = permute(sum(c .* lower_firing_strength,1),[2 1 3]);

        pay = pay1 + pay2;

        %         clear pay1_upper pay2_upper
        %         clear delta_f u

        payda2 = permute(payda2,[3,2,1]);
        payda1 = sum(lower_firing_strength,1);

        payda = payda1 + payda2;

        %         clear payda1 payda2

        output = pay./payda;

        %         clear pay_lower pay_upper payda

        output_lower = permute(min(output,[],2),[2 1 3]);
        output_upper = permute(max(output,[],2),[2 1 3]);

        %         clear output_lower_temp output_upper_temp

        output_mean = (output_lower + output_upper)./2;

    elseif type_reduction_method == "KM_BMM"

        temp_mf = [learnable_parameters.linear.a,learnable_parameters.linear.b];
        x = permute(x,[2 1 3]);
        temp_input = [x; ones(1, size(x, 2), size(x, 3))];
        temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation

        c = temp_mf*temp_input;
        c = reshape(c, [size(lower_firing_strength, 1), number_outputs, size(lower_firing_strength, 3)]);

        delta_f = upper_firing_strength - lower_firing_strength;
        delta_f = permute(delta_f,[3 1 2]);

        payda2 = delta_f*u;

        %         pay2 = pagemtimes((permute(c,[3 1 2]).*repmat(delta_f,1,1,number_outputs)),u);
        pay2 = pagemtimes((permute(c,[3 1 2]).*delta_f),u);

        pay2 = permute(pay2,[3,2,1]);
        pay1 = permute(sum(c .* lower_firing_strength,1),[2 1 3]);

        pay = pay1 + pay2;

        %         clear pay1_upper pay2_upper
        %         clear delta_f u

        payda2 = permute(payda2,[3,2,1]);
        payda1 = sum(lower_firing_strength,1);

        payda = payda1 + payda2;

        %         clear payda1 payda2

        output = pay./payda;

        %         clear pay_lower pay_upper payda

        output_lower = permute(min(output,[],2),[2 1 3]);
        output_upper = permute(max(output,[],2),[2 1 3]);

        %         clear output_lower_temp output_upper_temp

        alpha = sigmoid(learnable_parameters.linear.alpha);


        output_mean = (output_lower .* alpha) + (output_upper .* (1 - alpha));

    elseif type_reduction_method == "CQTR" || type_reduction_method == "CQTR_multi_alpha" 

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);

        temp_mf = [learnable_parameters.linear.a,learnable_parameters.linear.b];
        x = permute(x,[2 1 3]);
        temp_input = [x; ones(1, size(x, 2), size(x, 3))];
        temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation

        c = temp_mf*temp_input;
        c = reshape(c, [size(lower_firing_strength, 1), number_outputs, size(lower_firing_strength, 3)]);

        alpha = sigmoid(learnable_parameters.linear.alpha);

        numerator_lower_1 = sum((lower_firing_strength_temp.* c) .* alpha,1);
        numerator_upper_1 = sum((upper_firing_strength_temp.* c) .* (1 - alpha),1);

        numerator_lower_2 = sum(((lower_firing_strength_temp.^2).* c) .* (1 - alpha),1);
        numerator_upper_2 = sum(((upper_firing_strength_temp.^2).* c) .* (1 - alpha),1);

        numerator_3 = sum(((lower_firing_strength_temp .* upper_firing_strength_temp).* c) .* (1 - alpha),1);
        
        numerator_lower = 2.*(numerator_lower_1 + numerator_lower_2 - numerator_3);
        numerator_upper = 2.*(numerator_upper_1 + numerator_upper_2 - numerator_3);

        denominator_1 = sum(lower_firing_strength_temp.* alpha,1);
        denominator_2 = sum(upper_firing_strength_temp.* (1 - alpha),1);
        denominator_3 = sum((lower_firing_strength_temp.^2).* (1 - alpha),1);
        denominator_4 = sum((upper_firing_strength_temp.^2).* (1 - alpha),1);
        denominator_5 = sum((lower_firing_strength_temp .* upper_firing_strength_temp).* (1 - alpha),1);

        denominator = denominator_1 + denominator_2 + denominator_3 + denominator_4 - (2 .* denominator_5);
  
        output_lower = numerator_lower./denominator;
        output_upper = numerator_upper./denominator;

        output_mean = (output_lower + output_upper)./2;
        %test

    end


elseif output_type == "IV"

    if type_reduction_method == "SM"

        normalized_lower_firing_strength = lower_firing_strength./sum(lower_firing_strength, 1);
        normalized_upper_firing_strength = upper_firing_strength./sum(upper_firing_strength, 1);


        c_upper = learnable_parameters.IV.c + abs(learnable_parameters.IV.delta);
        c_lower = learnable_parameters.IV.c - abs(learnable_parameters.IV.delta);


        normalized_lower_firing_strength = repmat(normalized_lower_firing_strength,1,number_outputs);
        normalized_upper_firing_strength = repmat(normalized_upper_firing_strength,1,number_outputs);


        output_lower = normalized_lower_firing_strength.* c_lower;
        output_upper = normalized_upper_firing_strength.* c_upper;

        output_lower = sum(output_lower, 1);
        output_upper = sum(output_upper, 1);

        output_mean = (output_lower + output_upper)./2;

    elseif type_reduction_method == "BMM"

        normalized_lower_firing_strength = lower_firing_strength./sum(lower_firing_strength, 1);
        normalized_upper_firing_strength = upper_firing_strength./sum(upper_firing_strength, 1);

% 
        c_upper = learnable_parameters.IV.c + abs(learnable_parameters.IV.delta);
        c_lower = learnable_parameters.IV.c - abs(learnable_parameters.IV.delta);


        % c_upper = learnable_parameters.IV.c + abs(learnable_parameters.IV.delta_1);
        % c_lower = learnable_parameters.IV.c - abs(learnable_parameters.IV.delta_2);


        normalized_lower_firing_strength = repmat(normalized_lower_firing_strength,1,number_outputs);
        normalized_upper_firing_strength = repmat(normalized_upper_firing_strength,1,number_outputs);


        output_lower = normalized_lower_firing_strength.* c_lower;
        output_upper = normalized_upper_firing_strength.* c_upper;

        output_lower = sum(output_lower, 1);
        output_upper = sum(output_upper, 1);

        alpha = sigmoid(learnable_parameters.IV.alpha);

        output_mean = (output_lower .* alpha) + (output_upper .* (1 - alpha));



    elseif type_reduction_method == "NT"

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);

        c_lower = learnable_parameters.IV.c - abs(learnable_parameters.IV.delta);
        c_upper = learnable_parameters.IV.c + abs(learnable_parameters.IV.delta);
        

        numerator_lower = lower_firing_strength_temp.* c_lower;
        numerator_upper = upper_firing_strength_temp.* c_upper;

        numerator_lower = 2*sum(numerator_lower, 1);
        numerator_upper = 2*sum(numerator_upper, 1);

        denominator = sum(lower_firing_strength, 1) + sum(upper_firing_strength, 1);


        output_lower = numerator_lower./denominator;
        output_upper = numerator_upper./denominator;

        output_mean = (output_lower + output_upper)./2;

    elseif type_reduction_method == "NT_alpha"  || type_reduction_method == "NT_multi_alpha"

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);


        c_lower = learnable_parameters.IV.c - abs(learnable_parameters.IV.delta);
        c_upper = learnable_parameters.IV.c + abs(learnable_parameters.IV.delta);
        

        %         alpha = sigmoid(learnable_parameters.linear.alpha)^2;
        alpha = sigmoid(learnable_parameters.IV.alpha);

        numerator_lower = (lower_firing_strength_temp.* c_lower) .* alpha;
        numerator_upper = (upper_firing_strength_temp.* c_upper) .* (1 - alpha);

        numerator_lower = 2*sum(numerator_lower, 1);
        numerator_upper = 2*sum(numerator_upper, 1);

        denominator = sum(lower_firing_strength .* alpha, 1) + sum(upper_firing_strength .* (1 - alpha), 1);


        output_lower = numerator_lower./denominator;
        output_upper = numerator_upper./denominator;

        output_mean = (output_lower + output_upper)./2;

    elseif type_reduction_method == "NT_alpha2"

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);


        c_lower = learnable_parameters.IV.c - abs(learnable_parameters.IV.delta);
        c_upper = learnable_parameters.IV.c + abs(learnable_parameters.IV.delta);
        
        alpha = sigmoid(learnable_parameters.IV.alpha).^2;

        numerator_lower = (lower_firing_strength_temp.* c_lower) .* alpha;
        numerator_upper = (upper_firing_strength_temp.* c_upper) .* (1 - alpha);

        numerator_lower = 2*sum(numerator_lower, 1);
        numerator_upper = 2*sum(numerator_upper, 1);

        denominator = sum(lower_firing_strength .* alpha, 1) + sum(upper_firing_strength .* (1 - alpha), 1);


        output_lower = numerator_lower./denominator;
        output_upper = numerator_upper./denominator;

        output_mean = (output_lower + output_upper)./2;


    elseif type_reduction_method == "NTv2" || type_reduction_method == "NTv2_multi_alpha"

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);


        c_upper = learnable_parameters.IV.c + abs(learnable_parameters.IV.delta);
        c_lower = learnable_parameters.IV.c - abs(learnable_parameters.IV.delta);

        alpha = sigmoid(learnable_parameters.IV.alpha);
        beta = sigmoid(learnable_parameters.IV.beta);

        numerator_lower = sum((lower_firing_strength_temp.* c_lower) .* alpha,1) + sum((upper_firing_strength_temp.* c_upper) .* (1 - alpha),1);
        numerator_upper = sum((lower_firing_strength_temp.* c_lower) .* (1 - beta),1) + sum((upper_firing_strength_temp.* c_upper) .* beta,1);

        denominator_lower = sum(lower_firing_strength .* alpha, 1) + sum(upper_firing_strength .* (1 - alpha), 1);
        denominator_upper = sum(lower_firing_strength .* (1 - beta), 1) + sum(upper_firing_strength .* beta, 1);


        output_lower = numerator_lower./denominator_lower;
        output_upper = numerator_upper./denominator_upper;

        output_mean = (output_lower + output_upper)./2;


    elseif type_reduction_method == "NTv2_alpha2" 

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);


        c_upper = learnable_parameters.IV.c + abs(learnable_parameters.IV.delta);
        c_lower = learnable_parameters.IV.c - abs(learnable_parameters.IV.delta);

        alpha = sigmoid(learnable_parameters.IV.alpha).^2;
        beta = sigmoid(learnable_parameters.IV.beta).^2;

        numerator_lower = sum((lower_firing_strength_temp.* c_lower) .* alpha,1) + sum((upper_firing_strength_temp.* c_upper) .* (1 - alpha),1);
        numerator_upper = sum((lower_firing_strength_temp.* c_lower) .* (1 - beta),1) + sum((upper_firing_strength_temp.* c_upper) .* beta,1);

        denominator_lower = sum(lower_firing_strength .* alpha, 1) + sum(upper_firing_strength .* (1 - alpha), 1);
        denominator_upper = sum(lower_firing_strength .* (1 - beta), 1) + sum(upper_firing_strength .* beta, 1);


        output_lower = numerator_lower./denominator_lower;
        output_upper = numerator_upper./denominator_upper;

        output_mean = (output_lower + output_upper)./2;

    elseif type_reduction_method == "NTv3" || type_reduction_method == "NTv3_multi_alpha"

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);


        c_upper = learnable_parameters.IV.c + abs(learnable_parameters.IV.delta);
        c_lower = learnable_parameters.IV.c - abs(learnable_parameters.IV.delta);

        alpha = sigmoid(learnable_parameters.IV.alpha);
        beta = sigmoid(learnable_parameters.IV.beta);

        numerator_lower = sum((lower_firing_strength_temp.* c_lower) .* alpha,1) + sum((upper_firing_strength_temp .* c_lower) .* (1 - alpha),1);
        numerator_upper = sum((lower_firing_strength_temp.* c_upper) .* (1 - beta),1) + sum((upper_firing_strength_temp .* c_upper) .* beta,1);

        denominator_lower = sum(lower_firing_strength .* alpha, 1) + sum(upper_firing_strength .* (1 - alpha), 1);
        denominator_upper = sum(lower_firing_strength .* (1 - beta), 1) + sum(upper_firing_strength .* beta, 1);


        output_lower = numerator_lower./denominator_lower;
        output_upper = numerator_upper./denominator_upper;

        output_mean = (output_lower + output_upper)./2;


    elseif type_reduction_method == "NTv3_alpha2" 

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);


        c_upper = learnable_parameters.IV.c + abs(learnable_parameters.IV.delta);
        c_lower = learnable_parameters.IV.c - abs(learnable_parameters.IV.delta);

        alpha = sigmoid(learnable_parameters.IV.alpha).^2;
        beta = sigmoid(learnable_parameters.IV.beta).^2;

        numerator_lower = sum((lower_firing_strength_temp.* c_lower) .* alpha,1) + sum((upper_firing_strength_temp.* c_lower) .* (1 - alpha),1);
        numerator_upper = sum((lower_firing_strength_temp.* c_upper) .* (1 - beta),1) + sum((upper_firing_strength_temp.* c_upper) .* beta,1);

        denominator_lower = sum(lower_firing_strength .* alpha, 1) + sum(upper_firing_strength .* (1 - alpha), 1);
        denominator_upper = sum(lower_firing_strength .* (1 - beta), 1) + sum(upper_firing_strength .* beta, 1);


        output_lower = numerator_lower./denominator_lower;
        output_upper = numerator_upper./denominator_upper;

        output_mean = (output_lower + output_upper)./2;

    elseif type_reduction_method == "NTv4"

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);


        c_upper = learnable_parameters.IV.c + abs(learnable_parameters.IV.delta);
        c_lower = learnable_parameters.IV.c - abs(learnable_parameters.IV.delta);

        alpha = sigmoid(10*learnable_parameters.IV.alpha);
        beta = sigmoid(10*learnable_parameters.IV.beta);

        numerator_lower = sum((lower_firing_strength_temp.* c_lower) .* alpha,1) + sum((upper_firing_strength_temp.* c_lower) .* (1 - alpha),1);
        numerator_upper = sum((lower_firing_strength_temp.* c_upper) .* beta,1) + sum((upper_firing_strength_temp.* c_upper) .* (1 - beta),1);

        denominator_lower = sum(lower_firing_strength .* alpha, 1) + sum(upper_firing_strength .* (1 - alpha), 1);
        denominator_upper = sum(lower_firing_strength .* beta, 1) + sum(upper_firing_strength .* (1 - beta), 1);

        output_lower = numerator_lower./denominator_lower;
        output_upper = numerator_upper./denominator_upper;

        output_mean = (output_lower + output_upper)./2;


    elseif type_reduction_method == "KM"


        c_upper = learnable_parameters.IV.c + abs(learnable_parameters.IV.delta);
        c_lower = learnable_parameters.IV.c - abs(learnable_parameters.IV.delta);


        delta_f = upper_firing_strength - lower_firing_strength;
        delta_f = permute(delta_f,[3 1 2]);

        payda2 = delta_f*u;

        pay2_lower = pagemtimes((permute(c_lower,[3 1 2]).*delta_f),u);
        pay2_lower = permute(pay2_lower,[2,3,1]);
        pay1_lower = sum(c_lower .* lower_firing_strength,1);

        pay_lower = pay1_lower + pay2_lower;



        pay2_upper = pagemtimes((permute(c_upper,[3 1 2]).*delta_f),u);
        pay2_upper = permute(pay2_upper,[2,3,1]);
        pay1_upper = sum(c_upper .* lower_firing_strength,1);

        pay_upper = pay1_upper + pay2_upper;


        %         clear pay1_upper pay2_upper
        %         clear delta_f u


        payda2 = permute(payda2,[3,2,1]);
        payda1 = sum(lower_firing_strength,1);

        payda = payda1 + payda2;

        %         clear payda1 payda2


        pay_lower = permute(pay_lower,[2 1 3]);
        pay_upper = permute(pay_upper,[2 1 3]);

        output_lower = pay_lower./payda;
        output_upper = pay_upper./payda;

        %         clear pay_lower pay_upper payda

        output_lower = permute(min(output_lower,[],2),[2 1 3]);
        output_upper = permute(max(output_upper,[],2),[2 1 3]);

        %         clear output_lower_temp output_upper_temp

        output_mean = (output_lower + output_upper)./2;


    elseif type_reduction_method == "KM_BMM"


        c_upper = learnable_parameters.IV.c + abs(learnable_parameters.IV.delta);
        c_lower = learnable_parameters.IV.c - abs(learnable_parameters.IV.delta);


        delta_f = upper_firing_strength - lower_firing_strength;
        delta_f = permute(delta_f,[3 1 2]);

        payda2 = delta_f*u;

        pay2_lower = pagemtimes((permute(c_lower,[3 1 2]).*delta_f),u);
        pay2_lower = permute(pay2_lower,[2,3,1]);
        pay1_lower = sum(c_lower .* lower_firing_strength,1);

        pay_lower = pay1_lower + pay2_lower;



        pay2_upper = pagemtimes((permute(c_upper,[3 1 2]).*delta_f),u);
        pay2_upper = permute(pay2_upper,[2,3,1]);
        pay1_upper = sum(c_upper .* lower_firing_strength,1);

        pay_upper = pay1_upper + pay2_upper;


        %         clear pay1_upper pay2_upper
        %         clear delta_f u


        payda2 = permute(payda2,[3,2,1]);
        payda1 = sum(lower_firing_strength,1);

        payda = payda1 + payda2;

        %         clear payda1 payda2


        pay_lower = permute(pay_lower,[2 1 3]);
        pay_upper = permute(pay_upper,[2 1 3]);

        output_lower = pay_lower./payda;
        output_upper = pay_upper./payda;

        %         clear pay_lower pay_upper payda

        output_lower = permute(min(output_lower,[],2),[2 1 3]);
        output_upper = permute(max(output_upper,[],2),[2 1 3]);

        %         clear output_lower_temp output_upper_temp

        alpha = sigmoid(learnable_parameters.IV.alpha);

        output_mean = (output_lower .* alpha) + (output_upper .* (1 - alpha));

    end



elseif output_type == "IVL"


    if type_reduction_method == "SM"

        normalized_lower_firing_strength = lower_firing_strength./sum(lower_firing_strength, 1);
        normalized_upper_firing_strength = upper_firing_strength./sum(upper_firing_strength, 1);


        temp_mf = [learnable_parameters.IVL.a,learnable_parameters.IVL.b];
        temp_delta = [learnable_parameters.IVL.delta_a,learnable_parameters.IVL.delta_b];


        x = permute(x,[2 1 3]); %comment at
        temp_input = [x; ones(1, size(x, 2), size(x, 3))];
        temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation


        linear_lower = (temp_mf * temp_input) - (abs(temp_delta * temp_input));
        linear_upper = (temp_mf * temp_input) + (abs(temp_delta * temp_input));

        linear_lower = reshape(linear_lower, [size(normalized_lower_firing_strength, 1), number_outputs, size(normalized_lower_firing_strength, 3)]);
        linear_upper = reshape(linear_upper, [size(normalized_upper_firing_strength, 1), number_outputs, size(normalized_upper_firing_strength, 3)]);

        normalized_lower_firing_strength = repmat(normalized_lower_firing_strength,1,number_outputs);
        normalized_upper_firing_strength = repmat(normalized_upper_firing_strength,1,number_outputs);

        output_lower = normalized_lower_firing_strength.* linear_lower;
        output_upper = normalized_upper_firing_strength.* linear_upper;

        output_lower = sum(output_lower, 1);
        output_upper = sum(output_upper, 1);

        output_mean = (output_lower + output_upper)./2;

    elseif type_reduction_method == "BMM"

        normalized_lower_firing_strength = lower_firing_strength./sum(lower_firing_strength, 1);
        normalized_upper_firing_strength = upper_firing_strength./sum(upper_firing_strength, 1);


        temp_mf = [learnable_parameters.IVL.a,learnable_parameters.IVL.b];
        temp_delta = [learnable_parameters.IVL.delta_a,learnable_parameters.IVL.delta_b];


        x = permute(x,[2 1 3]); %comment at
        temp_input = [x; ones(1, size(x, 2), size(x, 3))];
        temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation


        linear_lower = (temp_mf * temp_input) - (abs(temp_delta * temp_input));
        linear_upper = (temp_mf * temp_input) + (abs(temp_delta * temp_input));

        linear_lower = reshape(linear_lower, [size(normalized_lower_firing_strength, 1), number_outputs, size(normalized_lower_firing_strength, 3)]);
        linear_upper = reshape(linear_upper, [size(normalized_upper_firing_strength, 1), number_outputs, size(normalized_upper_firing_strength, 3)]);

        normalized_lower_firing_strength = repmat(normalized_lower_firing_strength,1,number_outputs);
        normalized_upper_firing_strength = repmat(normalized_upper_firing_strength,1,number_outputs);

        output_lower = normalized_lower_firing_strength.* linear_lower;
        output_upper = normalized_upper_firing_strength.* linear_upper;

        output_lower = sum(output_lower, 1);
        output_upper = sum(output_upper, 1);

        alpha = sigmoid(learnable_parameters.IVL.alpha);

        output_mean = (output_lower .* alpha) + (output_upper .* (1 - alpha));

    elseif type_reduction_method == "NT"

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);

        temp_mf = [learnable_parameters.IVL.a,learnable_parameters.IVL.b];
        temp_delta = [learnable_parameters.IVL.delta_a,learnable_parameters.IVL.delta_b];

        x = permute(x,[2 1 3]); %comment at
        temp_input = [x; ones(1, size(x, 2), size(x, 3))];
        temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation


        linear_lower = (temp_mf * temp_input) - (abs(temp_delta * temp_input));
        linear_upper = (temp_mf * temp_input) + (abs(temp_delta * temp_input));

        linear_lower = reshape(linear_lower, [size(lower_firing_strength_temp, 1), number_outputs, size(lower_firing_strength_temp, 3)]);
        linear_upper = reshape(linear_upper, [size(upper_firing_strength_temp, 1), number_outputs, size(upper_firing_strength_temp, 3)]);

        numerator_lower = lower_firing_strength_temp.* linear_lower;
        numerator_upper = upper_firing_strength_temp.* linear_upper;

        numerator_lower = 2*sum(numerator_lower, 1);
        numerator_upper = 2*sum(numerator_upper, 1);

        denominator = sum(lower_firing_strength, 1) + sum(upper_firing_strength, 1);

        output_lower = numerator_lower./denominator;
        output_upper = numerator_upper./denominator;

        output_mean = (output_lower + output_upper)./2;

    elseif type_reduction_method == "NT_alpha" || type_reduction_method == "NT_multi_alpha"

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);

        temp_mf = [learnable_parameters.IVL.a,learnable_parameters.IVL.b];
        temp_delta = [learnable_parameters.IVL.delta_a,learnable_parameters.IVL.delta_b];

        x = permute(x,[2 1 3]); %comment at
        temp_input = [x; ones(1, size(x, 2), size(x, 3))];
        temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation


        linear_lower = (temp_mf * temp_input) - (abs(temp_delta * temp_input));
        linear_upper = (temp_mf * temp_input) + (abs(temp_delta * temp_input));

        linear_lower = reshape(linear_lower, [size(lower_firing_strength_temp, 1), number_outputs, size(lower_firing_strength_temp, 3)]);
        linear_upper = reshape(linear_upper, [size(upper_firing_strength_temp, 1), number_outputs, size(upper_firing_strength_temp, 3)]);

        %         alpha = sigmoid(learnable_parameters.linear.alpha)^2;
        alpha = sigmoid(learnable_parameters.IVL.alpha);

        numerator_lower = (lower_firing_strength_temp.* linear_lower) .* alpha;
        numerator_upper = (upper_firing_strength_temp.* linear_upper).* (1 - alpha);

        numerator_lower = 2*sum(numerator_lower, 1);
        numerator_upper = 2*sum(numerator_upper, 1);

        denominator = sum(lower_firing_strength .* alpha, 1) + sum(upper_firing_strength .* (1 - alpha), 1);

        output_lower = numerator_lower./denominator;
        output_upper = numerator_upper./denominator;

        output_mean = (output_lower + output_upper)./2;

    elseif type_reduction_method == "NT_alpha2"

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);

        temp_mf = [learnable_parameters.IVL.a,learnable_parameters.IVL.b];
        temp_delta = [learnable_parameters.IVL.delta_a,learnable_parameters.IVL.delta_b];

        x = permute(x,[2 1 3]); %comment at
        temp_input = [x; ones(1, size(x, 2), size(x, 3))];
        temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation


        linear_lower = (temp_mf * temp_input) - (abs(temp_delta * temp_input));
        linear_upper = (temp_mf * temp_input) + (abs(temp_delta * temp_input));

        linear_lower = reshape(linear_lower, [size(lower_firing_strength_temp, 1), number_outputs, size(lower_firing_strength_temp, 3)]);
        linear_upper = reshape(linear_upper, [size(upper_firing_strength_temp, 1), number_outputs, size(upper_firing_strength_temp, 3)]);

        alpha = sigmoid(learnable_parameters.IVL.alpha).^2;

        numerator_lower = (lower_firing_strength_temp.* linear_lower) .* alpha;
        numerator_upper = (upper_firing_strength_temp.* linear_upper).* (1 - alpha);

        numerator_lower = 2*sum(numerator_lower, 1);
        numerator_upper = 2*sum(numerator_upper, 1);

        denominator = sum(lower_firing_strength .* alpha, 1) + sum(upper_firing_strength .* (1 - alpha), 1);

        output_lower = numerator_lower./denominator;
        output_upper = numerator_upper./denominator;

        output_mean = (output_lower + output_upper)./2;


    elseif type_reduction_method == "NTv2" || type_reduction_method == "NTv2_multi_alpha"

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);

        temp_mf = [learnable_parameters.IVL.a,learnable_parameters.IVL.b];
        temp_delta = [learnable_parameters.IVL.delta_a,learnable_parameters.IVL.delta_b];

        x = permute(x,[2 1 3]); %comment at
        temp_input = [x; ones(1, size(x, 2), size(x, 3))];
        temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation


        linear_lower = (temp_mf * temp_input) - (abs(temp_delta * temp_input));
        linear_upper = (temp_mf * temp_input) + (abs(temp_delta * temp_input));

        linear_lower = reshape(linear_lower, [size(lower_firing_strength_temp, 1), number_outputs, size(lower_firing_strength_temp, 3)]);
        linear_upper = reshape(linear_upper, [size(upper_firing_strength_temp, 1), number_outputs, size(upper_firing_strength_temp, 3)]);

        alpha = sigmoid(learnable_parameters.IVL.alpha);
        beta = sigmoid(learnable_parameters.IVL.beta);

        numerator_lower = sum((lower_firing_strength_temp.* linear_lower) .* alpha,1) + sum((upper_firing_strength_temp.* linear_upper).* (1 - alpha),1);
        numerator_upper = sum((lower_firing_strength_temp.* linear_lower) .* (1 - beta),1) + sum((upper_firing_strength_temp.* linear_upper).* beta,1);

        denominator_lower = sum(lower_firing_strength .* alpha, 1) + sum(upper_firing_strength .* (1 - alpha), 1);
        denominator_upper = sum(lower_firing_strength .* (1 - beta), 1) + sum(upper_firing_strength .* beta, 1);

        output_lower = numerator_lower./denominator_lower;
        output_upper = numerator_upper./denominator_upper;

        output_mean = (output_lower + output_upper)./2;

    elseif type_reduction_method == "NTv2_alpha2"

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);

        temp_mf = [learnable_parameters.IVL.a,learnable_parameters.IVL.b];
        temp_delta = [learnable_parameters.IVL.delta_a,learnable_parameters.IVL.delta_b];

        x = permute(x,[2 1 3]); %comment at
        temp_input = [x; ones(1, size(x, 2), size(x, 3))];
        temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation


        linear_lower = (temp_mf * temp_input) - (abs(temp_delta * temp_input));
        linear_upper = (temp_mf * temp_input) + (abs(temp_delta * temp_input));

        linear_lower = reshape(linear_lower, [size(lower_firing_strength_temp, 1), number_outputs, size(lower_firing_strength_temp, 3)]);
        linear_upper = reshape(linear_upper, [size(upper_firing_strength_temp, 1), number_outputs, size(upper_firing_strength_temp, 3)]);

        alpha = sigmoid(learnable_parameters.IVL.alpha).^2;
        beta = sigmoid(learnable_parameters.IVL.beta).^2;

        numerator_lower = sum((lower_firing_strength_temp.* linear_lower) .* alpha,1) + sum((upper_firing_strength_temp.* linear_upper).* (1 - alpha),1);
        numerator_upper = sum((lower_firing_strength_temp.* linear_lower) .* (1 - beta),1) + sum((upper_firing_strength_temp.* linear_upper).* beta,1);

        denominator_lower = sum(lower_firing_strength .* alpha, 1) + sum(upper_firing_strength .* (1 - alpha), 1);
        denominator_upper = sum(lower_firing_strength .* (1 - beta), 1) + sum(upper_firing_strength .* beta, 1);

        output_lower = numerator_lower./denominator_lower;
        output_upper = numerator_upper./denominator_upper;

        output_mean = (output_lower + output_upper)./2;

    elseif type_reduction_method == "NTv3" || type_reduction_method == "NTv3_multi_alpha"

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);

        temp_mf = [learnable_parameters.IVL.a,learnable_parameters.IVL.b];
        temp_delta = [learnable_parameters.IVL.delta_a,learnable_parameters.IVL.delta_b];

        x = permute(x,[2 1 3]); %comment at
        temp_input = [x; ones(1, size(x, 2), size(x, 3))];
        temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation


        linear_lower = (temp_mf * temp_input) - (abs(temp_delta * temp_input));
        linear_upper = (temp_mf * temp_input) + (abs(temp_delta * temp_input));

        linear_lower = reshape(linear_lower, [size(lower_firing_strength_temp, 1), number_outputs, size(lower_firing_strength_temp, 3)]);
        linear_upper = reshape(linear_upper, [size(upper_firing_strength_temp, 1), number_outputs, size(upper_firing_strength_temp, 3)]);

        alpha = sigmoid(learnable_parameters.IVL.alpha);
        beta = sigmoid(learnable_parameters.IVL.beta);

        numerator_lower = sum((lower_firing_strength_temp.* linear_lower) .* alpha,1) + sum((upper_firing_strength_temp.* linear_lower).* (1 - alpha),1);
        numerator_upper = sum((lower_firing_strength_temp.* linear_upper) .* (1 - beta),1) + sum((upper_firing_strength_temp.* linear_upper).* beta,1);

        denominator_lower = sum(lower_firing_strength .* alpha, 1) + sum(upper_firing_strength .* (1 - alpha), 1);
        denominator_upper = sum(lower_firing_strength .* (1 - beta), 1) + sum(upper_firing_strength .* beta, 1);

        output_lower = numerator_lower./denominator_lower;
        output_upper = numerator_upper./denominator_upper;

        output_mean = (output_lower + output_upper)./2;

    elseif type_reduction_method == "NTv3_alpha2"

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);

        temp_mf = [learnable_parameters.IVL.a,learnable_parameters.IVL.b];
        temp_delta = [learnable_parameters.IVL.delta_a,learnable_parameters.IVL.delta_b];

        x = permute(x,[2 1 3]); %comment at
        temp_input = [x; ones(1, size(x, 2), size(x, 3))];
        temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation


        linear_lower = (temp_mf * temp_input) - (abs(temp_delta * temp_input));
        linear_upper = (temp_mf * temp_input) + (abs(temp_delta * temp_input));

        linear_lower = reshape(linear_lower, [size(lower_firing_strength_temp, 1), number_outputs, size(lower_firing_strength_temp, 3)]);
        linear_upper = reshape(linear_upper, [size(upper_firing_strength_temp, 1), number_outputs, size(upper_firing_strength_temp, 3)]);

        alpha = sigmoid(learnable_parameters.IVL.alpha).^2;
        beta = sigmoid(learnable_parameters.IVL.beta).^2;

        numerator_lower = sum((lower_firing_strength_temp.* linear_lower) .* alpha,1) + sum((upper_firing_strength_temp.* linear_lower).* (1 - alpha),1);
        numerator_upper = sum((lower_firing_strength_temp.* linear_upper) .* (1 - beta),1) + sum((upper_firing_strength_temp.* linear_upper).* beta,1);

        denominator_lower = sum(lower_firing_strength .* alpha, 1) + sum(upper_firing_strength .* (1 - alpha), 1);
        denominator_upper = sum(lower_firing_strength .* (1 - beta), 1) + sum(upper_firing_strength .* beta, 1);

        output_lower = numerator_lower./denominator_lower;
        output_upper = numerator_upper./denominator_upper;

        output_mean = (output_lower + output_upper)./2;

    elseif type_reduction_method == "NTv4" 

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);

        temp_mf = [learnable_parameters.IVL.a,learnable_parameters.IVL.b];
        temp_delta = [learnable_parameters.IVL.delta_a,learnable_parameters.IVL.delta_b];

        x = permute(x,[2 1 3]); %comment at
        temp_input = [x; ones(1, size(x, 2), size(x, 3))];
        temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation


        linear_lower = (temp_mf * temp_input) - (abs(temp_delta * temp_input));
        linear_upper = (temp_mf * temp_input) + (abs(temp_delta * temp_input));

        linear_lower = reshape(linear_lower, [size(lower_firing_strength_temp, 1), number_outputs, size(lower_firing_strength_temp, 3)]);
        linear_upper = reshape(linear_upper, [size(upper_firing_strength_temp, 1), number_outputs, size(upper_firing_strength_temp, 3)]);

        alpha = sigmoid(10*learnable_parameters.IVL.alpha);
        beta = sigmoid(10*learnable_parameters.IVL.beta);

        numerator_lower = sum((lower_firing_strength_temp.* linear_lower) .* alpha,1) + sum((upper_firing_strength_temp.* linear_lower) .* (1 - alpha),1);
        numerator_upper = sum((lower_firing_strength_temp.* linear_upper) .* beta,1) + sum((upper_firing_strength_temp.* linear_upper) .* (1 - beta),1);

        denominator_lower = sum(lower_firing_strength .* alpha, 1) + sum(upper_firing_strength .* (1 - alpha), 1);
        denominator_upper = sum(lower_firing_strength .* beta, 1) + sum(upper_firing_strength .* (1 - beta), 1);

        output_lower = numerator_lower./denominator_lower;
        output_upper = numerator_upper./denominator_upper;

        output_mean = (output_lower + output_upper)./2;

    elseif type_reduction_method == "KM"

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);

        temp_mf = [learnable_parameters.IVL.a,learnable_parameters.IVL.b];
        temp_delta = [learnable_parameters.IVL.delta_a,learnable_parameters.IVL.delta_b];

        x = permute(x,[2 1 3]); %comment at
        temp_input = [x; ones(1, size(x, 2), size(x, 3))];
        temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation

        linear_lower = (temp_mf * temp_input) - (abs(temp_delta * temp_input));
        linear_upper = (temp_mf * temp_input) + (abs(temp_delta * temp_input));

        linear_lower = reshape(linear_lower, [size(lower_firing_strength_temp, 1), number_outputs, size(lower_firing_strength_temp, 3)]);
        linear_upper = reshape(linear_upper, [size(upper_firing_strength_temp, 1), number_outputs, size(upper_firing_strength_temp, 3)]);

        delta_f = upper_firing_strength - lower_firing_strength;
        delta_f = permute(delta_f,[3 1 2]);

        payda2 = delta_f*u;

        pay2_lower = pagemtimes((permute(linear_lower,[3 1 2]).*delta_f),u);
        pay2_lower = permute(pay2_lower,[3,2,1]);
        pay1_lower = permute(sum(linear_lower .* lower_firing_strength,1),[2 1 3]);

        pay_lower = pay1_lower + pay2_lower;

        pay2_upper = pagemtimes((permute(linear_upper,[3 1 2]).*delta_f),u);
        pay2_upper = permute(pay2_upper,[3,2,1]);
        pay1_upper = permute(sum(linear_upper .* lower_firing_strength,1),[2 1 3]);

        pay_upper = pay1_upper + pay2_upper;
        %         clear pay1_upper pay2_upper
        %         clear delta_f u
        payda2 = permute(payda2,[3,2,1]);
        payda1 = sum(lower_firing_strength,1);

        payda = payda1 + payda2;
        %         clear payda1 payda2
        output_lower = pay_lower./payda;
        output_upper = pay_upper./payda;
        %         clear pay_lower pay_upper payda
        output_lower = permute(min(output_lower,[],2),[2 1 3]);
        output_upper = permute(max(output_upper,[],2),[2 1 3]);
        %         clear output_lower_temp output_upper_temp
        output_mean = (output_lower + output_upper)./2;


    elseif type_reduction_method == "KM_BMM"

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);

        temp_mf = [learnable_parameters.IVL.a,learnable_parameters.IVL.b];
        temp_delta = [learnable_parameters.IVL.delta_a,learnable_parameters.IVL.delta_b];

        x = permute(x,[2 1 3]); %comment at
        temp_input = [x; ones(1, size(x, 2), size(x, 3))];
        temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation

        linear_lower = (temp_mf * temp_input) - (abs(temp_delta * temp_input));
        linear_upper = (temp_mf * temp_input) + (abs(temp_delta * temp_input));

        linear_lower = reshape(linear_lower, [size(lower_firing_strength_temp, 1), number_outputs, size(lower_firing_strength_temp, 3)]);
        linear_upper = reshape(linear_upper, [size(upper_firing_strength_temp, 1), number_outputs, size(upper_firing_strength_temp, 3)]);

        delta_f = upper_firing_strength - lower_firing_strength;
        delta_f = permute(delta_f,[3 1 2]);

        payda2 = delta_f*u;

        pay2_lower = pagemtimes((permute(linear_lower,[3 1 2]).*delta_f),u);
        pay2_lower = permute(pay2_lower,[3,2,1]);
        pay1_lower = permute(sum(linear_lower .* lower_firing_strength,1),[2 1 3]);

        pay_lower = pay1_lower + pay2_lower;

        pay2_upper = pagemtimes((permute(linear_upper,[3 1 2]).*delta_f),u);
        pay2_upper = permute(pay2_upper,[3,2,1]);
        pay1_upper = permute(sum(linear_upper .* lower_firing_strength,1),[2 1 3]);

        pay_upper = pay1_upper + pay2_upper;
        %         clear pay1_upper pay2_upper
        %         clear delta_f u
        payda2 = permute(payda2,[3,2,1]);
        payda1 = sum(lower_firing_strength,1);

        payda = payda1 + payda2;
        %         clear payda1 payda2
        output_lower = pay_lower./payda;
        output_upper = pay_upper./payda;
        %         clear pay_lower pay_upper payda
        output_lower = permute(min(output_lower,[],2),[2 1 3]);
        output_upper = permute(max(output_upper,[],2),[2 1 3]);
        %         clear output_lower_temp output_upper_temp

        alpha = sigmoid(learnable_parameters.IVL.alpha);

        output_mean = (output_lower .* alpha) + (output_upper .* (1 - alpha));

    end


end
%output = reshape(output, [s, b]);


output_lower = dlarray(output_lower);
output_upper = dlarray(output_upper);
output_mean = dlarray(output_mean);

end