function loss = log_cosh_loss(yPred, yTrue, mbs, ts)
    
%     temp = cosh(yTrue - yPred);
%     temp = log(temp);
%     temp = sum(temp,3); %summing in the direction of minibatch
%     loss = 1/mbs*temp;

    loss = (1/(mbs*ts))*(sum(log(cosh(yTrue - yPred)),"all"));

end