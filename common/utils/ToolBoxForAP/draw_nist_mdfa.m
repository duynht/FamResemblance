function [AP] = draw_nist_mdfa(lbl, conf)

pos_conf = conf(lbl==1);
neg_conf = conf(lbl==0);

%------------------------------
%initialize the DCF parameters
Set_DCF (10, 1, 0.01);

%------------------------------
%compute Pmiss and Pfa from experimental detection output scores
[P_miss,P_fa,num_true,sumtrue,sumfalse] = Compute_DET (pos_conf, neg_conf);

%----------------------------------------------------------------------------------------------
P_rec = (num_true-sumtrue) ./ num_true;
P_pr = (num_true-sumtrue) ./ ((num_true-sumtrue) + sumfalse);
%----------------------------------------------------------------------------------------------


% Set tic marks
Pmiss_min = 0.01;
Pmiss_max = 0.99;
Pfa_min = 0.01;
Pfa_max = 0.99;
Set_DET_limits(Pmiss_min,Pmiss_max,Pfa_min,Pfa_max);

%-------------------------------------------
psum = 0;
for t = 0 : 0.1 : 1
    p = max(P_pr(P_rec>=t));
    if (isempty(p))
        p=0;
    end
    psum = psum+p;
end
AP = psum/11;
%------------------------------------------

