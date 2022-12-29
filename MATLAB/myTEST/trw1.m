
GenIsingModel

%% Exact Value
[nodeBel,edgeBel,logZ] = UGM_Infer_Exact(nodePot,edgePot,edgeStruct);
logZ

%% Loopy Belief Propagation
fprintf('Running loopy belief propagation for inference...\n');
[nodeBelLBP,edgeBelLBP,logZLBP] = UGM_Infer_LBP(nodePot,edgePot,edgeStruct);
logZLBP


%% Tree-Reweighted Belief Propagation
fprintf('Running tree-reweighted belief propagation for inference...\n');
[nodeBelTRBP,edgeBelTRBP,logZTRBP, mu] = UGM_Infer_TRBP(nodePot,edgePot,edgeStruct);
logZTRBP
%mu
i = 1;
a = zeros(11,1);
for lam = 0:0.1:1
	mu1 = mu + lam * (1-mu);
	[nodeBelTRBP,edgeBelTRBP,logZTRBP, b] = UGM_Infer_TRBP(nodePot,edgePot,edgeStruct,mu1);
	a(i,1) = logZTRBP;
	i = i+1;
end


c = 0:0.1:1;
lambda_I = interp1(a, c, logZ); % find the value of lambda corresponding to 
                                % the exact value of partition function

plot ( c, a(:,1), 'LineWidth',2 )
hold on
plot(lambda_I, logZ, 'o', 'LineWidth',2)
xlabel('\lambda')
ylabel('logZ')
grid on
saveas(gcf,'figure1.pdf')

