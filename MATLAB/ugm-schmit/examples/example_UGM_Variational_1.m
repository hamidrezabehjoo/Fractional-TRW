%% Make noisy X
getNoisyX

%% Mean field inference

fprintf('Running Mean Field Inference...\n');
[nodeBelMF,edgeBelMF,logZMF] = UGM_Infer_MeanField(nodePot,edgePot,edgeStruct);

logZMF
pause


%% Loopy Belief Propagation

fprintf('Running loopy belief propagation for inference...\n');
[nodeBelLBP,edgeBelLBP,logZLBP] = UGM_Infer_LBP(nodePot,edgePot,edgeStruct);

logZLBP
pause


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
%semilogy( c, a(:,1) )
plot ( c, a(:,1) )
xlabel('\lambda')
ylabel('logZ')
grid on
saveas(gcf,'myfigure.pdf')

