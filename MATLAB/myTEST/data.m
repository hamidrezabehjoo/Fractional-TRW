B = zeros(11, 5);
b = zeros(2,5);
  
B(:,1) = [   27.6860
   27.0942
   26.5595
   26.0790
   25.6477
   25.2599
   24.9103
   24.5938
   24.3062
   24.0437
   23.8032];

b(:,1) = [0.6812 24.6534];

B(:,2) = [   25.8786
   25.3861
   24.9459
   24.5593
   24.2199
   23.9203
   23.6540
   23.4157
   23.2009
   23.0063
   22.8288];
b(:,2) = [0.6199 23.6067];

B(:,3) = [   27.2799
   26.6966
   26.1784
   25.7157
   25.3007
   24.9271
   24.5899
   24.2844
   24.0066
   23.7532
   23.5211];

b(:,3) = [    0.6953   24.2987];

B(:,4) = [   26.2222
   25.6667
   25.1688
   24.7252
   24.3306
   23.9787
   23.6640
   23.3814
   23.1265
   22.8956
   22.6855];
b(:,4) = [ 0.7261 23.3150];




B(:,5) = [  28.0423
   27.4303
   26.8792
   26.3844
   25.9403
   25.5409
   25.1806
   24.8544
   24.5579
   24.2872
   24.0392];
b(:,5) = [ 0.617  25.1243];

c = 0:0.1:1;
figure;
colorstring = 'kbgry';
hold on
for ii = 1:5
 plot(c, B(:,ii), b(1,ii), b(2,ii),'o', 'Color', colorstring(ii), 'LineWidth', 2)
 %plot( b(1,ii), b(2,ii),'o')
 hold on
end


xlabel('\lambda')
ylabel('logZ')
grid on
saveas(gcf,'figure2.pdf')


