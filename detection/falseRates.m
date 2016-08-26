function [ falnrate, falprate ] = falseRates( N, numRight, numCir )
%False positives and false negatives per 100 nanodiscs
%
% Parameters:
%    N: number of nanodisc annotations
%    numRight: number of correct segmentations
%    numCir: number of all the segmentations
%
% Returns:
%    falnrate: false negatives per 100 nanodiscs
%    falprate: false positives per 100 nanodiscs
%


falnrate = (N-numRight)/N*100;
falprate = (numCir-numRight)/N*100;


%% plot curves

% % false positives and false negatives corresponding to different symmetry response thresholds

withfixFp = [30.26315789	21.71052632	14.80263158	11.51315789	9.868421053	7.565789474	4.605263158	4.276315789	3.289473684];
withfixFn = [9.539473684	9.868421053	11.51315789	12.82894737	14.47368421	20.06578947	25	31.25	37.82894737];

withoutfixFp = [28.14070352	25.6281407	16.58291457	11.05527638	8.542713568	6.532663317	6.030150754	4.020100503	2.512562814	2.512562814];
withoutfixFn = [6.532663317	7.035175879	8.040201005	10.05025126	12.06030151	14.57286432	20.10050251	28.14070352	32.66331658	35.67839196];

plot(withfixFp, withfixFn, 'r--');
hold on;
plot(withoutfixFp, withoutfixFn, 'b-');

xlabel('False positives per 100 nanodiscs','fontweight','Bold','FontSize',16,'FontName','Times New Roman');
ylabel('False negatives per 100 nanodiscs','fontweight','Bold','FontSize',16,'FontName','Times New Roman');

legend('Dataset A', 'Dataset B');
set(gca, 'XTick', 0:5:35);
set(gca, 'YTick', 0:5:40);

set(gca,'FontName','Times New Roman','fontweight','Bold','FontSize',16);

hold off;



end

