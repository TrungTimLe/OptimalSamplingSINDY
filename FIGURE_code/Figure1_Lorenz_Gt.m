close all, clear all, clc
%% Lorenz
% RL 0%
Gt_mean = [-0.4282 -2.0810 -0.0046 -2.4641];
Gt_std = [0 0 0 0];
RL1_0 = [1	1	1	1	1	1	1];
RL2_0 = [1	1	1	1	0	0	1];
RL3_0 = [-1	1	-1	1	1	1	0];
RL4_0 = [1	1	1	0	-1	-1	-1];
% RL 0.1%
Gt_mean = [-10.7780 -2.7034 -7.9276 -32.8570];
Gt_std = [6.2313 0.7387 5.4528 9.8091];
RL1_001 = [1	1	0	0	0	0	0	0	-1	1	1	1	1	1	1	1	1	0	1	1	1	0	0	0	0	0	0	0	0	0	0	0];
RL2_002 = [1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0	0	1];
RL3_003 = [1	-1	-1	-1	-1	-1	1	1	1	1	1	1	1	1	1	0	-1	0	0	-1	-1	0	-1];
RL4_004 = [1	1	1	1	1	1	1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1];

% RL 1%
Gt_mean = [-12.6991 -4.2923 -5.1786 -29.3749];
Gt_std = [8.9508 1.3439 2.4902 24.0940];
RL1_001 = [1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0	1	1];
RL2_002 = [1	1	1	1	1	1	-1	-1	1	1	1	1	-1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1];
RL3_003 = [0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0];
RL4_004 = [1	1	1	1	1	1	1	1	-1	-1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0];
%% CVDP
% RL 0%
Gt_mean = [-2.4385 -11.3935 nan -16.4521];
Gt_std = [0 0 0 0];
RL1_0 = [1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1];
RL2_0 = [1	1	1	1	1	-1	1	-1	1	1	-1	1	-1	-1	1	1	0	0	0	0	1	1	0	0	1	1	-1	1	0	0	0	0];
RL3_0 = nan;
RL4_0 = [1	1	1	1	1	1	1	1	1	1	1	1	0	1	1	1	1	1	1	1	1	1	1	1	-1	1	1	1	1	1	1	1	1];
% RL 0.1%
Gt_mean = [-9.9601 -14.8894 nan nan];
Gt_std = [9.5655 1.1223 nan nan];
RL1_001 = [1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1];
RL2_002 = [1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1];
RL3_003 = nan;
RL4_004 = nan;

% RL 1%
Gt_mean = [-12.6472 -19.7510 -5.1786 -29.3749];
Gt_std = [9.0383 1.2526 2.4902 24.0940];
RL1_001 = [1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1];
RL2_002 = [-1	-1	0	0	0	0	0	1	1	1	1	1	1	1	0	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1];
RL3_003 = [0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0];
RL4_004 = [1	1	1	1	1	1	1	1	-1	-1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0];

%% CVDPL
% RL 0%
Gt_mean = [-9.7688 -65.8034 nan -38.1060];
Gt_std = [0 0 0 0];
RL1_0 = [1	1	1	0	0	0	0	1	1	0	1	1	0	0	1	0	0	1	0	0	1	0	1	0	1	0	0	1	0	1	0	0	0	0	1	0	0	1	0	0	1	0	0	1	1	1	0	1	1	0	1	1	1	1	1	0	1	1	1	1	0	1	0	0	1	0	0	0	1	0	0	1	0	1];
RL2_0 = [-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	1	1	1	1	1	1	-1	1	1	1	-1	0	1	1	-1	0	0	-1	1	0	-1	1	0	-1	-1	1	0	0	1	-1	1	0	1	0	0	1	-1	0	0	-1	-1	-1	-1	1	0	-1	-1	-1	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1];
RL3_0 = nan;
RL4_0 = [1	1	1	1	1	1	0	1	1	0	1	1	0	1	1	1	0	1	1	1	1	1	1	0	0	1	1	1	1	1	1	1	1	0	1	1	1	0	1	1	1	1	1	1	0	1	1	1	0	1	1	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0];
% RL 0.1%
Gt_mean = [-24.8662 -24.6019 nan -118.5732];
Gt_std = [9.4781 1.5930 nan 68.5143];
RL1_001 = [1	1	1	1	1	1	1	1	1	1	1	0	0	0	0	0	1	0	0	0	0	0	0	0	-1	1	0	1	0	0	0	0	0	0	0	0	-1	1	0	0	1	0	0	0	0	0	0	0	0	-1	0	0	0	0	0	0	-1	-1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0];
RL2_002 = [1	1	1	1	1	1	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	-1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0];
RL3_003 = nan;
RL4_004 = [1	1	1	1	1	1	1	1	1	1	0	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0	1	0	0	0	0	1	1	0	1	1	-1	1	-1	1	0];

% RL 1%
Gt_mean = [-42.6902 -27.3793 nan  -304.5780];
Gt_std = [12.0506 1.7575 nan  118.3410];
RL1_001 = [1	1	1	1	1	1	1	1	1	1	1	1	1	-1	1	1	1	1	1	-1	1	1	-1	1	1	1	1	1	1	-1	1	1	-1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	-1	1	1	1	1	-1	1	-1	1	1	1	1	1	1	1	1	1	0	1	1	1	1	1	1	1	1	1	1	1	1	1	0	1	1	1	1	1	1	1	1	0	1	1	1	1	1	1	1	1	0	1	1	1	1	0	1	1	0	1	1	0	1	1	1	1	1	1	0	1	1	1];
RL2_002 = [1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0	1	1	1	1	1	1	1	0	0	1	1	1	1	1	1	1	1	1	1	1	0	1	1	0	0	1	1	1	1	0	1	1	1	1	1	0	1	0	0	1	1	0	1	1	1	1	0	1	1	1	1	1	1	1	1	1	1	0	1	1	1];
RL3_003 = nan ;
RL4_004 = [1	1	1	1	-1	1	1	1	1	-1	-1	1	1	1	-1	1	1	1	-1	1	1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	-1	1	1	1	1	-1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	-1	1	1	1	1	-1	-1	1	1	1	1	1	1	1	-1	1	-1	-1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	-1	1	-1	1	1	1	1	1	1	-1	-1	1	1	1	1	1	1	1	-1	1	1	1	1	1	1	1	1	-1	1	-1	1	1	1	1	1	1	1	1	1	-1	1	-1	1	1	1	1	1	-1	1	-1	-1	1	1	1	1	1	1	-1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1];