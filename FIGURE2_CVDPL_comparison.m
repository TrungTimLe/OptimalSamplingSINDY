clear all, close all, clc

%% Panel (a)
figure('Position',[1,1,1220,947])

subplot(3,1,1)
mean_mat = [55 55 51 68; 9.15E-04 2.77E-04 5.38E-04 9.74E-04;...
    18.53 26.15 18.00 16.62; 28.16 33.04 29.02 9.29;...
    6.80 4.93 10.52 32.24; 37 70 4088 340];
mean_mat_norm = [];
% Normalize data
for i = 1:size(mean_mat,1)
    tmp = mean_mat(i,:);
    mean_mat_norm(i,:) = tmp/max(tmp);
end

label_mean1 = mean_mat(:,1);
label_mean2 = mean_mat(:,2);
label_mean3 = mean_mat(:,3);
label_mean4 = mean_mat(:,4);

b = bar(1:6,mean_mat_norm);
xticks(1:6)
xticklabels({'Sample size $n$', '$\varepsilon_{{SINDy}}$', '$\log(\kappa(\mathbf{\Theta}))$', '$\log({tr}(\mathbf{I}))$', '$T_{{train}}$', '$T_{{elapsed}}$'})
set(gca, 'TickLabelInterpreter','latex','FontWeight','bold');
hold on
% Get the x coordinate of the bars

xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;

xtips2 = b(2).XEndPoints;
ytips2 = b(2).YEndPoints;

xtips3 = b(3).XEndPoints;
ytips3 = b(3).YEndPoints;

xtips4 = b(4).XEndPoints;
ytips4 = b(4).YEndPoints;

labels1 = string(round(label_mean1,2));
labels1(5) = strcat(labels1(5)," (s)"); labels1(6) = strcat(labels1(6)," (s)");
labels2 = string(round(label_mean2,2));
labels2(5) = strcat(labels2(5)," (s)"); labels2(6) = strcat(labels2(6)," (s)");
labels3 = string(round(label_mean3,2));
labels3(5) = strcat(labels3(5)," (s)"); labels3(6) = strcat(labels3(6)," (s)");
labels4 = string(round(label_mean4,2));
labels4(5) = strcat(labels4(5)," (s)"); labels4(6) = strcat(labels4(6)," (s)");
labels1(2) = "3.04E-04"; labels2(2) = "2.66E-04"; labels3(2) = "2.99E-04"; labels4(2) = "2.32E-04";

FS = 12;
ht = text(xtips1,ytips1+0.05,labels1,'HorizontalAlignment','left','FontSize',FS,'FontWeight','bold','Color',[0 0.4470 0.7410]);
set(ht,'Rotation',90)
ht = text(xtips2,ytips2+0.05,labels2,'HorizontalAlignment','left','FontSize',FS,'FontWeight','bold','Color',[0.8500 0.3250 0.0980]);
set(ht,'Rotation',90)
ht = text(xtips3,ytips3+0.05,labels3,'HorizontalAlignment','left','FontSize',FS,'FontWeight','bold','Color',[0.9290 0.6940 0.1250]);
set(ht,'Rotation',90)
ht = text(xtips4,ytips4+0.05,labels4,'HorizontalAlignment','left','FontSize',FS,'FontWeight','bold','Color',[0.4940 0.1840 0.5560]);
set(ht,'Rotation',90)

grid on
ylim([0 2])
yticks([0 0.5 1])
set(gca,'FontWeight','bold','LineWidth',1.2,'FontSize',14);
title({'(a) $\varepsilon_{tol} = 0.001$'}, 'Interpreter', 'latex');
ylabels = {'\color{blue}\fontsize{18}NSR = 0%', '\color{black}\fontsize{15}Normalized value'};
ylabel(ylabels,'FontWeight','bold','Interpreter', 'tex')
%% Panel (b)
subplot(3,1,2)
mean_mat = [162.27 0 79 180.86; 3.1131 0 3.0319 3.9283;...
    15.47 0 16.1064 14.13; 5.79 0 7.7513 3.44;...
    10.28 0 20.7560 50.07; 391 954 11392 629;
    100 0 0 91];

std_mat = [64.44 0 0 44.66; 1.2744 0 0 0.9435;...
    0.47 0 0 0.37; 1.14 0 0 1.02;...
    4.12 0 0 29.16];
mean_mat_norm = [];
std_mat_norm = [];
% Normalize data
for i = 1:size(mean_mat,1)
    tmp = mean_mat(i,:);
    mean_mat_norm(i,:) = tmp/max(tmp);
    if i <= 5
        std_mat_norm(i,:) = std_mat(i,:)/max(tmp);
    end
end

label_mean1 = mean_mat(:,1);
label_mean2 = mean_mat(:,2);
label_mean3 = mean_mat(:,3);
label_mean4 = mean_mat(:,4);

b = bar(1:7,mean_mat_norm);
xticks(1:7)
xticklabels({'Sample size $n$', '$\varepsilon_{{SINDy}}$', '$\log(\kappa(\mathbf{\Theta}))$', '$\log({tr}(\mathbf{I}))$', '$T_{{train}}$', '$T_{{elapsed}}$','Convergence rate'})
set(gca, 'TickLabelInterpreter','latex','FontWeight','bold');
hold on
% Get the x coordinate of the bars
[ngroups,nbars] = size(mean_mat_norm);
x = nan(nbars, ngroups);
for i = 1:nbars
    x(i,:) = b(i).XEndPoints;
end
errorbar(x(:,1:5)',mean_mat_norm(1:5,:),std_mat_norm,'k','linestyle','none');% Adding the errorbars
% Get the x coordinate of the bars
legend('Best uniform sampling','Greedy sampling','Randomized brute-force search','RL-based sampling','Error bar','FontSize',12,'Orientation','horizontal')
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;

xtips2 = b(2).XEndPoints;
ytips2 = b(2).YEndPoints;

xtips3 = b(3).XEndPoints;
ytips3 = b(3).YEndPoints;

xtips4 = b(4).XEndPoints;
ytips4 = b(4).YEndPoints;

labels1 = string(round(label_mean1,2));
labels2 = string(round(label_mean2,2));
labels2(1:5) = "N/A";
labels3 = string(round(label_mean3,2));
labels3(7) = "N/A";
labels4 = string(round(label_mean4,2));
labels1(5) = strcat(labels1(5)," (s)"); labels1(6) = strcat(labels1(6)," (s)"); labels1(7) = strcat(labels1(7),"%");
labels2(6) = strcat(labels2(6)," (s)"); labels2(7) = strcat(labels2(7),"%");
labels3(5) = strcat(labels3(5)," (s)"); labels3(6) = strcat(labels3(6)," (s)"); 
labels4(5) = strcat(labels4(5)," (s)"); labels4(6) = strcat(labels4(6)," (s)"); labels4(7) = strcat(labels4(7),"%");

err1 = [std_mat_norm(:,1)' 0 0];
ht = text(xtips1,ytips1+err1+0.05,labels1,'HorizontalAlignment','left','FontSize',FS,'FontWeight','bold','Color',[0 0.4470 0.7410]);
set(ht,'Rotation',90)
err1 = [std_mat_norm(:,2)' 0 0];
ht = text(xtips2,ytips2+err1+0.05,labels2,'HorizontalAlignment','left','FontSize',FS,'FontWeight','bold','Color',[0.8500 0.3250 0.0980]);
set(ht,'Rotation',90)
err1 = [std_mat_norm(:,3)' 0 0];
ht = text(xtips3,ytips3+err1+0.05,labels3,'HorizontalAlignment','left','FontSize',FS,'FontWeight','bold','Color',[0.9290 0.6940 0.1250]);
set(ht,'Rotation',90)
err1 = [std_mat_norm(:,4)' 0 0];
ht = text(xtips4,ytips4+err1+0.05,labels4,'HorizontalAlignment','left','FontSize',FS,'FontWeight','bold','Color',[0.4940 0.1840 0.5560]);
set(ht,'Rotation',90)

grid on
ylim([0 2.5])
yticks([0 0.5 1])
set(gca,'FontWeight','bold','LineWidth',1.2,'FontSize',14);
title({'(b) $\varepsilon_{tol} = 5$'}, 'Interpreter', 'latex');
ylabels = {'\color{blue}\fontsize{18}NSR = 0.1%', '\color{black}\fontsize{15}Normalized value'};
ylabel(ylabels,'FontWeight','bold','Interpreter', 'tex')

%% Panel (c)
subplot(3,1,3)
mean_mat = [292.06 0 122 311.65; 16.9572 0 13.9038 16.2855;...
    13.98 0 14.7927 13.86; 2.50 0 5.2039 2.22;...
    37.15 0 29.16 153.37; 407 1471 15048 737;
    85 0 0 84];

std_mat = [76.41 0 0 86.98; 2.6349 0 0 3.3686;...
    0.21 0 0 0.22; 0.62 0 0 0.62;...
    9.78 0 0 42.95];
mean_mat_norm = [];
std_mat_norm = [];
% Normalize data
for i = 1:size(mean_mat,1)
    tmp = mean_mat(i,:);
    mean_mat_norm(i,:) = tmp/max(tmp);
    if i <= 5
        std_mat_norm(i,:) = std_mat(i,:)/max(tmp);
    end
end

label_mean1 = mean_mat(:,1);
label_mean2 = mean_mat(:,2);
label_mean3 = mean_mat(:,3);
label_mean4 = mean_mat(:,4);

b = bar(1:7,mean_mat_norm);
xticks(1:7)
xticklabels({'Sample size $n$', '$\varepsilon_{{SINDy}}$', '$\log(\kappa(\mathbf{\Theta}))$', '$\log({tr}(\mathbf{I}))$', '$T_{{train}}$', '$T_{{elapsed}}$','Convergence rate'})
set(gca, 'TickLabelInterpreter','latex','FontWeight','bold');
hold on
% Get the x coordinate of the bars
[ngroups,nbars] = size(mean_mat_norm);
x = nan(nbars, ngroups);
for i = 1:nbars
    x(i,:) = b(i).XEndPoints;
end
errorbar(x(:,1:5)',mean_mat_norm(1:5,:),std_mat_norm,'k','linestyle','none');% Adding the errorbars
% Get the x coordinate of the bars

xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;

xtips2 = b(2).XEndPoints;
ytips2 = b(2).YEndPoints;

xtips3 = b(3).XEndPoints;
ytips3 = b(3).YEndPoints;

xtips4 = b(4).XEndPoints;
ytips4 = b(4).YEndPoints;

labels1 = string(round(label_mean1,2));
labels2 = string(round(label_mean2,2));
labels2(1:5) = "N/A";
labels3 = string(round(label_mean3,2));
labels3(7) = "N/A";
labels4 = string(round(label_mean4,2));
labels1(5) = strcat(labels1(5)," (s)"); labels1(6) = strcat(labels1(6),"s"); labels1(7) = strcat(labels1(7),"%");
labels2(6) = strcat(labels2(6)," (s)"); labels2(7) = strcat(labels2(7),"%");
labels3(5) = strcat(labels3(5)," (s)"); labels3(6) = strcat(labels3(6)," (s)"); 
labels4(5) = strcat(labels4(5)," (s)"); labels4(6) = strcat(labels4(6)," (s)"); labels4(7) = strcat(labels4(7),"%");

err1 = [std_mat_norm(:,1)' 0 0];
ht = text(xtips1,ytips1+err1+0.05,labels1,'HorizontalAlignment','left','FontSize',FS,'FontWeight','bold','Color',[0 0.4470 0.7410]);
set(ht,'Rotation',90)
err1 = [std_mat_norm(:,2)' 0 0];
ht = text(xtips2,ytips2+err1+0.05,labels2,'HorizontalAlignment','left','FontSize',FS,'FontWeight','bold','Color',[0.8500 0.3250 0.0980]);
set(ht,'Rotation',90)
err1 = [std_mat_norm(:,3)' 0 0];
ht = text(xtips3,ytips3+err1+0.05,labels3,'HorizontalAlignment','left','FontSize',FS,'FontWeight','bold','Color',[0.9290 0.6940 0.1250]);
set(ht,'Rotation',90)
err1 = [std_mat_norm(:,4)' 0 0];
ht = text(xtips4,ytips4+err1+0.05,labels4,'HorizontalAlignment','left','FontSize',FS,'FontWeight','bold','Color',[0.4940 0.1840 0.5560]);
set(ht,'Rotation',90)

grid on
ylim([0 2.5])
yticks([0 0.5 1])
xlabel('Evaluation metrics','FontWeight','bold')
set(gca,'FontWeight','bold','LineWidth',1.2,'FontSize',14);
title({'(c) $\varepsilon_{tol} = 20$'}, 'Interpreter', 'latex');
ylabels = {'\color{blue}\fontsize{18}NSR = 1%', '\color{black}\fontsize{15}Normalized value'};
ylabel(ylabels,'FontWeight','bold','Interpreter', 'tex')