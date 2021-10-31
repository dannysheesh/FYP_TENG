clear all; close all; clc;


%hyperparameters

% comparison table
% {1 = walking (socks)          -> TENG front leg = 'fyp7'} putty_test1_20211012_160312
% {2 = walking (shoes)          -> TENG front leg = 'fyp8'} putty_test1_20211012_161041
% {3 = walking                  -> TENG back leg = 'f}      putty_test1_20211020_163559
% {4 = walking                  -> TENG left leg}           No dataset
% {5 = walking                  -> TENG right leg}          putty_test1_20211020_163955
% {6 = walking                  -> TENG right kneecap}      No dataset
% {7 = jogging                  -> TENG front leg}          putty_test1_20211020_165302
% {8 = running (faster than 6)  -> TENG front leg}          putty_test1_20211020_165533


shoes_vs_socks = 1;             %set to 1 to include comparison

%select which values of the dataset to export
gait_index_1 = 1
gait_index_2 = 8
% gait_index_2 = 'end'

%select the test to be compared **DO NOT SELECT '3', '4' or '6'!!
comparison_1 = 8;
comparison_2 = 2;

%export 'comparison_1' dataset
export_vals = 1;                     %set export = 1 to export datapoints

LPF_oo = 1;                     %set = 1 for low pass filter
passing_frequency = 10;         %passband frequency for low-pass filter
sampling_frequency = 100;      %sampling frequency


min_offset = 25;                %number of samples before foot event
max_offset = 50;                %number of samples after foot event

train_precent = 0.5;            %percent of data to be used in training set

%if set to '1', fewer plot will be displayed, .etc
one_data = 0;

%%%%%ensure same amplitude on plots

% % % % % % % % if shoes_vs_socks == 1

%% use a for loop lol

%FYP7
if comparison_1 == 1 || comparison_2 == 1
    if comparison_1 == 1
        %fyp7 dataset - socks on
        teng_data1 = importdata('putty_test1_20211012_160312.txt');
        gait1 = readmatrix('fyp7_socks_gait_timings.xlsx');
        gait1(:, 1:2) = 100*(gait1(:, 1:2) - gait1(1, 3));   %subtract time offset
    else
        %fyp7 dataset - socks on
        teng_data2 = importdata('putty_test1_20211012_160312.txt');
        gait2 = readmatrix('fyp7_socks_gait_timings.xlsx');
        gait2(:, 1:2) = 100*(gait2(:, 1:2) - gait2(1, 3));   %subtract time offset
    end
end

%FYP8
if comparison_1 == 2 || comparison_2 == 2
    if comparison_1 == 2
        %fyp7 dataset - socks on
        teng_data1 = importdata('putty_test1_20211012_161041.txt');
        gait1 = readmatrix('fyp8_shoes_gait_timings.xlsx');
        gait1(:, 1:2) = 100*(gait1(:, 1:2) - gait1(1, 3));   %subtract time offset
    else
        %fyp7 dataset - socks on
        teng_data2 = importdata('putty_test1_20211012_161041.txt');
        gait2 = readmatrix('fyp8_shoes_gait_timings.xlsx');
        gait2(:, 1:2) = 100*(gait2(:, 1:2) - gait2(1, 3));   %subtract time offset
    end
end

%FYP - not done yet
if comparison_1 == 3 || comparison_2 == 3
    if comparison_1 == 3
        %fyp7 dataset - socks on
        teng_data1 = importdata('putty_test1_20211020_163559.txt');
        gait1 = readmatrix('FYP_walk_back_r1.xlsx');
        gait1(:, 1:2) = 100*(gait1(:, 1:2) - gait1(1, 3));   %subtract time offset
    else
        %fyp7 dataset - socks on
        teng_data2 = importdata('putty_test1_20211020_163559.txt');
        gait2 = readmatrix('FYP_walk_back_r1.xlsx');
        gait2(:, 1:2) = 100*(gait2(:, 1:2) - gait2(1, 3));   %subtract time offset
    end
end

%FYP - not done yet
if comparison_1 == 4 || comparison_2 == 4
    if comparison_1 == 4
        %fyp7 dataset - socks on
        teng_data1 = importdata('putty_test1_20211020_163955.txt');
        gait1 = readmatrix('FYP_walk_right1.xlsx');
        gait1(:, 1:2) = 100*(gait1(:, 1:2) - gait1(1, 3));   %subtract time offset
    else
        %fyp7 dataset - socks on
        teng_data2 = importdata('putty_test1_20211012_160312.txt');
        gait2 = readmatrix('fyp7_socks_gait_timings.xlsx');
        gait2(:, 1:2) = 100*(gait2(:, 1:2) - gait2(1, 3));   %subtract time offset
    end
end


if comparison_1 == 5 || comparison_2 == 5
    if comparison_1 == 5
        %fyp7 dataset - socks on
        teng_data1 = importdata('putty_test1_20211020_163955.txt');
        gait1 = readmatrix('FYP_walk_right1.xlsx');
        gait1(:, 1:2) = 100*(gait1(:, 1:2) - gait1(1, 3));   %subtract time offset
    else  %fyp7 dataset - socks on
        teng_data2 = importdata('putty_test1_20211020_163955.txt');
        gait2 = readmatrix('FYP_walk_right1.xlsx');
        gait2(:, 1:2) = 100*(gait2(:, 1:2) - gait2(1, 3));   %subtract time offset
    end 
end

%FYP - not done yet
if comparison_1 == 6 || comparison_2 == 6
    if comparison_1 == 6
        %fyp7 dataset - socks on
        teng_data = importdata('putty_test1_20211012_160312.txt');
        gait1 = readmatrix('FYP_jogging1.xlsx');
        gait1(:, 1:2) = 100*(gait1(:, 1:2) - gait1(1, 3));   %subtract time offset
    else
        %fyp7 dataset - socks on
        teng_data2 = importdata('putty_test1_20211012_160312.txt');
        gait2 = readmatrix('fyp7_socks_gait_timings.xlsx');
        gait2(:, 1:2) = 100*(gait2(:, 1:2) - gait2(1, 3));   %subtract time offset
    end
end
if comparison_1 == 7 || comparison_2 == 7
    if comparison_1 == 7
        %fyp7 dataset - socks on
        teng_data1 = importdata('putty_test1_20211020_165302.txt');
        gait1 = readmatrix('FYP_sprinting1.xlsx');
        gait1(:, 1:2) = 100*(gait1(:, 1:2) - gait1(1, 3));   %subtract time offset
    else
        %fyp7 dataset - socks on
        teng_data2 = importdata('putty_test1_20211020_165302.txt');
        gait2 = readmatrix('fyp7_socks_gait_timings.xlsx');
        gait2(:, 1:2) = 100*(gait2(:, 1:2) - gait2(1, 3));   %subtract time offset
    end
end

if comparison_1 == 8 || comparison_2 == 8
    if comparison_1 == 8
        %fyp7 dataset - socks on
        teng_data1 = importdata('putty_test1_20211020_165533.txt');
        gait1 = readmatrix('FYP_sprinting1.xlsx');
        gait1(:, 1:2) = 100*(gait1(:, 1:2) - gait1(1, 3));   %subtract time offset
    else
        %fyp7 dataset - socks on
        teng_data2 = importdata('putty_test1_20211020_165533.txt');
        gait2 = readmatrix('FYP_sprinting1.xlsx');
        gait2(:, 1:2) = 100*(gait2(:, 1:2) - gait2(1, 3));   %subtract time offset
    end
end        
        








teng_data1 = teng_data1.data;
teng_data2 = teng_data2.data;
        
        
        
        
        
        
%{      
    %fyp8 dataset - shoes on
    teng_data2 = importdata('putty_test1_20211012_161041.txt');
    gait2 = readmatrix('fyp8_shoes_gait_timings.xlsx');
    gait2(:, 1:2) = 100*(gait(:, 1:2) - gait(1, 3));   %subtract time offset
%}
    
%     teng = teng_data1.data;
    tt = linspace(0, length(teng_data1)/100, length(teng_data1));

    data2_socks = teng_data1;
    %low-pass filter
    if LPF_oo == 1
        teng_data1 = lowpass(teng_data1, passing_frequency, sampling_frequency);
        teng_data2 = lowpass(teng_data2, passing_frequency, sampling_frequency);
    end
    
% end
% %     figure(200)
% %     plot(linspace(0, length(teng_data)/100, length(teng_data)), teng_data)
    
    
    
    
    
    strike_launch = 1;
    [gait_cycles_stance1] = gait_storer78(teng_data1, gait1, min_offset, max_offset, strike_launch);

    gait_cycles_stance1_og = gait_cycles_stance1;
    
    strike_launch = 2;
    [gait_cycles_swing1] = gait_storer78(teng_data1, gait1, min_offset, max_offset, strike_launch);
    
    gait_cycles_swing1_og = gait_cycles_swing1;
    
    if gait_index_2 == 'end'
        gait_cycles_stance1_placeholder = gait_cycles_stance1(:, gait_index_1:end);
        gait_cycles_swing1_placeholder = gait_cycles_swing1(:, gait_index_1:end); 
    else
        gait_cycles_stance1_placeholder = gait_cycles_stance1(:, gait_index_1:gait_index_2);
        gait_cycles_swing1_placeholder = gait_cycles_swing1(:, gait_index_1:gait_index_2);
    end
    
    gait_cycles_stance1 = gait_cycles_stance1_placeholder;
    gait_cycles_swing1 = gait_cycles_swing1_placeholder;
    
if one_data ~= 1
    strike_launch = 1;
    [gait_cycles_stance2] = gait_storer78(teng_data2, gait2, min_offset, max_offset, strike_launch);

    strike_launch = 2;
    [gait_cycles_swing2] = gait_storer78(teng_data2, gait2, min_offset, max_offset, strike_launch);    

    
    num_1 = 8; num_2 = 17;
    
    figure(100)
%     plot(linspace(0, length(gait_cycles_stance1(:, 1)/100), length(gait_cycles_stance1(:, 1))), gait_cycles_stance1(:, num_1:num_2))
    plot(linspace(0, 10*length(gait_cycles_stance1(:, 1)/100), length(gait_cycles_stance1(:, 1))), gait_cycles_stance1(:, :), [250, 250], [600, 1800], [220, 220], [600, 700], [220, 220], [800, 900], [220, 220], [1000, 1100]...
        , [220, 220], [1200, 1300], [220, 220], [1400, 1500], [220, 220], [1600, 1700]    , [280, 280], [600, 700], [280, 280], [800, 900], [28, 28], [1000, 1100]...
        , [280, 280], [1200, 1300], [280, 280], [1400, 1500], [280, 280], [1600, 1700])                    
%     plot(linspace(0, length(gait_cycles_stance1(:, 1)/100), length(gait_cycles_stance1(:, 1))), mean(gait_cycles_stance1(:, :), 2))             %mean
%     title('Heel Strike w. Shoes On Wooden Floor - Rear Calf Measurement')
    title('Heel Strike running (fast) w. Shoes On Wooden Floor - Front Calf Measurement')
    xlabel('time (mS)')
    ylabel('voltage (mV)')
    
    figure(101)
%     plot(linspace(0, length(gait_cycles_swing1(:, 1)/100), length(gait_cycles_swing1(:, 1))), gait_cycles_swing1(:, num_1:num_2))
    plot(linspace(0, 10*length(gait_cycles_swing1(:, 1)/100), length(gait_cycles_swing1(:, 1))), gait_cycles_swing1(:, :), [250, 250], [600, 1800], [220, 220], [600, 700], [220, 220], [800, 900], [220, 220], [1000, 1100]...
        , [220, 220], [1200, 1300], [220, 220], [1400, 1500], [220, 220], [1600, 1700]    , [280, 280], [600, 700], [280, 280], [800, 900], [280, 280], [1000, 1100]...
        , [280, 280], [1200, 1300], [280, 280], [1400, 1500], [280, 280], [1600, 1700])
%     plot(linspace(0, length(gait_cycles_swing1(:, 1)/100), length(gait_cycles_swing1(:, 1))), mean(gait_cycles_swing1(:, :), 2))
%     title('Toe-Off w. Shoes On Wooden Floor - Rear Calf Measurement')
    title('Toe-off Running (fast) w. Shoes On Wooden Floor - Front Calf Measurement')
    xlabel('time (mS)')
    ylabel('voltage (mV)')
    
    
% [a] = plot_mat(stance1, swing1, stance2, swing2, title1, title2)
    
end
% % % % %     plot_mat(gait_cycles_stance1, gait_cycles_swing1, gait_cycles_stance2, gait_cycles_swing2, 'Socks on wooden floor', 'Sneakers on wooden floor', min_offset, max_offset)
   

%% uncomment this line:
% plot_mat(gait_cycles_stance1, gait_cycles_swing1, gait_cycles_stance2, gait_cycles_swing2, 'Socks on wooden floor', 'Sneakers on wooden floor', min_offset, max_offset)



%     plot(
    
    
% % %     figure(27)
% % %     subplot(2, 1, 1)
% % %     for i = 1:length(gait_cycles_stance(1, :))
% % %         hold on
% % %         plot(linspace(0, (abs(min_offset) + abs(max_offset))/100, length(gait_cycles_stance(:, i))), gait_cycles_stance(:, i))
% % %         plot(linspace((min_offset/100), (min_offset/100), 10), linspace(0, 2000, 10))
% % %         hold off
% % %     end
% % %     title('Stance Phase')
% % % 
% % %     subplot(2, 1, 2)
% % %     for i = 1:length(gait_cycles_swing(1, :))
% % %         hold on
% % %         plot(linspace(0, (abs(min_offset) + abs(max_offset))/100, length(gait_cycles_swing(:, i))), gait_cycles_swing(:, i))
% % %         plot(linspace((min_offset/100), (min_offset/100), 10), linspace(0, 2000, 10))
% % %         hold off
% % %     end
% % %     title('Swing Phase');
% % %     
% % %     if export_vals == 1
% % %         export_data(gait_cycles_stance, gait_cycles_swing, train_precent);
% % %     end
    
% % % % % % % % end



if export_vals == 1

%     export_data(gait_cycles_stance1, gait_cycles_swing1, train_precent)

    export_data(gait_cycles_stance1_og, gait_cycles_swing1_og, train_precent)


end











function [a] = plot_mat(stance1, swing1, stance2, swing2, title1, title2, min_offset, max_offset)


figure(27)
    title('Stance Comparison')
    subplot(2, 1, 1)
    for i = 1:length(stance1(1, :))
        hold on
        plot(linspace(0, (abs(min_offset) + abs(max_offset))/100, length(stance1(:, i))), stance1(:, i))
        plot(linspace((min_offset/100), (min_offset/100), 10), linspace(0, 2000, 10))
        hold off
    end
    title(title1)

    subplot(2, 1, 2)
    for i = 1:length(stance2(1, :))
        hold on
        plot(linspace(0, (abs(min_offset) + abs(max_offset))/100, length(stance2(:, i))), stance2(:, i))
        plot(linspace((min_offset/100), (min_offset/100), 10), linspace(0, 2000, 10))
        hold off
    end
    title(title2);

    
    figure(28)
    title('Swing Comparison')
    subplot(2, 1, 1)
    for i = 1:length(swing1(1, :))
        hold on
        plot(linspace(0, (abs(min_offset) + abs(max_offset))/100, length(swing1(:, i))), swing1(:, i))
        plot(linspace((min_offset/100), (min_offset/100), 10), linspace(0, 2000, 10))
        hold off
    end
    title(title1)

    subplot(2, 1, 2)
    for i = 1:length(swing2(1, :))
        hold on
        plot(linspace(0, (abs(min_offset) + abs(max_offset))/100, length(swing2(:, i))), swing2(:, i))
        plot(linspace((min_offset/100), (min_offset/100), 10), linspace(0, 2000, 10))
        hold off
    end
    title(title2);



end


function [gait_cycles] = gait_storer78(teng, gait, min_offset, max_offset, strike_launch)

gait_cycles = zeros((min_offset + max_offset + 1), length(gait(:, 1)));
size(gait_cycles)
strike_launch;

if strike_launch == 1               %heel strike
    for i = 1:length(gait(:, 1))
        i
        gait_cycles(:, i) = teng((gait(i, 1) - min_offset):(gait(i, 1) + max_offset));
    end
elseif strike_launch == 2           %toe-off
    for i = 1:length(gait(:, 2))
        gait_cycles(:, i) = teng((gait(i, 2) - min_offset):(gait(i, 2) + max_offset));
    end
end






end


















function [a] = export_data(stance, swing, train_percent)
% this function takes stance and swing data as inputs and converts/exports
% the files such that they are usable in a neural network

%should export this function into its own m file

% stance            - stance teng data
% swing             - swing teng data
% train_test_ratio  - ratio of output train to test data

% if statement ensuring that there exists the same number of steps in the
% stance and swing matrices
if length(stance(1, :)) ~= length(swing(1, :))
    error('Stance and Swing matrices should contain the same number of steps');
else

    % create matrix of train_test_ratio

%     shuffling stance and swing data
    
    % credit for row shuffle code - modified
    % https://www.mathworks.com/matlabcentral/answers/324891-how-to-randomly-shuffle-the-row-elements-of-a-predefined-matrix
    [m,n] = size(stance) ;
    idx = randperm(n) ;
    b = stance 
    for i = 1:length(stance(:, 1))
    b(i,idx) = stance(i,:);  % first row arranged randomly
    end
    stance_shuffled = b;
    stance_shuffled_train = stance_shuffled(:, 1:round(length(b(1, :))*train_percent));         % training portion of stance data
    stance_shuffled_test = stance_shuffled(:, (round(length(b(1, :))*train_percent)+1):end);    % testing portion of stance data
    size(stance_shuffled_train)
    stancetr1 = length(stance_shuffled_train)
    swingte1 = length(stance_shuffled_test)
    
    [m,n] = size(swing);
    idx = randperm(n);
    b = swing;
    for i = 1:length(swing(:, 1))
    b(i,idx) = swing(i,:);  % first row arranged randomly
    end
    swing_shuffled = b;
    swing_shuffled_train = swing_shuffled(:, 1:round(length(b(1, :))*train_percent));           % training portion of swing data
    swing_shuffled_test = swing_shuffled(:, (round(length(b(1, :))*train_percent)+1):end);      % testing portion of swing data
    


    %% training dataset
    
    file_directory = 1;     %set = 1 for filename relative to "C:\Users..."
    im = 1;                 %set im = 1 for files to be exported as images rather than .txt files

    train_count = 1;
    train_data = zeros(length(stance_shuffled_train(1, :)), 2);

    
    % for loop saving a .txt file for each stance
    for i = 1:length(stance_shuffled_train(1, :))

        % create file name that increases over iterations
            filename = strcat(strcat('C:\Users\mango\OneDrive\Documents\Uni\FYP\PuTTY logs\testing\training data\stance_train_data_', num2str(i)), '.txt');
        
        if file_directory == 1
%             filename2 = strcat(strcat('C:\Users\mango\OneDrive\Documents\Uni\FYP\PuTTY logs\testing\training data\stance_train_data_', num2str(i)), '.txt');
            filename2 = strcat(strcat('C:/Users/mango/OneDrive/Documents/Uni/FYP/PuTTY logs/testing/training data/stance_train_data_', num2str(i)), '.txt');
        else
            filename2 = strcat(strcat('stance_train_data_', num2str(i)), '.txt');
        end
        % save each heel strike into seperate .txt files
        
        if im == 1
            imwrite(stance_shuffled_train(:, 1), strcat(filename(1:end-3), '.png'), 'PNG')
            filename2(end-3:end) = '.png'; filename(end-3:end) = '.png';
        else
            writematrix(stance_shuffled_train(:, 1), filename)
        end
% % %         train_data(train_count, 1) = filename;
% % %         train_data(train_count, 2) = 0;
        st_sw_train(train_count) = 0;             % 0 = stance, 1 = swing
        filename_list_train(train_count) = {filename2};
        train_count = train_count + 1;
        %{
    % % %     writematrix(stance(:, 1), 'C:\Users\mango\OneDrive\Documents\Uni\FYP\PuTTY logs\testing\training data\gait_data_1.txt')
    % % %     directory = 'C:\Users\mango\OneDrive\Documents\Uni\FYP\PuTTY logs\testing\training data'; 
    % % % filename = 'data.xlsx';
    % % % writetable(C{p},fullfile(directory,filename));
        %}
    end


    % for loop saving a .txt file for each toe-off
    for i = 1:length(swing_shuffled_train(1, :))

        % create file name that increases over iterations
            filename = strcat(strcat('C:\Users\mango\OneDrive\Documents\Uni\FYP\PuTTY logs\testing\training data\swing_train_data_', num2str(i)), '.txt');
        
        if file_directory == 1
%             filename2 = strcat(strcat('C:\Users\mango\OneDrive\Documents\Uni\FYP\PuTTY logs\testing\training data\swing_train_data_', num2str(i)), '.txt');
            filename2 = strcat(strcat('C:/Users/mango/OneDrive/Documents/Uni/FYP/PuTTY logs/testing/training data/swing_train_data_', num2str(i)), '.txt');
        else
            filename2 = strcat(strcat('swing_train_data_', num2str(i)), '.txt');
        end
        % save each heel strike into seperate .txt files
        if im == 1
            imwrite(swing_shuffled_train(:, 1), strcat(filename(1:end-3), '.png'), 'PNG')
            filename2(end-3:end) = '.png'; filename(end-3:end) = '.png';
        else
            writematrix(swing_shuffled_train(:, 1), filename)
        end
        
% % %         train_data(train_count, 1) = filename;
% % %         train_data(train_count, 2) = 0;
        st_sw_train(train_count) = 1;             % 0 = stance, 1 = swing
        filename_list_train(train_count) = {filename2};
        train_count = train_count + 1;

    end


    %% testing dataset

    
    
    test_count = 1;
    test_data = zeros(length(stance_shuffled_train(1, :)), 2);
    
    % for loop saving a .txt file for each stance
    for i = 1:length(stance_shuffled_test(1, :))

        % create file name that increases over iterations
            filename = strcat(strcat('C:\Users\mango\OneDrive\Documents\Uni\FYP\PuTTY logs\testing\testing data\stance_test_data_', num2str(i)), '.txt');
        
        if file_directory == 1
            filename2 = strcat(strcat('C:\Users\mango\OneDrive\Documents\Uni\FYP\PuTTY logs\testing\testing data\stance_test_data_', num2str(i)), '.txt');
        else
            filename2 = strcat(strcat('stance_test_data_', num2str(i)), '.txt');
        end
        % save each heel strike into seperate .txt files
        if im == 1
            imwrite(stance_shuffled_test(:, 1), strcat(filename(1:end-3), '.png'), 'PNG')
            filename2(end-3:end) = '.png'; filename(end-3:end) = '.png';
        else
            writematrix(stance_shuffled_test(:, 1), filename)
        end
% % %         test_data(test_count, 1) = filename;
% % %         test_data(test_count, 2) = 0;
        st_sw_test(test_count) = 0;             % 0 = stance, 1 = swing
        filename_list_test(test_count) = {filename2};
        test_count = test_count + 1;

        %{
    % % %     writematrix(stance(:, 1), 'C:\Users\mango\OneDrive\Documents\Uni\FYP\PuTTY logs\testing\training data\gait_data_1.txt')
    % % %     directory = 'C:\Users\mango\OneDrive\Documents\Uni\FYP\PuTTY logs\testing\training data'; 
    % % % filename = 'data.xlsx';
    % % % writetable(C{p},fullfile(directory,filename));
        %}
    end


    % for loop saving a .txt file for each toe-off
    for i = 1:length(swing_shuffled_test(1, :))

        % create file name that increases over iterations
            filename = strcat(strcat('C:\Users\mango\OneDrive\Documents\Uni\FYP\PuTTY logs\testing\testing data\swing_test_data_', num2str(i)), '.txt');
        
        if file_directory == 1
            filename2 = strcat(strcat('C:\Users\mango\OneDrive\Documents\Uni\FYP\PuTTY logs\testing\testing data\swing_test_data_', num2str(i)), '.txt');
        else
            filename2 = strcat(strcat('swing_test_data_', num2str(i)), '.txt')
        end
        % save each heel strike into seperate .txt files
        if im == 1
            imwrite(swing_shuffled_test(:, 1), strcat(filename(1:end-3), '.png'), 'PNG')
            filename2(end-3:end) = '.png'; filename(end-3:end) = '.png';
        else
            writematrix(swing_shuffled_test(:, 1), filename)
        end
% % %         test_data(test_count, 1) = filename;
% % %         test_data(test_count, 2) = 1;
        st_sw_test(test_count) = 1;             % 0 = stance, 1 = swing
        filename_list_test(test_count) = {filename2};
        test_count = test_count + 1;

    end


    st_sw_train;
    files_train = table(filename_list_train', st_sw_train');
    writetable(files_train, 'C:\Users\mango\OneDrive\Documents\Uni\FYP\PuTTY logs\testing\files_train.csv');
    
    files_test = table(filename_list_test', st_sw_test');
    writetable(files_test, 'C:\Users\mango\OneDrive\Documents\Uni\FYP\PuTTY logs\testing\files_test.csv');
 



    
% size(swing_shuffled_train')
table1_train = [stance_shuffled_train, swing_shuffled_train]';
% [m, n] = size(table1_train);
stswsize = size(st_sw_train)

table1_train(1:end, 2:end+1) = table1_train;
table1_train(:, 1) = 1000*st_sw_train';
% % % table1_train(:, 1) = st_sw_train';

% beta1 = table1_train(4, :)

table1_test = [stance_shuffled_test, swing_shuffled_test]';
% [m, n] = size(table1_test);
table1_test(1:end, 2:end+1) = table1_test;
table1_test(:, 1) = 1000*st_sw_test';
% % % table1_test(:, 1) = st_sw_test';

% st_sw_train

%{

% % function table_out = row_shuffle(table_in)

% % % % % % % % % % % % % table1_1 = row_shuffle([table1_train; table1_test]);
% % % % % % % % % % % % % [m1, n1] = size(table1_1)
% % % % % % % % % % % % % 
% % % % % % % % % % % % % table1_train = table1_1(1:round(train_percent*m1), 1:end);
% % % % % % % % % % % % % table1_test = table1_1(round(train_percent*m1+1:end), 1:end);


% % % table1_train = row_shuffle(table1_train);
% % % table1_test = row_shuffle(table1_test);


% % table1_test

%     files_train = table(st_sw_train', stance_shuffled_train(1, :));
% %     files_train = table(table1_train);
% %     writetable(table1_train, 'C:\Users\mango\OneDrive\Documents\Uni\FYP\PuTTY logs\testing\files_train_all.csv');
%}

delete 'C:\Users\mango\OneDrive\Documents\Uni\FYP\PuTTY logs\testing\files_train_all.csv'
writematrix(table1_train, 'C:\Users\mango\OneDrive\Documents\Uni\FYP\PuTTY logs\testing\files_train_all.csv');

% export .xlsx file for matlab analysis code
delete 'C:\Users\mango\OneDrive\Documents\Uni\FYP\PuTTY logs\testing\files_train_all.xlsx'
writematrix(table1_train, 'C:\Users\mango\OneDrive\Documents\Uni\FYP\PuTTY logs\testing\files_train_all.xlsx');

%{
%     files_train = table(st_sw_train', stance_shuffled_train(1, :));
% %     files_test = table(table1_test);
% %     writetable(table1_test, 'C:\Users\mango\OneDrive\Documents\Uni\FYP\PuTTY logs\testing\files_test_all.csv');
%}

delete 'C:\Users\mango\OneDrive\Documents\Uni\FYP\PuTTY logs\testing\files_test_all.csv';
    writematrix(table1_test, 'C:\Users\mango\OneDrive\Documents\Uni\FYP\PuTTY logs\testing\files_test_all.csv');
    
% export .xlsx file for matlab analysis code
delete 'C:\Users\mango\OneDrive\Documents\Uni\FYP\PuTTY logs\testing\files_test_all.xlsx';
    writematrix(table1_test, 'C:\Users\mango\OneDrive\Documents\Uni\FYP\PuTTY logs\testing\files_test_all.xlsx');
    
% a =43

% stance_shuffled_train






end
end



function table_out = row_shuffle(table_in)


[m, n] = size(table_in);

table_out = zeros(m, n);

a = 1:m;    %create vector of values for each row

for i = 1:length(a)
    
    rand_val = round(rand()*(m-i) + 1);
    table_out(i,  1:n) = table_in(rand_val, 1:end);
end



end

















