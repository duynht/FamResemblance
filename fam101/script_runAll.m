%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                  Kinship Recognition Toolbox
%       Copyright (C) Jun 2014 Center for Research in Computer Vision
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This toolbox was created to foster research in kinship recognition. If you
% use any of the algorithms or datasets please cite the relevant literature:
%
%  1)  A. Dehghan, E.G. Ortiz, R. Villegas, and M. Shah. "Who Do I Look Like?
%  Determining Parent-Offspring Resemblance via Genetic Features." IEEE CVPR
%  2013.
%
%  2) R. Memisevic. "Learning to relate images." IEEE TPAMI, 2013.
%
%  3) J. Lu, X. Zhou, Y.-P. Tan, Y. Shang, and J. Zhou. "Neighborhood
%  Repulsed Metric Learning for Kinship Verification." IEEE TPAMI, 2013.
%
%  4) R. Fang, A. C. Gallagher, T. Chen, and A. Loui. "Kinship
%  Classification by Modeling Facial Feature Heredity." ICIP, 2013.
%
% This toolbox performs both kinship verification (KinFace) and
% identification (Fam101).
%
% script_runAll: Runs entire experimental pipeline from feature extraction
% to experimentation.
%
% Download Fam101 Datasets: http://goo.gl/k8GNGy
%
% Change all ~Paths.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For more information, see http://enriquegortiz.com/fbfaces.
%
% Contact: Enrique G. Ortiz (eortiz@cs.ucf.edu)
%          Afshin Dehghan (adehghan@cs.ucf.edu)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all; clc;
addpath(genpath('../common'));

% Process RAW Fam101 Data and Extract Features for all Data
% Specify Fam101 Dataset Location after download.
dataPath = '../Dataset/Family101_150x120';
outPath = './test_toolbox/fam101_feat';
prepareFam101;

% Create Training/Testing Splits
% Path to data processed in previous stage
dataPath = './test_toolbox/fam101_feat';
% Output Path for Train Data
outPath = './test_toolbox/traindata';
% Output Path for Test Data
outPath2 = './test_toolbox/testdata';
% Final Location of Combined Training/Testing Data
resDir = './data';
trainTestSplits;

% Train Models
% Results Directory
dataPath = './data';
resDir = './results';
% Training files for parent-offspring pairings
files = {'fs_train_p16.mat','md_train_p16.mat','fd_train_p16.mat','ms_train_p16.mat'};
trainModels;

% Test Models
% Testing files for parent-offspring pairings
files2 = {'fs_test_p16.mat','md_test_p16.mat','fd_test_p16.mat','ms_test_p16.mat'};
testModels;

% Output Graphs
tPath = resDir;
splits = {'fs','fd','ms','md'};
processResults;