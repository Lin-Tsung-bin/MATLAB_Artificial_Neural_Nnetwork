function R2=rs_new(y,yhat)
%RS_NEW  R-square statistic.
%
%        Syntax
%
%          R2=RS_NEW(Y,Yhat)
%
%          Y    表示為實驗值。
%          Yhat 表示為回歸模式預測值。
RESSS = norm(y-yhat)^2;   % Residual sum of squares.
TSS = norm(y-mean(y))^2;  % Total sum of squares.
REGSS = norm(yhat)^2;     % Regression sum of squares.
R2=1-RESSS/TSS;           % R-square statistic.
