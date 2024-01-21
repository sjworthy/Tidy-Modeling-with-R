---
title: "Chapter_9"
author: "Sam Worthy"
date: "2024-01-21"
output:
  html_document:
    keep_md: yes
---

Code from previous chapters


```r
library(tidymodels)
```

```
## ── Attaching packages ────────────────────────────────────── tidymodels 1.1.1 ──
```

```
## ✔ broom        1.0.5     ✔ recipes      1.0.8
## ✔ dials        1.2.0     ✔ rsample      1.2.0
## ✔ dplyr        1.1.3     ✔ tibble       3.2.1
## ✔ ggplot2      3.4.4     ✔ tidyr        1.3.0
## ✔ infer        1.0.5     ✔ tune         1.1.2
## ✔ modeldata    1.2.0     ✔ workflows    1.1.3
## ✔ parsnip      1.1.1     ✔ workflowsets 1.0.1
## ✔ purrr        1.0.2     ✔ yardstick    1.2.0
```

```
## ── Conflicts ───────────────────────────────────────── tidymodels_conflicts() ──
## ✖ purrr::discard() masks scales::discard()
## ✖ dplyr::filter()  masks stats::filter()
## ✖ dplyr::lag()     masks stats::lag()
## ✖ recipes::step()  masks stats::step()
## • Use suppressPackageStartupMessages() to eliminate package startup messages
```

```r
data(ames)
ames <- mutate(ames, Sale_Price = log10(Sale_Price))

set.seed(502)
ames_split <- initial_split(ames, prop = 0.80, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test  <-  testing(ames_split)

ames_rec <- 
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + 
           Latitude + Longitude, data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>% 
  step_other(Neighborhood, threshold = 0.01) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) %>% 
  step_ns(Latitude, Longitude, deg_free = 20)
  
lm_model <- linear_reg() %>% set_engine("lm")

lm_wflow <- 
  workflow() %>% 
  add_model(lm_model) %>% 
  add_recipe(ames_rec)

lm_fit <- fit(lm_wflow, ames_train)
```
## Judging Model Effectiveness

Focus of tidymodels is on empirical validation; this usually means using data that were not used to create the model as the substrate to measure effectiveness.

Best approach to empirical validation involves using resampling methods that will be introduced in Chapter 10.

Two common metrics for regression models:
1. root mean square error (RMSE) - measures accuracy
2. coefficient of determination (R-squared) - measures correlation

A models optimized for RMSE has more variability but relatively uniform accuracy across the range of the outcome. R-squared optimization has tighter correlation between observed and predicted values but model performs poorly in the tails.

Chapter focus is the yardstick package: a core tidymodels package with the focus of measuring model performance. 

### 9.1 Performance metrics and inference

The effectiveness of any given model depends on how the model will be used. 

* An inferential model is used primarily to understand relationships, and typically emphasizes the choice (and validity) of probabilistic distributions and other generative qualities that define the model.
* For a prediction model, predictive strength is of primary importance and other concerns about underlying statistical qualities may be less important. Predictive strength usually determined by how close our predictions come to the observed data. 

Advice: for those developing inferential models, use these techniques even when the model will not be used with the primary goal of prediction.

### 9.2 Regression Metrics

Functions in yardstick package are data frame-based with the general syntax of:


```r
function(data, truth, ...)
```

data is a data frame or tibble and truth is the column with the observed outcome values. 

Illustration from Section 8.8.
Model lm_wflow_fit combines a linear regression model with a predictor set supplemented with an interaction and spline functions for longitude and latitude. 


```r
ames_test_res <- predict(lm_fit, new_data = ames_test %>% select(-Sale_Price))
ames_test_res
```

```
## # A tibble: 588 × 1
##    .pred
##    <dbl>
##  1  5.07
##  2  5.31
##  3  5.28
##  4  5.33
##  5  5.30
##  6  5.24
##  7  5.67
##  8  5.52
##  9  5.34
## 10  5.00
## # ℹ 578 more rows
```

The predicted numeric outcome from the regression model is named .pred. Match the predicted values with their corresponding observed outcome values.


```r
ames_test_res <- bind_cols(ames_test_res, ames_test %>% select(Sale_Price))
ames_test_res
```

```
## # A tibble: 588 × 2
##    .pred Sale_Price
##    <dbl>      <dbl>
##  1  5.07       5.02
##  2  5.31       5.39
##  3  5.28       5.28
##  4  5.33       5.28
##  5  5.30       5.28
##  6  5.24       5.26
##  7  5.67       5.73
##  8  5.52       5.60
##  9  5.34       5.32
## 10  5.00       4.98
## # ℹ 578 more rows
```

It is best practice to analyze the predictions on the transformed scale even if the predictions are reported using the original units. 


```r
ggplot(ames_test_res, aes(x = Sale_Price, y = .pred)) + 
  # Create a diagonal line:
  geom_abline(lty = 2) + 
  geom_point(alpha = 0.5) + 
  labs(y = "Predicted Sale Price (log10)", x = "Sale Price (log10)") +
  # Scale and size the x- and y-axis uniformly:
  coord_obs_pred()
```

![](Chapter_9_files/figure-html/unnamed-chunk-5-1.png)<!-- -->

Compute RMSE:


```r
rmse(ames_test_res, truth = Sale_Price, estimate = .pred)
```

```
## # A tibble: 1 × 3
##   .metric .estimator .estimate
##   <chr>   <chr>          <dbl>
## 1 rmse    standard      0.0736
```

This shows the standard format of the output of yardstick functions. Metrics for numeric outcomes usually have a value of "standard" for the .estimator column. 

To compute multiple metrics at once, can create a metric set. mae is mean absolute error.


```r
ames_metrics <- metric_set(rmse, rsq, mae)
ames_metrics(ames_test_res, truth = Sale_Price, estimate = .pred)
```

```
## # A tibble: 3 × 3
##   .metric .estimator .estimate
##   <chr>   <chr>          <dbl>
## 1 rmse    standard      0.0736
## 2 rsq     standard      0.836 
## 3 mae     standard      0.0549
```

The yardstick package does NOT contain a function for adjusted R-squared. This modification of the coefficient of determination is commonly used when the same data used to fit the model are used to evaluate the model. 

### 9.3 Binary Classification Metrics

The modeldata package contains example predictions from a test data set with two classes, Class1 and Class2. The second and third columns are the predicted class probabilities for the test set while predicted are the discrete predictions.


```r
data(two_class_example)
tibble(two_class_example)
```

```
## # A tibble: 500 × 4
##    truth   Class1   Class2 predicted
##    <fct>    <dbl>    <dbl> <fct>    
##  1 Class2 0.00359 0.996    Class2   
##  2 Class1 0.679   0.321    Class1   
##  3 Class2 0.111   0.889    Class2   
##  4 Class1 0.735   0.265    Class1   
##  5 Class2 0.0162  0.984    Class2   
##  6 Class1 0.999   0.000725 Class1   
##  7 Class1 0.999   0.000799 Class1   
##  8 Class1 0.812   0.188    Class1   
##  9 Class2 0.457   0.543    Class2   
## 10 Class2 0.0976  0.902    Class2   
## # ℹ 490 more rows
```

For the hard class predictions, a variety of yardstick functions are helpful:


```r
# A confusion matrix
conf_mat(two_class_example, truth = truth, estimate = predicted)
```

```
##           Truth
## Prediction Class1 Class2
##     Class1    227     50
##     Class2     31    192
```


```r
# Accuracy
accuracy(two_class_example, truth, predicted)
```

```
## # A tibble: 1 × 3
##   .metric  .estimator .estimate
##   <chr>    <chr>          <dbl>
## 1 accuracy binary         0.838
```


```r
# Matthews correlation coefficient:
mcc(two_class_example, truth, predicted)
```

```
## # A tibble: 1 × 3
##   .metric .estimator .estimate
##   <chr>   <chr>          <dbl>
## 1 mcc     binary         0.677
```


```r
# F1 metric:
f_meas(two_class_example, truth, predicted)
```

```
## # A tibble: 1 × 3
##   .metric .estimator .estimate
##   <chr>   <chr>          <dbl>
## 1 f_meas  binary         0.849
```

Combining these three classification metrics together


```r
classification_metrics <- metric_set(accuracy, mcc, f_meas)
classification_metrics(two_class_example, truth = truth, estimate = predicted)
```

```
## # A tibble: 3 × 3
##   .metric  .estimator .estimate
##   <chr>    <chr>          <dbl>
## 1 accuracy binary         0.838
## 2 mcc      binary         0.677
## 3 f_meas   binary         0.849
```

The Matthews correlation coefficient and F1 score both summarize the confusion matrix, but compared to mcc(), which measures the quality of both positive and negative examples, the f_meas() metric emphasizes the positive class, i.e. the event of interest. Yardstick functions have a standard argument called event_level to distinguish positive and negative levels. The default is that the first level of the outcome factor is the event of interest.

As an example where the second level is the event:


```r
f_meas(two_class_example, truth, predicted, event_level = "second")
```

```
## # A tibble: 1 × 3
##   .metric .estimator .estimate
##   <chr>   <chr>          <dbl>
## 1 f_meas  binary         0.826
```

There are numerous classification metrics that use the predicted probabilities as inputs rather than the hard class predictions. For example, the receiver operating characteristic (ROC) curve computes the sensitivity and specificity over a continuum of different event thresholds. The predicted class column is not used. There are two yardstick functions for this method: roc_curve() computes the data points that make up the ROC curve and roc_auc() computes the area under the curve. The ... placeholder is used to pass the appropriate class probability column.


```r
two_class_curve <- roc_curve(two_class_example, truth, Class1)
two_class_curve
```

```
## # A tibble: 502 × 3
##    .threshold specificity sensitivity
##         <dbl>       <dbl>       <dbl>
##  1 -Inf           0                 1
##  2    1.79e-7     0                 1
##  3    4.50e-6     0.00413           1
##  4    5.81e-6     0.00826           1
##  5    5.92e-6     0.0124            1
##  6    1.22e-5     0.0165            1
##  7    1.40e-5     0.0207            1
##  8    1.43e-5     0.0248            1
##  9    2.38e-5     0.0289            1
## 10    3.30e-5     0.0331            1
## # ℹ 492 more rows
```


```r
roc_auc(two_class_example, truth, Class1)
```

```
## # A tibble: 1 × 3
##   .metric .estimator .estimate
##   <chr>   <chr>          <dbl>
## 1 roc_auc binary         0.939
```

The two_class_curve object can be used in ggplot to visualize the curve. There is an autoplot() method that will take care of the details.


```r
autoplot(two_class_curve)
```

![](Chapter_9_files/figure-html/unnamed-chunk-17-1.png)<!-- -->

If the curve was close to the diagonal line, then the model's predictions would be no better than random guessing. Since the curve is up in the top, left-hand corner, we see that our model performs well at different thresholds. 

### 9.4 Multiclass Classification Metrics

Considering data with three or more classes.


```r
data(hpc_cv)
tibble(hpc_cv)
```

```
## # A tibble: 3,467 × 7
##    obs   pred     VF      F       M          L Resample
##    <fct> <fct> <dbl>  <dbl>   <dbl>      <dbl> <chr>   
##  1 VF    VF    0.914 0.0779 0.00848 0.0000199  Fold01  
##  2 VF    VF    0.938 0.0571 0.00482 0.0000101  Fold01  
##  3 VF    VF    0.947 0.0495 0.00316 0.00000500 Fold01  
##  4 VF    VF    0.929 0.0653 0.00579 0.0000156  Fold01  
##  5 VF    VF    0.942 0.0543 0.00381 0.00000729 Fold01  
##  6 VF    VF    0.951 0.0462 0.00272 0.00000384 Fold01  
##  7 VF    VF    0.914 0.0782 0.00767 0.0000354  Fold01  
##  8 VF    VF    0.918 0.0744 0.00726 0.0000157  Fold01  
##  9 VF    VF    0.843 0.128  0.0296  0.000192   Fold01  
## 10 VF    VF    0.920 0.0728 0.00703 0.0000147  Fold01  
## # ℹ 3,457 more rows
```

Data set has factors for the observed and predicted outcomes along with four other column of predicted probabilities for each class. 

The functions for metrics that use the discrete class predictions are identical to their binary counterparts.


```r
# accuracy
accuracy(hpc_cv, obs, pred)
```

```
## # A tibble: 1 × 3
##   .metric  .estimator .estimate
##   <chr>    <chr>          <dbl>
## 1 accuracy multiclass     0.709
```


```r
# matthews correlation coefficient
mcc(hpc_cv, obs, pred)
```

```
## # A tibble: 1 × 3
##   .metric .estimator .estimate
##   <chr>   <chr>          <dbl>
## 1 mcc     multiclass     0.515
```

There are methods for taking metrics designed to handle outcomes with only two classes and extend them for outcomes with more than two classes. For example, there are wrapper methods that can be used to apply sensitivity to our four-class outcome. The options are:

1. macro-averaging computes a set of one-versus-all metrics using the standard two-class statistics. These are averaged.
2. macro-weighted averaging does the same but the average is weighted by the number of samples in each class.
3. micro-averaging computes the contribution for each class, aggregates them, then computes a single metric from the aggregates. 

Using sensitivity as an example, the usual two-class calculation is the ratio of the number of correctly predicted events divided by the number of true events. The manual calculations for these averaging methods are:


```r
class_totals <- 
  count(hpc_cv, obs, name = "totals") %>% 
  mutate(class_wts = totals / sum(totals))
class_totals
```

```
##   obs totals  class_wts
## 1  VF   1769 0.51023940
## 2   F   1078 0.31093164
## 3   M    412 0.11883473
## 4   L    208 0.05999423
```


```r
cell_counts <- 
  hpc_cv %>% 
  group_by(obs, pred) %>% 
  count() %>% 
  ungroup()
```



```r
# Compute the four sensitivities using 1-vs-all
one_versus_all <- 
  cell_counts %>% 
  filter(obs == pred) %>% 
  full_join(class_totals, by = "obs") %>% 
  mutate(sens = n / totals)
one_versus_all
```

```
## # A tibble: 4 × 6
##   obs   pred      n totals class_wts  sens
##   <fct> <fct> <int>  <int>     <dbl> <dbl>
## 1 VF    VF     1620   1769    0.510  0.916
## 2 F     F       647   1078    0.311  0.600
## 3 M     M        79    412    0.119  0.192
## 4 L     L       111    208    0.0600 0.534
```


```r
# Three different estimates:
one_versus_all %>% 
  summarize(
    macro = mean(sens), 
    macro_wts = weighted.mean(sens, class_wts),
    micro = sum(n) / sum(totals)
  )
```

```
## # A tibble: 1 × 3
##   macro macro_wts micro
##   <dbl>     <dbl> <dbl>
## 1 0.560     0.709 0.709
```

Yardstick functions can automatically apply these methods via the estimator argument:


```r
sensitivity(hpc_cv, obs, pred, estimator = "macro")
```

```
## # A tibble: 1 × 3
##   .metric     .estimator .estimate
##   <chr>       <chr>          <dbl>
## 1 sensitivity macro          0.560
```


```r
sensitivity(hpc_cv, obs, pred, estimator = "macro_weighted")
```

```
## # A tibble: 1 × 3
##   .metric     .estimator     .estimate
##   <chr>       <chr>              <dbl>
## 1 sensitivity macro_weighted     0.709
```


```r
sensitivity(hpc_cv, obs, pred, estimator = "micro")
```

```
## # A tibble: 1 × 3
##   .metric     .estimator .estimate
##   <chr>       <chr>          <dbl>
## 1 sensitivity micro          0.709
```

When dealing with probability estimates, there are some metrics with multiclass analogs. Multiclass technique for ROC curves, all of the class probability columns must be given to the function.


```r
roc_auc(hpc_cv, obs, VF, F, M, L)
```

```
## # A tibble: 1 × 3
##   .metric .estimator .estimate
##   <chr>   <chr>          <dbl>
## 1 roc_auc hand_till      0.829
```

Macro-weighted averaging is also available as an option for applyign this metric to a multiclass outcome:


```r
roc_auc(hpc_cv, obs, VF, F, M, L, estimator = "macro_weighted")
```

```
## # A tibble: 1 × 3
##   .metric .estimator     .estimate
##   <chr>   <chr>              <dbl>
## 1 roc_auc macro_weighted     0.868
```

All of these performance metrics can be computed using dplyr groupings. Recall that these data have a column for the resampling groups. Notice how we can pass a grouped data frame to the metric function to compute the metrics for each group.


```r
hpc_cv %>% 
  group_by(Resample) %>% 
  accuracy(obs, pred)
```

```
## # A tibble: 10 × 4
##    Resample .metric  .estimator .estimate
##    <chr>    <chr>    <chr>          <dbl>
##  1 Fold01   accuracy multiclass     0.726
##  2 Fold02   accuracy multiclass     0.712
##  3 Fold03   accuracy multiclass     0.758
##  4 Fold04   accuracy multiclass     0.712
##  5 Fold05   accuracy multiclass     0.712
##  6 Fold06   accuracy multiclass     0.697
##  7 Fold07   accuracy multiclass     0.675
##  8 Fold08   accuracy multiclass     0.721
##  9 Fold09   accuracy multiclass     0.673
## 10 Fold10   accuracy multiclass     0.699
```

The groupings also translate to the autoplot() methods.


```r
# Four 1-vs-all ROC curves for each fold
hpc_cv %>% 
  group_by(Resample) %>% 
  roc_curve(obs, VF, F, M, L) %>% 
  autoplot
```

![](Chapter_9_files/figure-html/unnamed-chunk-31-1.png)<!-- -->

This visualization shows us that the different groups all perform about the same, but that the VF class is predicted better than the F or M classes, since VF ROC curves are more in the top-left corner. 

### Chapter Summary

1. Different metrics measure different aspects of a model fit.
2. Measuring model performance is important even when a given model will be not used primarily for prediction.
3. Functions from the yardstick package measure the effectiveness of a model using data.
4. The primary tidymodels interface uses tidyverse principles and data frames. 
5. Different metrics are appropriate for regression and classification metrics and, within these, there are sometimes different ways to estimate the statistics, such as for multiclass outcomes.
