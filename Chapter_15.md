---
title: "Chapter_15"
author: "Sam Worthy"
date: "2024-03-17"
output: 
  html_document: 
    keep_md: yes
---

## Screening many models

### 15.1 Modeling Concrete Mixture Strength

Using the concrete mixture data from Applied Predictive Modeling. Chapter 10 of that book demostrated models to predict the compressive strength of conrete mixtures using the ingredients as predictors. A wide variety of models were evaluated with different predictor sets and preprocessing needs. How can workflow sets make such a process of large scale testing for models easier?

Let's define the data splitting and resampling schemes:


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
## • Use tidymodels_prefer() to resolve common conflicts.
```

```r
tidymodels_prefer()
data(concrete, package = "modeldata")
glimpse(concrete)
```

```
## Rows: 1,030
## Columns: 9
## $ cement               <dbl> 540.0, 540.0, 332.5, 332.5, 198.6, 266.0, 380.0, …
## $ blast_furnace_slag   <dbl> 0.0, 0.0, 142.5, 142.5, 132.4, 114.0, 95.0, 95.0,…
## $ fly_ash              <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0…
## $ water                <dbl> 162, 162, 228, 228, 192, 228, 228, 228, 228, 228,…
## $ superplasticizer     <dbl> 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,…
## $ coarse_aggregate     <dbl> 1040.0, 1055.0, 932.0, 932.0, 978.4, 932.0, 932.0…
## $ fine_aggregate       <dbl> 676.0, 676.0, 594.0, 594.0, 825.5, 670.0, 594.0, …
## $ age                  <int> 28, 28, 270, 365, 360, 90, 365, 28, 28, 28, 90, 2…
## $ compressive_strength <dbl> 79.99, 61.89, 40.27, 41.05, 44.30, 47.03, 43.70, …
```

The compressive_strength column is the outcome. The age predictor tells us the age of the concrete sample at testing in days (concrete strengthens over time) and the rest of the predictors like cement and water are concrete components in units of kilograms per cubic meter.

For some cases in this data set, the same concrete formula was tested multiple times. To address this, we will use the mean compressive strength per concrete mixture for modeling.


```r
concrete <- 
   concrete %>% 
   group_by(across(-compressive_strength)) %>% 
   summarize(compressive_strength = mean(compressive_strength),
             .groups = "drop")
nrow(concrete)
```

```
## [1] 992
```

Let's split the data using the default 3:1 ratio of training-to-test and resample the training set using five repeats of 10-fold cross validation:


```r
set.seed(1501)
concrete_split <- initial_split(concrete, strata = compressive_strength)
concrete_train <- training(concrete_split)
concrete_test  <- testing(concrete_split)

set.seed(1502)
concrete_folds <- 
   vfold_cv(concrete_train, strata = compressive_strength, repeats = 5)
```

Some models (notably neural networks, KNN, and support vector machines) require predictors that have been centered and scaled, so some model workflows will require recipes with these preprocessing steps. For other models, a traditional response surface design model expansion (i.e., quadratic and two-way interactions) is a good idea. 

For these purposes, we create two recipes:


```r
normalized_rec <- 
   recipe(compressive_strength ~ ., data = concrete_train) %>% 
   step_normalize(all_predictors()) 

poly_recipe <- 
   normalized_rec %>% 
   step_poly(all_predictors()) %>% 
   step_interact(~ all_predictors():all_predictors())
```

For the models, we use the parsnip addin to create a set of model specifications.


```r
library(rules)
library(baguette)

linear_reg_spec <- 
   linear_reg(penalty = tune(), mixture = tune()) %>% 
   set_engine("glmnet")

nnet_spec <- 
   mlp(hidden_units = tune(), penalty = tune(), epochs = tune()) %>% 
   set_engine("nnet", MaxNWts = 2600) %>% 
   set_mode("regression")

mars_spec <- 
   mars(prod_degree = tune()) %>%  #<- use GCV to choose terms
   set_engine("earth") %>% 
   set_mode("regression")

svm_r_spec <- 
   svm_rbf(cost = tune(), rbf_sigma = tune()) %>% 
   set_engine("kernlab") %>% 
   set_mode("regression")

svm_p_spec <- 
   svm_poly(cost = tune(), degree = tune()) %>% 
   set_engine("kernlab") %>% 
   set_mode("regression")

knn_spec <- 
   nearest_neighbor(neighbors = tune(), dist_power = tune(), weight_func = tune()) %>% 
   set_engine("kknn") %>% 
   set_mode("regression")

cart_spec <- 
   decision_tree(cost_complexity = tune(), min_n = tune()) %>% 
   set_engine("rpart") %>% 
   set_mode("regression")

bag_cart_spec <- 
   bag_tree() %>% 
   set_engine("rpart", times = 50L) %>% 
   set_mode("regression")

rf_spec <- 
   rand_forest(mtry = tune(), min_n = tune(), trees = 1000) %>% 
   set_engine("ranger") %>% 
   set_mode("regression")

xgb_spec <- 
   boost_tree(tree_depth = tune(), learn_rate = tune(), loss_reduction = tune(), 
              min_n = tune(), sample_size = tune(), trees = tune()) %>% 
   set_engine("xgboost") %>% 
   set_mode("regression")

cubist_spec <- 
   cubist_rules(committees = tune(), neighbors = tune()) %>% 
   set_engine("Cubist") 
```

The analysis specifies that the neural network should have up to 27 hidden units in the layer. The extract_parameter_set_dials() function extracts the parameter set, which we modify to have the correct parameter range:


```r
nnet_param <- 
   nnet_spec %>% 
   extract_parameter_set_dials() %>% 
   update(hidden_units = hidden_units(c(1, 27)))
```

### 15.2 Creating the workflow set

Workflow sets take named lists of preprocessors and model specifications and combine them into an object containing multiple workflows. There are three possible kinds of preprocessors:

1. A standard R forumula
2. A recipe object (prior to estimation/prepping)
3. a dplyr-style selector to choose the outcome and predictors

As a first workflow set example, let's combine the recipe that only standardizes the predictors to the nonlinear models that requires the predictors to be in the same units.


```r
normalized <- 
   workflow_set(
      preproc = list(normalized = normalized_rec), 
      models = list(SVM_radial = svm_r_spec, SVM_poly = svm_p_spec, 
                    KNN = knn_spec, neural_network = nnet_spec)
   )
normalized
```

```
## # A workflow set/tibble: 4 × 4
##   wflow_id                  info             option    result    
##   <chr>                     <list>           <list>    <list>    
## 1 normalized_SVM_radial     <tibble [1 × 4]> <opts[0]> <list [0]>
## 2 normalized_SVM_poly       <tibble [1 × 4]> <opts[0]> <list [0]>
## 3 normalized_KNN            <tibble [1 × 4]> <opts[0]> <list [0]>
## 4 normalized_neural_network <tibble [1 × 4]> <opts[0]> <list [0]>
```

Since there is only a single preprocessor, this function creates a set of workflows with this value. If the preprocessor contained more than one entry, the function would create all combinations of preprocessors and models. 

The wflow_id column is automatically created but can be modified using a call to mutate(). The info column contains a tibble with some identifiers and the workflow object. The workflow can be extracted:


```r
normalized %>% extract_workflow(id = "normalized_KNN")
```

```
## ══ Workflow ════════════════════════════════════════════════════════════════════
## Preprocessor: Recipe
## Model: nearest_neighbor()
## 
## ── Preprocessor ────────────────────────────────────────────────────────────────
## 1 Recipe Step
## 
## • step_normalize()
## 
## ── Model ───────────────────────────────────────────────────────────────────────
## K-Nearest Neighbor Model Specification (regression)
## 
## Main Arguments:
##   neighbors = tune()
##   weight_func = tune()
##   dist_power = tune()
## 
## Computational engine: kknn
```

The option column is a placeholder for any arguments to use when we evaluate the workflow. For example, to add the neural network parameter object:


```r
normalized <- 
   normalized %>% 
   option_add(param_info = nnet_param, id = "normalized_neural_network")
normalized
```

```
## # A workflow set/tibble: 4 × 4
##   wflow_id                  info             option    result    
##   <chr>                     <list>           <list>    <list>    
## 1 normalized_SVM_radial     <tibble [1 × 4]> <opts[0]> <list [0]>
## 2 normalized_SVM_poly       <tibble [1 × 4]> <opts[0]> <list [0]>
## 3 normalized_KNN            <tibble [1 × 4]> <opts[0]> <list [0]>
## 4 normalized_neural_network <tibble [1 × 4]> <opts[1]> <list [0]>
```

Whne a function from the tune or finetune package is used to tune (or resample) the workflow, this argument will be used.

The result column is a placeholder for the output of the tuning or resampling functions. 

For the other nonlinear models, let's create another workflow set that uses dplyr selectors for the outcome and predictors:


```r
model_vars <- 
   workflow_variables(outcomes = compressive_strength, 
                      predictors = everything())

no_pre_proc <- 
   workflow_set(
      preproc = list(simple = model_vars), 
      models = list(MARS = mars_spec, CART = cart_spec, CART_bagged = bag_cart_spec,
                    RF = rf_spec, boosting = xgb_spec, Cubist = cubist_spec)
   )
no_pre_proc
```

```
## # A workflow set/tibble: 6 × 4
##   wflow_id           info             option    result    
##   <chr>              <list>           <list>    <list>    
## 1 simple_MARS        <tibble [1 × 4]> <opts[0]> <list [0]>
## 2 simple_CART        <tibble [1 × 4]> <opts[0]> <list [0]>
## 3 simple_CART_bagged <tibble [1 × 4]> <opts[0]> <list [0]>
## 4 simple_RF          <tibble [1 × 4]> <opts[0]> <list [0]>
## 5 simple_boosting    <tibble [1 × 4]> <opts[0]> <list [0]>
## 6 simple_Cubist      <tibble [1 × 4]> <opts[0]> <list [0]>
```

Finally, we assemble the set that uses nonlinear terms and interactions with the appropriate models:


```r
with_features <- 
   workflow_set(
      preproc = list(full_quad = poly_recipe), 
      models = list(linear_reg = linear_reg_spec, KNN = knn_spec)
   )
```

These objects are tibbles with the extra class of workflow_set. Row bindings does not affect the state of the sets and the result is itself a workflow set:


```r
all_workflows <- 
   bind_rows(no_pre_proc, normalized, with_features) %>% 
   # Make the workflow ID's a little more simple: 
   mutate(wflow_id = gsub("(simple_)|(normalized_)", "", wflow_id))
all_workflows
```

```
## # A workflow set/tibble: 12 × 4
##    wflow_id             info             option    result    
##    <chr>                <list>           <list>    <list>    
##  1 MARS                 <tibble [1 × 4]> <opts[0]> <list [0]>
##  2 CART                 <tibble [1 × 4]> <opts[0]> <list [0]>
##  3 CART_bagged          <tibble [1 × 4]> <opts[0]> <list [0]>
##  4 RF                   <tibble [1 × 4]> <opts[0]> <list [0]>
##  5 boosting             <tibble [1 × 4]> <opts[0]> <list [0]>
##  6 Cubist               <tibble [1 × 4]> <opts[0]> <list [0]>
##  7 SVM_radial           <tibble [1 × 4]> <opts[0]> <list [0]>
##  8 SVM_poly             <tibble [1 × 4]> <opts[0]> <list [0]>
##  9 KNN                  <tibble [1 × 4]> <opts[0]> <list [0]>
## 10 neural_network       <tibble [1 × 4]> <opts[1]> <list [0]>
## 11 full_quad_linear_reg <tibble [1 × 4]> <opts[0]> <list [0]>
## 12 full_quad_KNN        <tibble [1 × 4]> <opts[0]> <list [0]>
```

### 15.3 Tuning and evaluating the models

Almost all of the members of all_workflows contain tuning parameters. To evaluate their performance, we can use the standard tuning or resampling functions (e.g. tune_grid()). The workflow_map() function will apply the same function to all of the workflows in the set; the default is tune_grid(). 

For this example, grid search is applied to each workflow using up to 25 different parameter candidates. There are a set of common options to use with each execution of tune_grid(). For example, in the following code we will use the same resampling and control objects for each workflow, along with a grid size of 25. The workflow_map() function has an additonal argument called seed, which is used to ensure that each execution of tune_grid() consumes the same random numbers. 


```r
grid_ctrl <-
   control_grid(
      save_pred = TRUE,
      parallel_over = "everything",
      save_workflow = TRUE
   )

grid_results <-
   all_workflows %>%
   workflow_map(
      seed = 1503,
      resamples = concrete_folds,
      grid = 25,
      control = grid_ctrl
   )
```

The results show that the option and result columns have been updated.


```r
grid_results
```

The option column now contains all of the options that we used in the workflow_map() call. This makes our results reproducible. In the result columns, the "tune[+]" and rsmp[+] notations mean that the object had no issues. A value such as tune[x] occurs if all the models failed for some reason.

There are a few convenience functions examining results such as grid_results. The rank_results() function will order the models by some performance metric. By default, it uses the first metric in the metric set (RMSE in this instance). Let's filter to look only at RMSE:


```r
grid_results %>% 
   rank_results() %>% 
   filter(.metric == "rmse") %>% 
   select(model, .config, rmse = mean, rank)
```

Also by default, the function ranks all of the candidate sets; that's why the same model can show up multiple times in the output. An option called select_best can be used to rank the models using their best tuning parameter combination.

The autoplot() method plots the rankings; it also has a select_best argument. 


```r
autoplot(
   grid_results,
   rank_metric = "rmse",  # <- how to order models
   metric = "rmse",       # <- which metric to visualize
   select_best = TRUE     # <- one point per workflow
) +
   geom_text(aes(y = mean - 1/2, label = wflow_id), angle = 90, hjust = 1) +
   lims(y = c(3.5, 9.5)) +
   theme(legend.position = "none")
```

In case you want to see the tuning parameter results for a specific model. the id argument can take a single value from the wflow_id column for which model to plot.


```r
autoplot(grid_results, id = "Cubist", metric = "rmse")
```

There are also methods for collection_predictions() and collect_metrics().

The example model screening with our concrete mixture data fits a total of 12,600 models. Using 2 workers in parallel, the estimation process took 1.9 hours to complete.

### 15.4 Efficiently screening models

One effective method for screening a large set of models efficiently is to use the racing approach. With a workflow set, we can use the workflow_map() function for this racing approach. Recall that after we pipe in our workflow set, the argument we use is the function to apply to the workflows; in this case, we can use a value of "tune_race_anova". We also pass an appropriate control object; otherwise the options would be the same as the code in the previous section.


```r
library(finetune)

race_ctrl <-
   control_race(
      save_pred = TRUE,
      parallel_over = "everything",
      save_workflow = TRUE
   )

race_results <-
   all_workflows %>%
   workflow_map(
      "tune_race_anova",
      seed = 1503,
      resamples = concrete_folds,
      grid = 25,
      control = race_ctrl
   )
```

The new object looks very similar, although the elements of the result oclumn show a value of race[+] indicating a different type of object.


```r
race_results
```

The same helpful functions are available for this object to interrogate the results and in fact the basic autoplot() method.


```r
autoplot(
   race_results,
   rank_metric = "rmse",  
   metric = "rmse",       
   select_best = TRUE    
) +
   geom_text(aes(y = mean - 1/2, label = wflow_id), angle = 90, hjust = 1) +
   lims(y = c(3.0, 9.5)) +
   theme(legend.position = "none")
```

Overall, the racing approach estimated a total of 1,050 models, 8.33% of the full set of 12,600 models in the full grid. As a result, the racing approach was 4.8-fold faster.

For both objects, we rank the results, merge them, and plot them against one another.


```r
matched_results <- 
   rank_results(race_results, select_best = TRUE) %>% 
   select(wflow_id, .metric, race = mean, config_race = .config) %>% 
   inner_join(
      rank_results(grid_results, select_best = TRUE) %>% 
         select(wflow_id, .metric, complete = mean, 
                config_complete = .config, model),
      by = c("wflow_id", ".metric"),
   ) %>%  
   filter(.metric == "rmse")

library(ggrepel)

matched_results %>% 
   ggplot(aes(x = complete, y = race)) + 
   geom_abline(lty = 3) + 
   geom_point() + 
   geom_text_repel(aes(label = model)) +
   coord_obs_pred() + 
   labs(x = "Complete Grid RMSE", y = "Racing RMSE") 
```

While the racing approach selected the same candidate parameters as the complete grid for only 41.67% of the models, the performance metrics of the models selected by racing were nearly equal. The correlation of RMSE values was 0.968 and the rank correlation was 0.951. This indicates that, within a model, there were multiple tuning paramter combinations that had nearly identical results.

#### 15.5 Finalizing a Model

The process of choosing the final model and fitting it on the training set is straightforward. The first step is to pick a workflow to finalize. Since the boosted tree model worked well, we'll extract that from the set, update the parameters with the numerically best setttings, and fit to the training set:


```r
best_results <- 
   race_results %>% 
   extract_workflow_set_result("boosting") %>% 
   select_best(metric = "rmse")
best_results

boosting_test_results <- 
   race_results %>% 
   extract_workflow("boosting") %>% 
   finalize_workflow(best_results) %>% 
   last_fit(split = concrete_split)
```

We can see the test set metrics results, and visualize the predictions


```r
collect_metrics(boosting_test_results)
```


```r
boosting_test_results %>% 
   collect_predictions() %>% 
   ggplot(aes(x = compressive_strength, y = .pred)) + 
   geom_abline(color = "gray50", lty = 2) + 
   geom_point(alpha = 0.5) + 
   coord_obs_pred() + 
   labs(x = "observed", y = "predicted")
```

