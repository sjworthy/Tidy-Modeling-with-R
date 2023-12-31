---
title: "Chapter_8"
author: "Sam Worthy"
date: "2023-12-18"
output:
  html_document:
    keep_md: yes
---

Code from previous chapter 

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
## • Search for functions across packages at https://www.tidymodels.org/find/
```

```r
data(ames)

ames <- mutate(ames, Sale_Price = log10(Sale_Price))

set.seed(502)
ames_split <- initial_split(ames, prop = 0.80, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test  <-  testing(ames_split)
```


## Feature Engineering with recipes

Feature engineering entails reformatting predictor values to make them easier for a model to use effectively. Example: you have two predictors in a data set that can be more effectively represented in your model as a ratio. 

This chapter introduces the recipes package that can be used to combine different feature engineering and preprocessing tasks into a single object and then apply these transformations to different data sets. 

### 8.1 A simple recipe () for the Ames housing data

Fitting an ordinary linear regression model to the data


```r
lm(Sale_Price ~ Neighborhood + log10(Gr_Liv_Area) + Year_Built + Bldg_Type, data = ames)
```

```
## 
## Call:
## lm(formula = Sale_Price ~ Neighborhood + log10(Gr_Liv_Area) + 
##     Year_Built + Bldg_Type, data = ames)
## 
## Coefficients:
##                                         (Intercept)  
##                                          -0.8551717  
##                           NeighborhoodCollege_Creek  
##                                           0.0135485  
##                                NeighborhoodOld_Town  
##                                          -0.0289607  
##                                 NeighborhoodEdwards  
##                                          -0.0493174  
##                                NeighborhoodSomerset  
##                                           0.0499653  
##                      NeighborhoodNorthridge_Heights  
##                                           0.1335758  
##                                 NeighborhoodGilbert  
##                                          -0.0337334  
##                                  NeighborhoodSawyer  
##                                          -0.0042779  
##                          NeighborhoodNorthwest_Ames  
##                                           0.0004589  
##                             NeighborhoodSawyer_West  
##                                          -0.0174582  
##                                NeighborhoodMitchell  
##                                           0.0004695  
##                               NeighborhoodBrookside  
##                                          -0.0110205  
##                                NeighborhoodCrawford  
##                                           0.0914254  
##                  NeighborhoodIowa_DOT_and_Rail_Road  
##                                          -0.0839821  
##                              NeighborhoodTimberland  
##                                           0.0604062  
##                              NeighborhoodNorthridge  
##                                           0.0845868  
##                             NeighborhoodStone_Brook  
##                                           0.1459657  
## NeighborhoodSouth_and_West_of_Iowa_State_University  
##                                          -0.0282535  
##                             NeighborhoodClear_Creek  
##                                           0.0480071  
##                          NeighborhoodMeadow_Village  
##                                          -0.0899124  
##                               NeighborhoodBriardale  
##                                          -0.0465821  
##                     NeighborhoodBloomington_Heights  
##                                           0.0402528  
##                                 NeighborhoodVeenker  
##                                           0.0885538  
##                         NeighborhoodNorthpark_Villa  
##                                           0.0262051  
##                                 NeighborhoodBlueste  
##                                           0.0322372  
##                                  NeighborhoodGreens  
##                                           0.1751507  
##                             NeighborhoodGreen_Hills  
##                                           0.2229230  
##                                NeighborhoodLandmark  
##                                          -0.0119925  
##                                  log10(Gr_Liv_Area)  
##                                           0.6343996  
##                                          Year_Built  
##                                           0.0020678  
##                                   Bldg_TypeTwoFmCon  
##                                          -0.0312306  
##                                     Bldg_TypeDuplex  
##                                          -0.1038443  
##                                      Bldg_TypeTwnhs  
##                                          -0.0968859  
##                                     Bldg_TypeTwnhsE  
##                                          -0.0414929
```

When this function is executed, the data are converted from a data frame to a numeric design matrix (model matrix) and then the least squares method is used to estimate parameters. 

A recipe is also an object that defines a series of steps for data processing. The recipe defines the steps via step_*() functions without immediately executing them; it is only a specification of what should be done. 

Recipe equivalent to the previous formula


```r
library(tidymodels) # Includes the recipes package
tidymodels_prefer()

simple_ames <- 
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type,
         data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>% 
  step_dummy(all_nominal_predictors())
simple_ames
```

```
## 
```

```
## ── Recipe ──────────────────────────────────────────────────────────────────────
```

```
## 
```

```
## ── Inputs
```

```
## Number of variables by role
```

```
## outcome:   1
## predictor: 4
```

```
## 
```

```
## ── Operations
```

```
## • Log transformation on: Gr_Liv_Area
```

```
## • Dummy variables from: all_nominal_predictors()
```

Break down of this code:

1. The call to recipe() with a formula tells the recipe the roles of the "ingredients" or variables (e.g. predictor, outcome). It only uses the data ames_train to determine the data types for the columns.
2. step_log() declares that Gr_Liv_Area should be log transformed
3. step_dummy() specifices which variables should be converted from a qualitative format to a quantitative format, using dummy or indicator variables. An indicator or dummy variable is a binary numeric variable (a column of ones and zeros) that encodes qualitative information.

The function all_nomial_predictors() captures the names of any predictor columns that are currently factor or character (i.e. nominal) in nature. Other selectors specifics to the recipes package are: all_numeric_predictors(), all_numeric(), all_predictors(), and all_outcomes().

Advantages to using a recipe over a formula or raw predictors:

1. These computations can be recycled across models since they are not tightly coupled to the modeling function.
2. A recipe enables a broader set of data processing choices than formulas can offer.
3. The syntax can be very compact
4. All data processing can be captured in a single R object instead of in scripts that are repeated or even spread across different files. 

### 8.2 Using Recipes

Previous workflow code:


```r
lm_model <- linear_reg() %>% set_engine("lm")

lm_wflow <- 
  workflow() %>% 
  add_model(lm_model) %>% 
  add_variables(outcome = Sale_Price, predictors = c(Longitude, Latitude))

lm_fit <- fit(lm_wflow, ames_train)
```

To improve this approach with more complex feature engineering let's use the simple_ames recipe to preprocess data for modeling. 


```r
lm_wflow %>% 
  add_recipe(simple_ames)
```

Did not work because we can only have one preprocessing method at a time, so need to remove the existing preprocessor before adding the recipe. 


```r
lm_wflow <- 
  lm_wflow %>% 
  remove_variables() %>% 
  add_recipe(simple_ames)
lm_wflow
```

```
## ══ Workflow ════════════════════════════════════════════════════════════════════
## Preprocessor: Recipe
## Model: linear_reg()
## 
## ── Preprocessor ────────────────────────────────────────────────────────────────
## 2 Recipe Steps
## 
## • step_log()
## • step_dummy()
## 
## ── Model ───────────────────────────────────────────────────────────────────────
## Linear Regression Model Specification (regression)
## 
## Computational engine: lm
```

Let's estimate both the recipe and model using a simple call to fit()


```r
lm_fit <- fit(lm_wflow, ames_train)
```

The predict() metho applies the same preprocessing that was used on the training set to the new data before passing them along to the model's predict() method:


```r
predict(lm_fit, ames_test %>% slice(1:3))
```

```
## Warning in predict.lm(object = object$fit, newdata = new_data, type =
## "response", : prediction from a rank-deficient fit may be misleading
```

```
## # A tibble: 3 × 1
##   .pred
##   <dbl>
## 1  5.08
## 2  5.32
## 3  5.28
```

If we need the bare model object or recipe, there are extract_* functions that can retrieve them:


```r
lm_fit %>% 
  extract_recipe(estimated = TRUE)
```

```
## 
```

```
## ── Recipe ──────────────────────────────────────────────────────────────────────
```

```
## 
```

```
## ── Inputs
```

```
## Number of variables by role
```

```
## outcome:   1
## predictor: 4
```

```
## 
```

```
## ── Training information
```

```
## Training data contained 2342 data points and no incomplete rows.
```

```
## 
```

```
## ── Operations
```

```
## • Log transformation on: Gr_Liv_Area | Trained
```

```
## • Dummy variables from: Neighborhood, Bldg_Type | Trained
```


```r
lm_fit %>% 
  # This returns the parsnip object:
  extract_fit_parsnip() %>% 
  # Now tidy the linear model object:
  tidy() %>% 
  slice(1:5)
```

```
## # A tibble: 5 × 5
##   term                       estimate std.error statistic   p.value
##   <chr>                         <dbl>     <dbl>     <dbl>     <dbl>
## 1 (Intercept)                -0.669    0.231        -2.90 3.80e-  3
## 2 Gr_Liv_Area                 0.620    0.0143       43.2  2.63e-299
## 3 Year_Built                  0.00200  0.000117     17.1  6.16e- 62
## 4 Neighborhood_College_Creek  0.0178   0.00819       2.17 3.02e-  2
## 5 Neighborhood_Old_Town      -0.0330   0.00838      -3.93 8.66e-  5
```

### 8.3 How data are used by the recipe()

Data are passed to recipes at different stages. First, when calling recipe(..., data), the data set is used to determine the data types of each column so that selectors such as all_numeric() or all_numeric_predictors() can be used. Second, when preparing the data using fit(workflow, data), the training data are used for all estimation operations including a recipe that may be part of the workflow, from determining factor levels to computing PCA components and everything in between.

All preprocessing and feature engineering steps use ONLY the training data. 

Finally, when using predict(workflow, new_data), no model or preprocessor parameters like those from recipes are re-estimated using the values in new_data. Take centering and scaling using step_normalize() as an example. Using this step, the means and standard deviations from the appropriate columns are determined from the training set; new samples at prediction time are standardized using these values from training when predict() is invoked.

### 8.4 Examples of recipe steps

#### 8.1 Encoding qualitative data in a numeric format

One of the most common feature engineering tasks is transforming nominal or qualitative data (factors or characters) so that they can be encoded or represented numerically. Sometimes we can alter the factor levels of a qualitative clumn in helpful ways prior to such transformations. For example, step_unknown() can be used to change missing values to a dedicated factor level. Similarly, if we anticipate that new factor level may be encountered in future data, step_novel() can allot a new level for this purpose.

Additionally, step_other() can be used to analyze frequencies of the factor levels in the training set and convert infrequently occurring values to a catch-all level of "other", with a threshold that can be specified. A good example is the Neighborhood predictor. We see that two neighborhoods have less than 5 properties in the training data (Landmark and Green Hills). If we add step_other(Neighborhood, threshold = 0.01) to our recipe, the bottom 1% of the neighborhoods will be lumped into a new level called "other".


```r
simple_ames <- 
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type,
         data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>% 
  step_other(Neighborhood, threshold = 0.01) %>% 
  step_dummy(all_nominal_predictors())
```

The most common method for converting a factor predictor to a numeric format is to create dummy or indicator variables. For dummy variables, the single Bldg_Type column would be replaced with four numeric columns whose values are either zero or one. These binary variables represent specific factor level values. In R, the convention is to exclude a column for the first factor level. 

Why not all five? The most basic reason is simplicity; if you know the value of these four columns, you can determine the last value because there are mutually exclusive categories. More techinallly, the classical justification is that a number of models have numerical issues when there are linear dependencies between columns. If all five building type indicator columns are included, they would add up to the intercept column. This would cause an issue, or perhaps an outright error, in the underlying matrix algebra. 

The full set of encodings cna be used for some models, called one-hot encoding; one_hot argument of step_dummy().

One helpful feature of step_dummy() is that there is more control over how the resulting dummy variables are named.In base R, dummy variable names mash the variable name with the level, resulting in names like NeighborhoodVeenker. Recipes use an underscore as the separator between the name and level (Neighborhood_Veenker) and there is an option to use custom formatting for the names. The default naming convention in recipes makes it easier to capture those new columns in future steps using a selector; starts_with("Neighborhood_").

Traditional dummy variables require that all of the possible categories be known to create a full set of numeric features. Feature hashing methods only consider the value of the category to assign it to a predefined pool of dummy variables. Effect or likelihood encodings replace the original data with a single numeric column that measures the effect of those data. 

Different recipe steps behave differently when applied to variables in the data. For example, step_log() modifies a column in place without changing the name. Other steps, such as step_dummy() eliminate the original data column and replace it with one or more columns with different names. 

#### 8.2 Interaction terms

Interactions effects involve two or more predictors. Numerically, an interaction term between predictors is encoded as their product. Interactions are defined in terms of their effect on the outcome and can be combinations of different types of data. 

After exploring the Ames training set, we might find that the regression sloopes for the gross living area differ for different building types:


```r
ggplot(ames_train, aes(x = Gr_Liv_Area, y = 10^Sale_Price)) + 
  geom_point(alpha = .2) + 
  facet_wrap(~ Bldg_Type) + 
  geom_smooth(method = lm, formula = y ~ x, se = FALSE, color = "lightblue") + 
  scale_x_log10() + 
  scale_y_log10() + 
  labs(x = "Gross Living Area", y = "Sale Price (USD)")
```

![](Chapter_8_files/figure-html/unnamed-chunk-12-1.png)<!-- -->

How are interactions specified in a recipe? A base R formula would take an interaction using a : where * expands those columns to the main effects and interaction term. 


```r
Sale_Price ~ Neighborhood + log10(Gr_Liv_Area) + Bldg_Type + 
  log10(Gr_Liv_Area):Bldg_Type
# or
Sale_Price ~ Neighborhood + log10(Gr_Liv_Area) * Bldg_Type 
```

Recipes are more explicit and sequential, they give more control. The additional step would look like step_interact(~interaction terms) where the terms on the right-hand side of the tilde are the interactions. 


```r
simple_ames <- 
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type,
         data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>% 
  step_other(Neighborhood, threshold = 0.01) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  # Gr_Liv_Area is on the log scale from a previous step
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") )
```

Additional interactions can be specificed in this formula by separating them by +. Also note that the recipe will only use interactions between different variables; if the formula uses var_1:var:1, this term will be ignored. 

Suppose that, in a recipe, we had not yet made dummy variables for building types. It would be inappropriate to include a factor column in this step, such as:


```r
 step_interact( ~ Gr_Liv_Area:Bldg_Type )
```

```
## Gr_Liv_Area:Bldg_Type ~ list(list(terms = list(~), role = "predictor", 
##     trained = FALSE, objects = NULL, sep = "_x_", keep_original_cols = TRUE, 
##     skip = FALSE, id = "interact_NkzTj"))
```

This is telling the underlying (base R) code used by step_interact() to make dummy variables and then form the interactions. A warning states that this might generate unexpected results. 

As with naming dummy variables, recipes provides more coherent names for interaction terms. In this case, the interaction is named Gr_Liv_Area_x_Bldg_Type_Duplex instead of Gr_Liv_Area:Bldg_TypeDuplex

#### 8.4.3 Spline functions

When a predictor has a nonlinear relationship with the outcome, some types of predictor models can adaptively approximate this relationship during training. However, is is not uncommon to try to use a simple model and add in specific nonlinear features for predictors. One common method for doing this is to use spline functions to represent the data. Splines replace the existing numeric predictor with a set of columns that allow a model to emulate a flexible, nonlinear relationship. As more spline terms are added to the data, the capacity to nonlinearly represent the relationship increases. 

If you have ever used geom_smooth() with ggplot, you have probably used a spline representation of the data. Each panel below uses a different number of smooth splines for the latitude predictor:


```r
library(patchwork)
library(splines)

plot_smoother <- function(deg_free) {
  ggplot(ames_train, aes(x = Latitude, y = 10^Sale_Price)) + 
    geom_point(alpha = .2) + 
    scale_y_log10() +
    geom_smooth(
      method = lm,
      formula = y ~ ns(x, df = deg_free),
      color = "lightblue",
      se = FALSE
    ) +
    labs(title = paste(deg_free, "Spline Terms"),
         y = "Sale Price (USD)")
}

( plot_smoother(2) + plot_smoother(5) ) / ( plot_smoother(20) + plot_smoother(100) )
```

![](Chapter_8_files/figure-html/unnamed-chunk-16-1.png)<!-- -->

The ns() function in the splines package generates feature columns using functions called natural splines.

In recipes, multiple steps can create these types of terms. To add a natural spline representation for this predictor:


```r
recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + Latitude,
         data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>% 
  step_other(Neighborhood, threshold = 0.01) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) %>% 
  step_ns(Latitude, deg_free = 20)
```

```
## 
```

```
## ── Recipe ──────────────────────────────────────────────────────────────────────
```

```
## 
```

```
## ── Inputs
```

```
## Number of variables by role
```

```
## outcome:   1
## predictor: 5
```

```
## 
```

```
## ── Operations
```

```
## • Log transformation on: Gr_Liv_Area
```

```
## • Collapsing factor levels for: Neighborhood
```

```
## • Dummy variables from: all_nominal_predictors()
```

```
## • Interactions with: Gr_Liv_Area:starts_with("Bldg_Type_")
```

```
## • Natural splines on: Latitude
```

The user would need to determine if both neighborshood and latitude should be in the model since they both represent the same underlying data in different ways.

#### 8.4.4 Feature Extraction

Another common method for representing multiple features at once is called feature extraction. Most of these techniques create new features from the predictors that capture the information in the broader set as a whole. For example, PCA tries to extract as much of the original information in the predictor set as possible using a smaller number of features. PCA is a linear extraction method, meaning that each new feature is a linear combination of the original predictors. 

In the Ames data, PCA might be an option to represent potentially redundant measures of size of the property variables as a smaller feature set. Apart from the gross living area, these predictors have the suffix SF in their names (square feet) so a recipe step for PCA might look like:


```r
  # Use a regular expression to capture house size predictors: 
  step_pca(matches("(SF$)|(Gr_Liv)"))
```

PCA assumes that all the predictors are on the same scale. That's true in this case, but often this step can be preceded by step_normalize() which will center and scale each column.

There are existing recipe steps for other extraction methods, such as: independent component analysis (ICA), non-negative matrix factorization (NNMF), multidimensionsal scaling (MDS), uniform manifold approximation and projection (UMAP), etc. 

#### 8.4.5 Row sampling steps

Recipe steps can affect the rows of the a data set as well. For example, subsampling techniques for class imbalances change the class proportions in the data being given to the model. These techniques often don't improve overall performance but can generate better behaved distributions of the predicted class probabilities. Approaches to try when subsampling your data with class imbalance:

1. Downsampling the data keeps the minority class and takes a random sample of the majority class so that class frequencies are blaanced.
2. Upsampling replicates samples from the minority class to balance the classes. Some techniques do this by synthesizing new samples that resemble the minority class data while other methods simply add the same minority samples repeatedly.
3. Hybrid methods do a combination of both.

The themis package has recipe steps that can be used to address class imbalance via subsampling. For simple downsampling:


```r
  step_downsample(outcome_column_name)
```

Other step functions are row-based as well: step_filter(), step_sample(), step_slice(), and step_arrange(). In almost all uses of these steps, the skip argument should be set to TRUE.

#### 8.4.6 General transformations

step_mutate() can be used to conduct a variety of basic operations to the data. It is best used for straightforward transformations like computing a ratio of two variables.

##### 8.4.7 natural language processing

Recipes can also handle data that are not in the traditional structure where the columns are features. For example, the textrecipes package can apply natural language processing methods to the data. The input column is typically a string of text, and different steps can be used to tokenize the data (e.g. split the text into separate words), filter out tokens, and create new features appropriate for modeling.

### 8.5 skipping steps for new data

The sale price data are already log-transformed in the sames data frame. Why not use: 


```r
 step_log(Sale_Price, base = 10)
```

This will cause a failure when the recipe is applied to new properties with an unknown sale price. Since price is what we are trying to predict, there probably won't be a column in the data for this variable. In fact, to avoid information leakage, many tidymodels packages isolate the data being used when making any predictions. This means that the training set and any outcome columns are not available for use at prediction time.

For simple transformations of the outcome column(s), we strongly suggest that those operations be conducted outside of the recipe.

However, there are other circumstances where this is not an adequate solution. For example, in classification models where there is a severe class imbalance, it is common to conduct subsampling of the data that are given to the modeling function. For example, suppose that there were two classes and a 10% event rate. As simple, albeit controversial approach would be to downsample the data so that the model is provided with all of the events and a random 10% of the nonevent samples.

The problem is that the same subsampling process should not be applied to the data being predicted. As a result, when using a recipe, we need a mechanism to ensure that some operations are applied only to the data that are given to the model. Each step function has an option called skip that, when set to TRUE, will be ignored by the predict() function. In this way, you can isolate the steps that affect the modeling data without causing errors when applied to new samples. However, all steps are applied when using fit().

At the time of this writing, the step functions in the recipes and themis packages that are only applied to the training data are: step_adasyn(), step_bsmote(), step_downsample(), step_filter(), step_naomit(), step_nearmiss(), step_rose(), step_sample(), step_slice(), step_smote(), step_smotenc(), step_tomek(), and step_upsample().

### 8.6 Tidy a recipe()

Before proceeding, let's create an extended recipe for the Ames data using some of the new steps we've discussed


```r
ames_rec <- 
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + 
           Latitude + Longitude, data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>% 
  step_other(Neighborhood, threshold = 0.01) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) %>% 
  step_ns(Latitude, Longitude, deg_free = 20)
```

The tidy() method, when called with the recipe object, gives a summary of the recipe steps:


```r
tidy(ames_rec)
```

```
## # A tibble: 5 × 6
##   number operation type     trained skip  id            
##    <int> <chr>     <chr>    <lgl>   <lgl> <chr>         
## 1      1 step      log      FALSE   FALSE log_9aI3F     
## 2      2 step      other    FALSE   FALSE other_NSTJP   
## 3      3 step      dummy    FALSE   FALSE dummy_5HSyx   
## 4      4 step      interact FALSE   FALSE interact_DMSaM
## 5      5 step      ns       FALSE   FALSE ns_ULMsR
```

This result can be helpful for identifying individual steps, perhaps to then be able to execute the tidy() method on one specific step.

We can identify the id argument in any step function call; otherwise it is generate using a random suffix. Setting this value can be helpful if the same type of step is added to the recipe more than once. Let's specify the id ahead of time for step_other(), since we'll want to tidy() it:


```r
ames_rec <- 
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + 
           Latitude + Longitude, data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>% 
  step_other(Neighborhood, threshold = 0.01, id = "my_id") %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) %>% 
  step_ns(Latitude, Longitude, deg_free = 20)
```

We'll refit the workflow with this new recipe:


```r
lm_wflow <- 
  workflow() %>% 
  add_model(lm_model) %>% 
  add_recipe(ames_rec)

lm_fit <- fit(lm_wflow, ames_train)
```

The tidy() method can be called again with the id identifier we specified to get our results for applying step_other()


```r
estimated_recipe <- 
  lm_fit %>% 
  extract_recipe(estimated = TRUE)

tidy(estimated_recipe, id = "my_id")
```

```
## # A tibble: 22 × 3
##    terms        retained           id   
##    <chr>        <chr>              <chr>
##  1 Neighborhood North_Ames         my_id
##  2 Neighborhood College_Creek      my_id
##  3 Neighborhood Old_Town           my_id
##  4 Neighborhood Edwards            my_id
##  5 Neighborhood Somerset           my_id
##  6 Neighborhood Northridge_Heights my_id
##  7 Neighborhood Gilbert            my_id
##  8 Neighborhood Sawyer             my_id
##  9 Neighborhood Northwest_Ames     my_id
## 10 Neighborhood Sawyer_West        my_id
## # ℹ 12 more rows
```

The tidy() results we see here for using step_other() show which factor levels were retained, i.e. not added to the new "other" category. 

The tidy() method can be called with the number identifed as well, if we know which step in the recipe we need:


```r
tidy(estimated_recipe, number = 2)
```

```
## # A tibble: 22 × 3
##    terms        retained           id   
##    <chr>        <chr>              <chr>
##  1 Neighborhood North_Ames         my_id
##  2 Neighborhood College_Creek      my_id
##  3 Neighborhood Old_Town           my_id
##  4 Neighborhood Edwards            my_id
##  5 Neighborhood Somerset           my_id
##  6 Neighborhood Northridge_Heights my_id
##  7 Neighborhood Gilbert            my_id
##  8 Neighborhood Sawyer             my_id
##  9 Neighborhood Northwest_Ames     my_id
## 10 Neighborhood Sawyer_West        my_id
## # ℹ 12 more rows
```

Each tidy method returns the relevant information about that step. For example, the tidy() method for step_dummy() returns a column with the variables that were converted to dummy variables and another column with all of the known levels for each column.

### 8.7 Column Roles

When a formula is used with the initial call to recipe() it assigns roles to each of the columns, depending on which side of the tilde they are on. Those roles are either "predictor" or "outcome". However, other roles can be assigned.

To solve this, the add_role(), remove_role(), and update_role() functions can be helpful. For example, for the house price data, the role of the street address column could be modified using:


```r
ames_rec %>% update_role(address, new_role = "street address")
```

After this change, the address column in the dataframe will no longer be predicted but instead will be a "street address" according to the recipe. Any character string can be used as a role. Also, columns can have multiple roles so that they can be selected under more than one context. 

This can be helpful when the data are resampled. It helps to keep the columns that are not involved with the model fit in the same data frame. Resampling creates alternate versions of the data mostly by row subsampling. If the street address were in another column, additional subsampling would be required and might lead to more complex code and a higher likelihood of errors.

Finally, all step functional have a role field that can assign roles to the results of the step. In many cases, columns affected by a step retain their existing role. For example, the step_log() calls to our ames_rec object affected the Gr_Liv_Area column. For that step, the default behavior is to keep the existing role for this column since no new column is created. As a counter-example, the step to produce splines defaults new columns to have a role of "predictor" since that is usually how spline columns are used in a model. 

### code for later chapter


```r
library(tidymodels)
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



