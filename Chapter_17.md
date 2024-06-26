---
title: "Chapter_17"
author: "Sam Worthy"
date: "2024-04-07"
output: 
  html_document: 
    keep_md: yes
---

# Encoding Categorical Data

In R, the preferred representation for categorical or nominal data is a factor, which is a variable that can take on a limited number of different values. Internally, factors are stored as a vector of integer values together with a set of text labels. 

For some realistic data sets, straightforward dummy variables are not a good fit. This often happens because there are too many categories or there are new categories at prediction time. 


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
library(embed)
library(textrecipes)
library(rlang)
```

```
## 
## Attaching package: 'rlang'
```

```
## The following objects are masked from 'package:purrr':
## 
##     %@%, flatten, flatten_chr, flatten_dbl, flatten_int, flatten_lgl,
##     flatten_raw, invoke, splice
```

## 17.1 Is an encoding necessary?

A minority of models, such as those based on trees or rules, can handle categorical data natively and do not require encoding or transformation of these kinds of features. These models that can handle categorical features natively can also deal with numeric continuous features, making the transformation or encoding of such variables optional. Dummy encodings did not typically result in better model performance but often required more time to train the models. 

## 17.2 Encoding ordinal predictors

Sometimes qualitative columns can be ordered, such as low, medium, and high. In base R, the default encoding strategy is to make new numeric columns that are polynomial expansions of the data. While this is not unreasonable, it is not an approach that people tend to find useful. For example, an 11-degree polynomial is probably not the most effective way of encoding an ordinal factor for the months of the year. Instead, consider tryign recipe steps related to ordered factors, such as step_unorder(), to convert to regular factors, and step_ordinalscore(), which maps specific numeric values to each factor level.

## 17.3 Using the outcome for encoding predictors

There are multiple options for encodings more complex than dummy or indicator variables. One method called effect or likelihood encodings replaces the original categorical variables with a single numeric column that measures the effect of those data. For example, for the neighborhood predictor in the Ames housing data, we can compute the mean or median sale price for each neighborhood and substitute these means for the original data values. 


```r
data(ames)
ames <- mutate(ames, Sale_Price = log10(Sale_Price))

set.seed(502)
ames_split <- initial_split(ames, prop = 0.80, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test  <-  testing(ames_split)
```



```r
ames_train %>%
  group_by(Neighborhood) %>%
  summarize(mean = mean(Sale_Price),
            std_err = sd(Sale_Price) / sqrt(length(Sale_Price))) %>% 
  ggplot(aes(y = reorder(Neighborhood, mean), x = mean)) + 
  geom_point() +
  geom_errorbar(aes(xmin = mean - 1.64 * std_err, xmax = mean + 1.64 * std_err)) +
  labs(y = NULL, x = "Price (mean, log scale)")
```

![](Chapter_17_files/figure-html/unnamed-chunk-3-1.png)<!-- -->

This kind of effect encoding works well when your categorical variable has many levels. In tidymodels, the embed package includes several recipe step functions for different kinds of effect encodings, such as step_lencode_glm(), step_lencode_mixed(), and step_lencode_bayes(). These steps use a generalized linear model to estimate the effect of each level in a categorical predictor on the outcome. When using a recipe step like step_lencode_glm(), specify the variable being encoded first then the outcome using vars().


```r
ames_glm <- 
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + 
           Latitude + Longitude, data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>% 
  step_lencode_glm(Neighborhood, outcome = vars(Sale_Price)) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) %>% 
  step_ns(Latitude, Longitude, deg_free = 20)

ames_glm
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
## predictor: 6
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
## • Linear embedding for factors via GLM for: Neighborhood
```

```
## • Dummy variables from: all_nominal_predictors()
```

```
## • Interactions with: Gr_Liv_Area:starts_with("Bldg_Type_")
```

```
## • Natural splines on: Latitude, Longitude
```

We can prep() our recipe to fit or estimate parameters for the preprocessing transformations using training data. We can then tidy() this prepared recipe to see the results.


```r
glm_estimates <-
  prep(ames_glm) %>%
  tidy(number = 2)

glm_estimates
```

```
## # A tibble: 29 × 4
##    level              value terms        id               
##    <chr>              <dbl> <chr>        <chr>            
##  1 North_Ames          5.15 Neighborhood lencode_glm_yj20u
##  2 College_Creek       5.29 Neighborhood lencode_glm_yj20u
##  3 Old_Town            5.07 Neighborhood lencode_glm_yj20u
##  4 Edwards             5.09 Neighborhood lencode_glm_yj20u
##  5 Somerset            5.35 Neighborhood lencode_glm_yj20u
##  6 Northridge_Heights  5.49 Neighborhood lencode_glm_yj20u
##  7 Gilbert             5.28 Neighborhood lencode_glm_yj20u
##  8 Sawyer              5.13 Neighborhood lencode_glm_yj20u
##  9 Northwest_Ames      5.27 Neighborhood lencode_glm_yj20u
## 10 Sawyer_West         5.24 Neighborhood lencode_glm_yj20u
## # ℹ 19 more rows
```

When we use the newly encoded Neighborhood numeric variable created via this method, we substitute the original level (such as North_Ames) with the estimate for Sale_Price from the GLM.

Effect encoding methods like this one can also seamlessly handle situations where a novel factor level is encountered in the data. This value is the predicted price from the GLM when we don't have any specific neighborhood information.


```r
glm_estimates %>%
  filter(level == "..new")
```

```
## # A tibble: 1 × 4
##   level value terms        id               
##   <chr> <dbl> <chr>        <chr>            
## 1 ..new  5.23 Neighborhood lencode_glm_yj20u
```

Effect encodings can be powerful but should be used with care. The effects should be computed from the training set, after data splitting .This type of supervised preprocessing should be rigorously resampled to avoid overfitting.

When you create encoding for your categorical variable, you are effectively layering a mini-model inside your actual model. The possibility of overfitting with effect encodings is a representative example for why feature engineering must be considered part of the model process, and why feature engineering must be estimated together with model parameters inside resampling. 

#### 17.3.1 Effect encodings with partial pooling

Creating an effect encoding with step_lencode_glm() estimates the effect separately for each factor level. However, some of these neighborhoods have many houses in them, and some have only a few. There is much more uncertainty in our measurement of price for the single training set home in the Landmark neighborhood than the 354 training set homes in North Ames. We can use partial pooling to adjust these estimates so that levels with small sample sizes are shruken toward the overall mean. The effects for each level are modeled all at once using a mixed ofr hierarchical generalized linear model:


```r
ames_mixed <- 
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + 
           Latitude + Longitude, data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>% 
  step_lencode_mixed(Neighborhood, outcome = vars(Sale_Price)) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) %>% 
  step_ns(Latitude, Longitude, deg_free = 20)

ames_mixed
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
## predictor: 6
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
## • Linear embedding for factors via mixed effects for: Neighborhood
```

```
## • Dummy variables from: all_nominal_predictors()
```

```
## • Interactions with: Gr_Liv_Area:starts_with("Bldg_Type_")
```

```
## • Natural splines on: Latitude, Longitude
```

Let's prep() and tidy() this recipe to see the results:


```r
mixed_estimates <-
  prep(ames_mixed) %>%
  tidy(number = 2)

mixed_estimates
```

```
## # A tibble: 29 × 4
##    level              value terms        id                 
##    <chr>              <dbl> <chr>        <chr>              
##  1 North_Ames          5.15 Neighborhood lencode_mixed_AmBz1
##  2 College_Creek       5.29 Neighborhood lencode_mixed_AmBz1
##  3 Old_Town            5.07 Neighborhood lencode_mixed_AmBz1
##  4 Edwards             5.10 Neighborhood lencode_mixed_AmBz1
##  5 Somerset            5.35 Neighborhood lencode_mixed_AmBz1
##  6 Northridge_Heights  5.49 Neighborhood lencode_mixed_AmBz1
##  7 Gilbert             5.28 Neighborhood lencode_mixed_AmBz1
##  8 Sawyer              5.13 Neighborhood lencode_mixed_AmBz1
##  9 Northwest_Ames      5.27 Neighborhood lencode_mixed_AmBz1
## 10 Sawyer_West         5.24 Neighborhood lencode_mixed_AmBz1
## # ℹ 19 more rows
```

New levels are then encoded at close to the same value as with the GLM:


```r
mixed_estimates %>%
  filter(level == "..new")
```

```
## # A tibble: 1 × 4
##   level value terms        id                 
##   <chr> <dbl> <chr>        <chr>              
## 1 ..new  5.23 Neighborhood lencode_mixed_AmBz1
```

Let's visually compare the effects using partial pooling vs. no pooling.


```r
glm_estimates %>%
  rename(`no pooling` = value) %>%
  left_join(
    mixed_estimates %>%
      rename(`partial pooling` = value), by = "level"
  ) %>%
  left_join(
    ames_train %>% 
      count(Neighborhood) %>% 
      mutate(level = as.character(Neighborhood))
  ) %>%
  ggplot(aes(`no pooling`, `partial pooling`, size = sqrt(n))) +
  geom_abline(color = "gray50", lty = 2) +
  geom_point(alpha = 0.7) +
  coord_fixed()
```

```
## Joining with `by = join_by(level)`
```

```
## Warning: Removed 1 rows containing missing values (`geom_point()`).
```

![](Chapter_17_files/figure-html/unnamed-chunk-10-1.png)<!-- -->

Notice that most estimates for neighborhood effects are about the same when we compare pooling to no pooling. However, the neighborhood with the fewest homes in them have been pulled (either up or down) toward the mean effect. When we use pooling, we shrink the effect estimates toward the mean because we don't have as much evidence about the price in those neighborhoods.

## 17.4 Feature hashing

Traditional dummy variables require that all the possible categories be known to create a full set of numeric features. Feature hashing methods also create dummy variables, but only consider the value of the category to assign it to a predefined pool of dummy variables. Let's look at the Neighborhood values in Ames again and use the rland::hash() function to understand more.


```r
ames_hashed <-
  ames_train %>%
  mutate(Hash = map_chr(Neighborhood, hash))

ames_hashed %>%
  select(Neighborhood, Hash)
```

```
## # A tibble: 2,342 × 2
##    Neighborhood    Hash                            
##    <fct>           <chr>                           
##  1 North_Ames      076543f71313e522efe157944169d919
##  2 North_Ames      076543f71313e522efe157944169d919
##  3 Briardale       b598bec306983e3e68a3118952df8cf0
##  4 Briardale       b598bec306983e3e68a3118952df8cf0
##  5 Northpark_Villa 6af95b5db968bf393e78188a81e0e1e4
##  6 Northpark_Villa 6af95b5db968bf393e78188a81e0e1e4
##  7 Sawyer_West     5b8ac3fb32a9671d55da7b13c6e5ef49
##  8 Sawyer_West     5b8ac3fb32a9671d55da7b13c6e5ef49
##  9 Sawyer          2491e7eceb561d5a5f19abea49145c68
## 10 Sawyer          2491e7eceb561d5a5f19abea49145c68
## # ℹ 2,332 more rows
```

If we input Briardale to this hashing function, we will always get the same output. The neighborhoods in this case are called the "keys", while the outputs are the "hashes".

A hashing function takes an input of variable size and maps it to an output of fixed size. Hashing functions are commonly used in cryptography and databases.

The rlang::hash() function generates a 128-bit hash, which means there are 2^128 possible hash values. This is great for some applications but doesn't help with feature hashing of high cardinality variables (variables with many levels). In feature hashing, the number of possible hashes is a hyperparameter and is set by the model developer through computing the modulo of the integer hashes. We can get sixteen possible hash values using Hash %% 16:


```r
ames_hashed %>%
  ## first make a smaller hash for integers that R can handle
  mutate(Hash = strtoi(substr(Hash, 26, 32), base = 16L),  
         ## now take the modulo
         Hash = Hash %% 16) %>%
  select(Neighborhood, Hash)
```

```
## # A tibble: 2,342 × 2
##    Neighborhood     Hash
##    <fct>           <dbl>
##  1 North_Ames          9
##  2 North_Ames          9
##  3 Briardale           0
##  4 Briardale           0
##  5 Northpark_Villa     4
##  6 Northpark_Villa     4
##  7 Sawyer_West         9
##  8 Sawyer_West         9
##  9 Sawyer              8
## 10 Sawyer              8
## # ℹ 2,332 more rows
```

Now instead of the 28 neighborhoods in our original data or an incredibly huge umber of the original hashes, we have 16 hash values. This method is very fast and memory efficient, and it can be a good strategy when there are a large number of possible categories. 

We can implement feature hashing using a tidymodels recipe step from the textrecipes package:


```r
ames_hash <- 
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + 
           Latitude + Longitude, data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>% 
  step_dummy_hash(Neighborhood, signed = FALSE, num_terms = 16L) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) %>% 
  step_ns(Latitude, Longitude, deg_free = 20)
```

```
## 1 package (text2vec) is needed for this step but is not installed.
```

```
## To install run: `install.packages("text2vec")`
```

```r
ames_hash
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
## predictor: 6
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
## • Feature hashing with: Neighborhood
```

```
## • Dummy variables from: all_nominal_predictors()
```

```
## • Interactions with: Gr_Liv_Area:starts_with("Bldg_Type_")
```

```
## • Natural splines on: Latitude, Longitude
```

Feature hashing is fast and efficient but has a few downsides. For example, different category values often map to the same hash value. This is called a collision or aliasing. How often does this happen with our neighborhood in Ames?

The number of neighborhoods mapped to each hash value varies between zero and four. All of the hash values greater than one are examples of hash collisions. 

What are some things to consider when using feature hashing?

* Feature hashing is not directly interpretable because hash functions cannot be reversed. We can't determine what the input category levels were from the hash value, or if a collision occurred.
* The number of hash values is a tuning parameter of this preprocessing technique, and you should try several values to determine what is best for your particular modeling approach. A lower number of hash values results in more collisions, but a high number may not be an improvement over your original high cardinality variable.
* Feature hashing can handle new category levels at prediction time, since it does not rely on pre-determined dummy variables.
* You can reduce hash collisions with a signed hash by using signed=TRUE. This expands the values from only 1 to either +1 or -1, depending on the sign of the hash.

It is likely that some hash columns will contain all zeros, as we see in this example. We recommend a zero-variance filter vai step_zv() to filter out such columns.

## 17.5 More encoding options

We can build a full set of entity embeddings to transform a categorical variable with many levels to a set of lower-dimensional vectors. This approach is best suited to a nominal variable with many category levels, many more than the example we've used with neighborhoods in Ames.

Embeddings for a categorical variable can be learned via a TensorFlow neural network with the step_embed() function in embed. We can use the outcome alone or optionally the outcome plus a set of additional predictors. Like in feature hashing, the number of new encoding columns to create is a hyperparamter of the feature engineering. We also must make decisions about the neural network structure (the number of hidden units) and how to fit the neural network (how many epochs to train, how much of the data to use for validation in measuring metrics).

Yet one more option available for dealing with a binary outcome is to transform a set of category levels based on their association with the binary outcome. This weight of evidence (WoE) transformation uses the logarithm of the "Bayes factor" (the ratio of the posterior odds to the prior odds) and creates a dictionary mapping each category level to a WoE value. WoE encodings can be determined with the step_wod() function in embed.

