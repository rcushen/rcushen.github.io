---
layout: post
title: "A Survey of Shrinkage Methods"
date: 2018-12-23 12:00 +1000
categories: statistics
---

*In this post, I explore some of the common shrinkage methods employed to combat the problem of overfitting. Specifically, the LASSO, ridge regression, and the elastic-net are detailed. The techniques are motivated by common issues that arise in the estimation of a known real-world parameter.*

# A Survey of Shrinkage Methods

A common problem encountered when estimating statistical models is *overfitting*. In the era of 'Big Data', this arises all the more often, as almost anything can be 'statistically significant' and the temptation to include more and more variables can hence be hard to avoid. However, there's no such thing as a free lunch. As we shall see, a surplus of regressors can both conceal the most important variables and compromise the quality of our parametric inference, dulling the robustness of our model. *Shrinkage* provides a method to combat this issue, applying an additional penalty in the model-fitting process for complexity. In this post we shall explore the exact motivation for shrinkage, and describe the three most common forms: LASSO, ridge regression, and the elastic-net. All are closely interrelated, though each have their own unique strengths and weaknesses in helping us improve our models.

## Introduction

### Motivation

Consider the following scenario. We want to predict a dependent variable \\(y\\) using a set of candidate regressors \\( \\{ x_j^T \\}_{j = 1}^k\\) and a linear model specification, but are unsure of exactly which regressors should be employed. All of our candidate regressors are correlated with the response \\(y\\), but we have no guarantees that they in fact form part of the data-generating process (i.e. the underlying structural relationship), which remains opaque to us. Indeed, we admit the possibility that many of these regressors are spurious correlates.

At face value, the solution may seem obvious: use all regressors, as we know that the quality of the fit (measured by \\(R^2\\)) will increase monotonically with the addition of correlated variables, even if they have no theoretical justification to be included in the model. So long as the four assumptions which underpin the Gauss-Markov Theorem are satisfied, we know that our model will remain the best linear unbiased estimator of the relevant parameters \\(\beta\\). However, adding variables in this fashion can cause two issues.

1. Often we want to keep our model simple, both to improve external validity and to aid in interpretation. This is to say, we have a preference toward simplicity over complexity. In particular, we want a model that incorporates only those regressors with a strong influence on the dependent variable.
2. If these independent variables are correlated with one another (as they almost certainly are), the quality of our parameter estimates \\(\hat{\beta}\\) will fall. The technical term here is *multicollinearity*, and it will cause the variance of our estimates to increase. Written crudely,

\\[
\frac{\text{d} \operatorname{Var}[\hat{\beta}]}{\text{d} \ k}\geq 0
\\]

​where \\(k\\) is the number of variables included in the model.

In other words, we may be overfitting the data. This is a perfect example of the *bias-variance tradeoff*: by introducing additional regressors, we may be reducing bias, but we are increasing the variance of both our model as a whole and the parameter estimates which comprise it.

We therefore may wish to impose a penalty on complexity. Concretely, we want our model to only give weight to the key variables involved in the data generating process. Shrinkage performs exactly this task, offering both a means of culling the less-influential variables and improving the quality of our parameter estimates for those which remain.

### The Dataset

To illustrate this problem of overfitting—and explore some potential solutions—we will be using the taxi fares dataset taken from [Google Cloud](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction). This contains data on millions of taxi rides in New York City between 2009 and 2015. In particular, we have the pickup date and time, pickup and dropoff coordinates, and number of passengers available as variables, and each row signifies a single taxi trip.

```R
> glimpse(taxi_data)
Observations: 9,781,194
Variables: 7
$ pickup_datetime   <dttm> 2009-06-15 17:26:21, 2010-01-05 16:52:16,...
$ fare_amount       <dbl> 4.5, 16.9, 5.7, 7.7, 5.3, 12.1, 7.5, 16.5,...
$ pickup_longitude  <dbl> -73.84431, -74.01605, -73.98274, -73.98713...
$ pickup_latitude   <dbl> 40.72132, 40.71130, 40.76127, 40.73314, 40...
$ dropoff_longitude <dbl> -73.84161, -73.97927, -73.99124, -73.99157...
$ dropoff_latitude  <dbl> 40.71228, 40.78200, 40.75056, 40.75809, 40...
$ passenger_count   <int> 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, ..
```

Our goal will be to model the `fare_amount` variable, hereafter denoted as \\(y\\), as some function of the independent variables \\(x\\). This is to say, we will be performing parametric inference over the unknown function \\(f\\), where \\(y=f(x)\\).

Unusually however, we in fact already have the solution for \\(f\\) available to us. Taxi fares follow a closed-form equation: some combination of distance and time, plus other flat charges. Taking the figures from the [NYC government web page](http://www.nyc.gov/html/tlc/html/passenger/taxicab_rate.shtml), we can write the fare amount \\(y\\) as
\\[
y = 2.5 + 1.553 \times \text{distance} \ + 0.50 \times \text{time} \ + C
\\]
where distance and time are measured in kilometres and minutes respectively, and \\(C\\) signifies other flat charges (e.g. the Newark surcharge of \$10, the peak hour surcharge of \$1). By this calculation, a taxi ride from Washington Square to Times Square (3.7 kilometres, taking around 19 minutes according to Google Maps) would cost
\\[
y = 2.5 + 1.553 \times 3.701 + 0.50 \times 8 + (0.50 + 0.30 )= $13.05,
\\]
taking \\(C\\) to be composed of the NY State Tax Surcharge and NY Improvement Surcharge. This matches roughly with the calculator on www.taxifarefinder.com, (\$15.71) as well as with the average fare over this distance in our data (\$11.9).

*This is a simplification, as pricing is in fact discrete and not continuous—the meter will only tick over every 1/5 mile and every 120 seconds—but it will serve as a perfectly ample benchmark. There has also been a structural change in these prices during our sample period, which will be addressed later.*

So this is our goal: to accurately estimate the parameter \$1.553 per kilometer.

### Exploratory Visualisations

Before building a model, let's appraise the variables available to us in ```taxi_data```. A histogram of the fares variable shows a reasonable distribution, which is encouraging for modelling purposes as it suggests the conditional mean \\(\mathbb{E} [ y | x]\\) is likely to be well-defined. Most fares look to be under \$20, with a positive skew indicating the presence of a few larger outliers.

![fares_distribution](/assets/fares_distribution.png){:class="img-responsive" border=0}

If we look at the average fare each month across our sample, a marked shift is apparent: fares have gone up. Further investigation confirms this, with the Taxi and Limousine Commission increasing fares by 25% in September 2012. For the sake of simplicity, we will exclude these trips that occurred under a different pricing regime and instead focus on the latter three years of the sample.

![monthly_mean_fares](/assets/monthly_mean_fares.png){:class="img-responsive" border=0}

What about the pickup and dropoff locations? If we create a simple scatterplot of pickup coordinates, we see something resembling a street map of NYC. As would be expected, the majority of activity occurs in Manhattan, testament to the small average fares observed in the data.

![pickup_coordinates](/assets/pickup_coordinates.png){:class="img-responsive" border=0}

One conspicuously absent feature in our data is the actual distance of the taxi trip. This will have to be calculated using the pickup and dropoff coordinates, as well as a little spherical trigonometry. While we could simply take the Euclidean distance between the two points, this would be inaccurate - the earth is a sphere, so Euclidean approximations fail outside of small neighbourhoods. Instead we use the *haversine* formula, which gives the distance between two points on a globe. Taken from [this blog](https://www.movable-type.co.uk/scripts/latlong.html), the formula is given by
\\[
\begin{align}
a &= \sin^2 \left(  \frac{\Delta \phi}{2} \right) + \cos \phi_1 \cdot \cos \phi_2 \cdot \sin^2 \left( \frac{\Delta \lambda}{2} \right) \\\
c &= 2 \cdot \operatorname{arctan2}(\sqrt{a}, \sqrt{1-a}) \\\
h &= R \cdot c
\end{align}
\\]
where \\(\phi\\) is latitude, \\(\lambda\\) is longitude and \\(r\\) is the radius of the earth (which we take to be 6,371,000 metres). \\(h\\) is then the as-the-crow distance of the taxi ride measured in metres.

Applying this formula to every single observation in ```taxi_data``` gives us a new ```haversine``` variable, which can be used as a measure of the taxi fare distance. However, we can in fact generate an even more accurate calculation by taking advantage of a well-known property of New York: almost the entire city sits on a grid. Most streets in NYC are perpendicular to one another, meaning we may be better-off using the eponymous *Manhattan* distance metric. Known more technically as the \\(\ell_1\\)-norm, this distance metric describes a space in which only perpendicular movement is possible – which sounds much more appropriate than as-the-crow flies haversine distance, looking at a street map of NYC! The conversion is very simple, this time requiring only regular trigonometry.
\\[
m = h (|\cos b | + | \sin b |)
\\]
The \\(h\\) term is the haversine distance, and \\(b\\) is the bearing of the taxi trip, with \\(29^\circ\\) subtracted off to account for the rotation of the NYC grid with respect to true north (a contender for most useless piece of pub trivia).

![metric_comparisons](/assets/metric_comparisons.png){:class="img-responsive" border=0}

If we compare the distribution of haversine and Manhattan distance calculations, we see that the two are nevertheless highly similar. The key difference is the longer tails in the Manhattan histogram; Manhattan distance is strictly greater than haversine distance, which makes sense, as the two sides of a rectangle will always exceed the length of the diagonal. Most importantly, comparison of our calculated Manhattan distances with a sample of true driving distances taken from Google Maps reveals a surprising level of accuracy. As such, we will use this calculated Manhattan distance as our distance moving forward.

## Analysis

### The Overfitting Problem

Before we get to overfitting, let's first consider a simple linear model of fare on distance.

```R
> summary(lm_manhattan)

Call:
lm(formula = fare_amount ~ manhattan, data = processed_taxi_data)

Residuals:
    Min      1Q  Median      3Q     Max
-109.16   -1.72   -0.76    0.87  896.37

Coefficients:
             Estimate Std. Error t value Pr(>|t|)    
(Intercept) 3.827e+00  3.175e-03    1205   <2e-16 ***
manhattan   2.140e-03  5.163e-07    4145   <2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 4.776 on 4171454 degrees of freedom
Multiple R-squared:  0.8046,	Adjusted R-squared:  0.8046
F-statistic: 1.718e+07 on 1 and 4171454 DF,  p-value: < 2.2e-16
```

Distance is clearly a significant variable. However, the coefficient estimated of \$2.14 has overshot the true value by 38%, which we know to be \$1.553. The same is true for the intercept, which we know should be equal to around \$3.30. This has occurred because of *omitted variable bias*: variables exist outside of our system which have not been accounted for, but which influence the response \\(y\\). Of course, thanks to the explicit closed-form solution for \\(y\\), we know this missing variable to be time. Since it has not been explicitly included in the model, it will be instead captured by the error term \\(\varepsilon\\), and since we expect time to be positively correlated with both distance and fare amount, it has caused the parameter estimates to be greater than their true values.

Let's try adding two more variables: the number of passengers, and the direction of travel.

```R
> summary(lm_generalised)

Call:
lm(formula = fare_amount ~ manhattan + passenger_count + bearing,
    data = processed_taxi_data)

Residuals:
    Min      1Q  Median      3Q     Max
-109.33   -1.74   -0.75    0.88  896.57

Coefficients:
                  Estimate Std. Error  t value Pr(>|t|)    
(Intercept)      3.762e+00  4.322e-03  870.387   <2e-16 ***
manhattan        2.142e-03  5.164e-07 4146.994   <2e-16 ***
passenger_count  1.657e-02  1.718e-03    9.649   <2e-16 ***
bearing         -1.663e-03  2.217e-05  -75.022   <2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 4.773 on 4171452 degrees of freedom
Multiple R-squared:  0.8049,	Adjusted R-squared:  0.8049
F-statistic: 5.737e+06 on 3 and 4171452 DF,  p-value: < 2.2e-16
```

This was, however, a trick: there is absolutely no reason why number of passengers should have any influence on the fare! The direction of travel might, but it is unclear whether this is functioning as a useful instrument of time, or if it is just noise. That the adjusted \\(R\\)-squared has not changed suggests this is unlikely. Moreover, the inclusion of these variables has had very little effect upon on the parameter estimates for the intercept and distance. The intercept has fallen slightly, as would be expected, but the distance parameter estimate has remained unchanged.

To demonstrate this even further, let's add two more variables which should be entirely spurious: the specific coordinates of pickup and dropoff for the trip. This brings our total number of regressors up to seven.

```R
> summary(lm_even_generaliseder)

Call:
lm(formula = fare_amount ~ manhattan + passenger_count + bearing +
    pickup_latitude + pickup_longitude + dropoff_latitude + dropoff_longitude,
    data = processed_taxi_data)

Residuals:
    Min      1Q  Median      3Q     Max
-108.28   -1.72   -0.77    0.86  897.52

Coefficients:
                    Estimate Std. Error  t value Pr(>|t|)    
(Intercept)        3.963e+02  9.406e+00   42.133   <2e-16 ***
manhattan          2.118e-03  6.459e-07 3278.794   <2e-16 ***
passenger_count    1.678e-02  1.708e-03    9.823   <2e-16 ***
bearing            8.318e-04  2.603e-05   31.954   <2e-16 ***
pickup_latitude    5.745e+00  9.978e-02   57.573   <2e-16 ***
pickup_longitude   1.177e+01  8.093e-02  145.450   <2e-16 ***
dropoff_latitude  -1.175e+01  8.482e-02 -138.549   <2e-16 ***
dropoff_longitude -9.777e+00  8.200e-02 -119.231   <2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 4.747 on 4171448 degrees of freedom
Multiple R-squared:  0.807,	Adjusted R-squared:  0.807
F-statistic: 2.492e+06 on 7 and 4171448 DF,  p-value: < 2.2e-16
```

Everything is significant, despite having no reason to be! This is precisely the issue that shrinkage will attempt to solve.

### Shrinkage Methods

Shrinkage can be understood via a number of paradigms. The most obvious of these is that it involves  applying an additional penalty upon the magnitude of the least squares parameters, according to some metric. However, shrinkage can also be cast in a Bayesian framework. Imposing the aforementioned penalty is equivalent to applying a prior; in this case, our prior belief is that the magnitude of estimated coefficients should not be too large. The weight of this prior is controlled by the hyperparameter \\(\lambda\\), which we will see directly dictates the strength of the penalty. This \\(\lambda\\) value can been seen as the common feature of all shrinkage methods – their difference lies in the precise structure of the penalty function. Let us explore three of these variants now.

#### LASSO

The LASSO is fundamentally an exercise in variable selection. Hence the name: *least absolute shrinkage and selection operator*. The penalty in this case is driven by the sum of the least squares parameter estimates. More technically, this is an \\(\ell_1\\)-norm penalty applied to the \\(\hat{\beta}\\) vector. We can therefore express the objective function as
\\[
\min_{\beta_0, \beta} \left\\{ \\frac{1}{N} \sum_{i=1}^N ( y_i - \beta_0 - x_i^T \beta )^2) + \lambda \sum_{j=1}^p \beta_j \right\\} ,
\\]
which makes clear the distinction between the ordinary least squares component and the LASSO penalty. Note that the LASSO penalty does not include the intercept term – for this reason, the variables are often centred before estimation.

Unfortunately, the \\(\ell_1\\)-norm in the LASSO objective function means it cannot be differentiated, and hence does not lend itself well to simple optimisation. Computation of estimates is in fact a quadratic programming problem. Fortunately, efficient algorithms do exist (e.g. least angle regression), allowing us derive solutions even on large datasets. In ```R```, the LASSO can be implemented using the ```glmnet``` package. Written by one of the original developers of the LASSO technique (1), the package also includes a \\(k\\)-fold cross-validation tool to aid in selection of \\(\lambda\\).

Fitting the same specification as ```lm_even_generaliseder``` yields the following results.

![cross-validation_LASSO](/assets/cross-validation_LASSO.png){:class="img-responsive" border=0}

Visibly, we can see that the effect of dropping most variables is very small. The numbers along the top denote the number of variables that remain in the fit as \\(\lambda\\) is varied; all the way down to one variable, there is almost no consequence on the quality of the fit. While they may be significantly *correlated* with the response, they clearly do not have a large effect on the *quality* of predictions. This is exactly the problem of big data: significant correlations, but tiny effects.

This becomes all the more clear when we look to how the estimated coefficients change as \\(\lambda\\) is altered.

![coefficient_profiles_LASSO](/assets/coefficient_profiles_LASSO.png){:class="img-responsive" border=0}

The x-axis in this case is the \\(\ell_1\\)-norm, which will be inversely related to the magnitude of \\(\lambda\\). From the start, we see that ```manhattan``` is the most significant variable, which is unsurprising. However, even when the other coefficients finally enter, their magnitude is miniscule relative to ```manhattan```. The conclusion here is clear: distance driven by a taxicab is by far the most dominant predictor of fare. Note that this was not at all obvious from our initial fit of ```lm_even_generaliseder```! Without having seen these LASSO paths, one may have been inclined to value all seven regressors equally.

Applying the LASSO has therefore shown us that ```manhattan``` is by far the most important variable in our model. However, which value of \\(\lambda\\) should we select? This is an important question, as it will determine the coefficient estimate that is reported. There are two standard options. Either we choose the value which minimises the mean-squared error during cross-validation (the first vertical dashed line), or we choose the value which generates the most regularised model with an error within one standard error of the minimum (the second vertical dashed line). In our case, we choose the latter, yielding the following estimates.

```R
> coef(model, s = "lambda.1se")
8 x 1 sparse Matrix of class "dgCMatrix"
                              1
(Intercept)       -2.712184e-17
manhattan          8.326637e-01
passenger_count    .           
bearing            .           
pickup_latitude    .           
pickup_longitude   .           
dropoff_latitude   .           
dropoff_longitude  .   
```

These are of course scaled estimates; after unscaling, we obtain a slope estimate of \$1.98, which already brings us closer to the correct parameter of \$1.553. So far, so good! LASSO has clearly provided an improvement over our initial linear fit. What does ridge regression have to offer?

#### Ridge Regression

Ridge regression is highly similar to the LASSO, and was in fact conceived first. One small but crucial tweak is made: instead of an \\(\ell_1\\)-norm, we now apply an \\(\ell_2\\)-norm penalty to \\(\hat{\beta}\\). The objective function is therefore:
\\[
\min_{\beta_0, \beta} \left\\{ \\frac{1}{N} \sum_{i=1}^N ( y_i - \beta_0 - x_i^T \beta )^2) + \lambda \sum_{j=1}^p \beta_j^2 \right\\} ,
\\]
What is the consequence of moving to an \\(\ell_2\\)-norm? Larger values of \\(\hat{\beta_i}\\) are now penalised even more aggressively, meaning estimated coefficients of a small magnitude are preferred – so to flip this around, if a variable is going to have a large estimated coefficient, it better have a large effect upon the response. It can be shown that this constraint on size causes a corresponding reduction in variance of parameter estimates, making the technique perfect for datasets that suffer from multicollinearity (such as our taxi fares data!). Moreover, the penalty term is now continuous, and can therefore be solved analytically.

The best way to understand how ridge regression differs from the LASSO is nevertheless to see it in action. The ```glmnet``` package is again used for implementation, and we one again fit the same model specification as ```lm_even_generaliseder```.

![cross-validation_ridge](/assets/cross-validation_ridge.png){:class="img-responsive" border=0}

A starkly different profile emerges. Where the LASSO saw little change in mean-squared error for a broad spectrum of \\(\lambda\\) values, the consequences on fit here are immediate. All seven regressors also remain in the model throughout cross-validation: the variable selection that LASSO provided has disappeared. Looking to the coefficient paths,

![coefficient_profiles_ridge](/assets/coefficient_profiles_ridge.png){:class="img-responsive" border=0}

we again see a very different result. Where the LASSO coefficient paths had to be computed in a piecewise fashion, the continuous nature of the \\(\ell_2\\)-norm means that we can derive a closed-form solution for the entire ridge regression paths. All variables again begin at zero, but this time all immediately take on some magnitude. As the most dominant variable, ```manhattan``` quickly races away from the pack, while the others increase only gradually. Note however that these other variables ultimately taper off; once again, our shrinkage method is telling us that ```manhatttan``` stands out as the most influential variable in the specified set of regressors, and that the others are not necessarily worth inclusion.

With regard to \\(\lambda\\) selection, we use the same methodology for the sake of consistency. This yields the following coefficients.

```R
> coef(model, s = "lambda.1se")
8 x 1 sparse Matrix of class "dgCMatrix"
                              1
(Intercept)        2.165771e-15
manhattan          7.459888e-01
passenger_count    2.790367e-03
bearing            5.100334e-03
pickup_latitude   -1.290562e-02
pickup_longitude   8.133066e-02
dropoff_latitude  -5.036787e-02
dropoff_longitude  1.390468e-02
```

After unscaling, we discover an even more accurate estimate of \$1.775. This is an overshoot of the correct value (\$1.553) of only 14% – not bad, given we only have one out of the two key variables that in fact determine a taxi fare! The \\(\ell_2\\)-norm penalty has clearly done its job here, shrinking the magnitude of the parameter estimates even more dramatically than the LASSO. Let's see whether the elastic-net can do better.

#### Elastic-net

If the LASSO tends to pick certain regressors and discard others, and ridge regression shrinks the coefficients of correlated regressors towards each other, then the elastic-net is something of a combination. It also introduces the potential for collective shrinkage for regressors correlated in groups.

To achieve this combination, one might initially consider imposing an \\(\ell_p\\)-norm for \\(p \in (1,2)\\). However, the differentiability of \\(| \beta_j |^p\\) in this case means that such a penalty would never set coefficients to zero, and hence the method would lose the ability to perform variable selection. Instead, we sum the  \\(\ell_1\\) and \\(\ell_2\\)-norms, subject to a secondary hyperparameter \\(\alpha\\). This gives the following objective function:
\\[
\min_{\beta_0, \beta} \left\\{ \frac{1}{N} \sum_{i=1}^N ( y_i - \beta_0 - x_i^T \beta )^2) + \lambda \sum_{j=1}^p ( \alpha \beta_j^2 +(1-\alpha)|\beta_j|) \right\\}
\\]
The elastic-net has particular applicability in biostatistics, where hundreds of genes are often used as regressors. These genes are often correlated in groups—"genes tend to operate in molecular pathways" (2)—motivating some kind of compromise between ridge and LASSO. Indeed this was exactly the context in which elastic-net was devised, and it has since seen widespread popularity. So, given that the algorithm manages to both "select variables like the LASSO, and shrinks together the coefficients of correlated predictors like ridge" (3), why would you bother with anything else?

On many occasions, there may be no reason to. Elastic-net can be a robust regularisation technique where prior knowledge is low, and no particular justification for the LASSO or ridge regression is available. However, the NFL (no free lunch) principle tells us that elastic-net is not a magic panacea. Added flexibility brings added complexity, this time in the form of the additional hyperparameter \\(\alpha\\). The \\(\alpha\\) value can be chosen via cross-validation using some kind of grid-search, but in practice is often chosen arbitrarily. Recall too that the entire enterprise of shrinkage was prompted by a desire for model simplicity! As such, it is always wise to preserve a healthy dose of scepticism.

Having said that, let us examine the results of elastic-net regression when applied to our taxi fares prediction problem. The trusty ```glmnet``` package is used again for implementation – in fact we have been using the same elastic-net function all along,

```R
model <- cv.glmnet(x = design_matrix_scaled,
                   y = response_scaled,
                   alpha = 0.5)
```

since it should by now be clear that the LASSO and ridge regression are merely edge cases of the elastic-net, with \\(\alpha = 0\\) and \\(\alpha = 1\\) respectively.  In this case, we pick a middling value of \\(\alpha = 0.5\\). Fitting the ```lm_even_generaliseder``` specification yields the following cross-validation path.

![cross-validation_elastic-net](/assets/cross-validation_elastic-net.png){:class="img-responsive" border=0}

This much more closely resembles the LASSO fit. Comparing the values of \\(\log (\lambda)\\), however, we see that the variable selection occurs less aggressively, taking a longer time to reach just one variable. This is not surprising; the effect of the \\(\ell_1\\)-norm has been attenuated by \\(\alpha > 0\\). Looking to the coefficient paths,

![coefficient_profiles_elastic-net](/assets/coefficient_profiles_elastic-net.png){:class="img-responsive" border=0}

we see the same again: a profile similar to the LASSO, though a tad less dramatic.

To choose \\(\lambda\\), we again take the value corresponding to the most regularised model with error within one standard error of the minimum. This yields the following estimates.

```R
> coef(model, s = "lambda.1se")
8 x 1 sparse Matrix of class "dgCMatrix"
                              1
(Intercept)        7.974242e-16
manhattan          8.244123e-01
passenger_count    .           
bearing            .           
pickup_latitude    .           
pickup_longitude   2.132230e-02
dropoff_latitude  -8.497121e-03
dropoff_longitude  .            
```

After unscaling, we obtain a value for ```manhattan``` of \$1.96, which is still more accurate than the LASSO, but does not beat ridge regression. Evidently the addition of an \\(\ell_2\\)-norm penalty has had an effect, but not to the same degree as a pure ridge regression. Elastic-net is not always the best option!

## Conclusions

The world of big data clearly presents some unique challenges to traditional statistical or econometric methods. Everything is significant – but not everything is valuable. In other words, one might find highly significant \\(p\\)-values, but tiny effects! Shrinkage methods can provide a useful mechanism for separating signal from noise in these circumstances, allowing us to impose structured penalties on complexity and thereby distill our models to only those variables of greatest influence. As we saw in the example explored above, only one of out seven 'statistically significant' variables was found to be of great import, despite all demonstrating ostensibly large effects. Moreover, trading off a little bias for lower variance can be key to robust parametric inference—as was shown by our pursuit of the elusive \$1.553 parameter—and this is exactly the effect of certain shrinkage methods.

Given the plethora of techniques available, when should each be used? In our case, ridge regression was most appropriate, as we were also tackling the added problem of omitted variable bias. Differing contexts nonetheless warrant different approaches, and where appropriate, the elastic-net can offer a more general approach. One added application of shrinkage that wasn't explored in this post is 'Big Data' in the wide sense; this is to say, those circumstances where we have \\(k > > n\\), with a large number of regressors \\(k\\) and a small number of observations \\(n\\). Techniques such as the LASSO can help determine unique solutions to these problems where otherwise there may be infinitely many – though this is perhaps a story for another day!

Shrinkage methods can also be incorporated into econometrics. In fact, I am currently contributing to a paper that looks at the application of exactly these shrinkage methods to a VAR model describing interlinkages in the global economy. Instability of VAR estimates is a common problem, and ridge regression provides a useful technique for mitigating these issues. It is clear that, despite having been around since the 1980s, the properties and applications of shrinkage methods are still very much being discovered!

(1): ```glmnet``` vignette, available at https://web.stanford.edu/~hastie/Papers/Glmnet_Vignette.pdf.

(2): Hastie, Tibshirani and Friedman, The Elements of Statistical Learning (Springer-Verlag, 2009), 662

(3): Ibid., 73
