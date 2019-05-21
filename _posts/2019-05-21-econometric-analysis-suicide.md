---
layout: post
title: "Econometric Analysis of Suicide Rates"
date: 2019-05-21 12:00 +1000
categories: econometrics
excerpt: An exploration of suicide rates and how they vary across demographic cohorts.
---

# An Econometric Analysis of Suicide Rates

Suicide is a tragic phenomenon that afflicts all populations. This blog post seeks to understand which factors most strongly influence its appearance, and how these may have changed over time. Specifically, we will seek to answer three questions: 

1. How do suicide rates vary across key demographic cohorts (i.e. gender, age, generation and country)? 
2. Across these cohorts, has the rate changed over time? 
3. Does material well-being influence the suicide rate? In particular, how does suicide correlate with per capita GDP? 

## The Dataset

To answer these questions, we will be using the *Suicide Rates Overview 1985 to 2016* dataset [from Kaggle](https://www.kaggle.com/russellyates88/suicide-rates-overview-1985-to-2016). This is a union of four different datasets, taken from sources such as the World Bank and the World Health Organisation. It is composed of several basic demographic variables (year, country, age group, generation, sex, total population size), as well two economic indicators (GDP and HDI). Our goal will be to use these variables to model the suicide rate across different cohorts.

```
> glimpse(suicide_data)
Observations: 9,410
Variables: 12
$ country              <chr> "Argentina", "Argentina", "Argentina", "Argent...
$ year                 <int> 1990, 1990, 1990, 1990, 1990, 1990, 1990, 1990...
$ sex                  <chr> "male", "male", "male", "female", "male", "fem...
$ age                  <fct> 75+ years, 55-74 years, 35-54 years, 75+ years...
$ suicides_no          <int> 226, 502, 439, 61, 182, 190, 174, 163, 91, 76,...
$ population           <int> 411100, 2128000, 3619000, 643000, 2297000, 247...
$ `suicides/100k pop`  <dbl> 54.97, 23.59, 12.13, 9.49, 7.92, 7.67, 6.60, 4...
$ `country-year`       <chr> "Argentina1990", "Argentina1990", "Argentina19...
$ `HDI for year`       <dbl> 0.705, 0.705, 0.705, 0.705, 0.705, 0.705, 0.70...
$ `gdp_for_year ($)`   <dbl> 141352368715, 141352368715, 141352368715, 1413...
$ `gdp_per_capita ($)` <int> 4859, 4859, 4859, 4859, 4859, 4859, 4859, 4859...
$ generation           <fct> G.I. Generation, G.I. Generation, Silent, G.I....
```

A cursory examination of the dataset reveals that data integrity is overall quite high, though there are several idiosyncrasies that need to addressed. Outside of the period 1990-2014, many countries are missing from the dataset. We hence omit these years, as they will not allow a reasonable comparison across countries. It is also important to check whether, for each year, each country has data for each age group and gender - if this were not the case, we could be accidentally making inferences about entire populations that might only apply to certain segments of the population. Fortunately we do have a complete set of values. A final check for explicitly missing values reveals that the HDI is only available every fifth year for most of the sample. This is not necessarily a reason to drop the variable, but is certainly worth taking into account if we use it in a model later.

![HDI_availability](/assets/HDI_availability.png){:class="img-responsive" border=0}

All of these idiosyncrasies are likely a product of the fact that the dataset is a union of several different sources. This is a reminder to be wary whenever we are combining multiple datasets!

## Exploratory Analysis

Whenever we are interested in modelling a particular variable, an important first step is understanding precisely how it is distributed. Since the suicide rate is continuous, a histogram is appropriate for this task.

![suicide_rate_hist](/assets/suicide_rate_hist.png){:class="img-responsive" border=0}

As would be expected, the variable is strongly right-skewed. Most cohorts have very low suicide rates, while a few have very high rates. In particular, it is clear that female cohorts dominate at the low end. This fact is true across all age groups...

![suicide_rate_hist_ages](/assets/suicide_rate_hist_ages.png){:class="img-responsive" border=0}

...across generations...

![suicide_rate_hist_gens](/assets/suicide_rate_hist_gens.png){:class="img-responsive" border=0}

...and across countries. 

![suicide_rate_countries](/assets/suicide_rate_countries.png){:class="img-responsive" border=0}

It seems that across our sample, high rates of suicide are an overwhelmingly male phenomenon. Also, the average suicide rate appears to increase with age, seen in the shifting of the distributions to the right. This is true for men and women, though more so for the former. Hearteningly, suicide appears less common across Gen X and the Millennial generations, perhaps suggesting a historical trend downward in suicide rates.

To further explore this trend over time, we next compute how the overall average suicide rate for men and women has evolved across the sample period. I have also included my home country, Australia, as a reference class.

![rate_over_time](/assets/rate_over_time.png){:class="img-responsive" border=0}

This chart validates our earlier hypothesis: it does indeed appear that there is a historical trend downward in suicide rates. Aggregation to this level however can understate changes in particular cohorts. If we focus exclusively on Australia and examine the changes over time in more detail, we see an example of this.

![rate_over_time_ages](/assets/rate_over_time_ages.png){:class="img-responsive" border=0}

Improvement has clearly not been uniform across age groups. Suicide rates for men aged 15-34 and 75+ have fallen dramatically, whereas for ages 35-74 rates have remained more constant. Female suicide rates have remained flat across the board, and for 35-54, they appear to have increased slightly. Aggregation has obscured this nuance; using sweeping statistics such as the average over a large group of countries can make it easy to forget that there may be important trends hidden beneath the surface.

What about changes in GDP per capita and changes in suicide rates? As a preview of the modelling section, we pick three high and low GDP countries and observe how these variables have evolved over time.

![gdp_and_suicide_rate](/assets/gdp_and_suicide_rate.png){:class="img-responsive" border=0}

Results are inconclusive. All six countries have seen increases in per capita GDP, but the responses of suicide rates are inconsistent. The United Status has seen a slight increase in suicide rates for both men and women; Guatemala has seen a dramatic one. Others react more intuitively, with suicide rates declining slightly alongside gains in GDP. Greece is an interesting example: a small downward trend in suicide rates is dramatically interrupted, seemingly as a result of the financial crisis that the nation endured in 2008. To tease out these relationships more fully, we need a model.

## Building a Model

To answer our first two questions, descriptive statistics were appropriate. However, to more accurately hypothesise at the causal relationship between variables, we will need to build a model. This model will use a set of independent variables—for example, sex and per capita GDP—to predict the value of the dependent variable, which in our case is the suicide rate. In doing so, we will be able to better understand which factors most influence suicide rates, and in particular, whether per capita GDP demonstrates a large effect.

The modelling technique we will employ is linear regression. This is perhaps the simplest example of supervised learning, using weighted linear combinations of input variables to predict a response. The weights are estimated using least squares, which seeks to minimise the sum of squared differences between predictions and true observed values. Aside from its simplicity, regression is advantageous in that it permits an intuitive interpretation for the estimated parameters as the marginal effects of each variable. For our purposes, this will be vital, as it will allow us to understand the specific influence of each explanatory variable.

The model specification is as follows:

\\[
\log(\text{suicide rate}) = \beta_0 + \beta_1 \cdot \text{sex} + \beta_2 \cdot \text{age} + \beta_3 \cdot \log(\text{GDP per capita}) \\\ + \beta_4 \cdot \text{generation} + \beta_5 \cdot \log(\text{population}) + \beta_6 \cdot \text{year} \\\ + \beta_7 \cdot \text{sex} \cdot \log(\text{GDP per capita})
\\]

Log transformations are used to combat the skewness of the numeric variables, yielding distributions that are much close to normal. Care is however required in the interpretation of their estimated parameters. The categorical variables (e.g. age, generation) will be converted to dummy variables, and we also include an interaction variable to allow the effect of per capita GDP to differ between men and women.

Fitting this model using the ```lm``` class in R yields the following results.

```R
Call:
lm(formula = log_suicide_rate ~ sex + age + log_per_capita_gdp + 
    generation + log_population + year + sex * log_per_capita_gdp, 
    data = model_data)

Residuals:
    Min      1Q  Median      3Q     Max 
-4.9070 -0.4630  0.0799  0.5550  2.6634 

Coefficients:
                             Estimate Std. Error t value Pr(>|t|)    
(Intercept)                35.8009537  4.1428274   8.642  < 2e-16 ***
sexmale                     1.9432758  0.1323939  14.678  < 2e-16 ***
age25-34 years              0.2794368  0.0342825   8.151 4.07e-16 ***
age35-54 years              0.5458158  0.0538084  10.144  < 2e-16 ***
age55-74 years              0.6505373  0.0839825   7.746 1.05e-14 ***
age75+ years                0.9335020  0.0991988   9.410  < 2e-16 ***
log_per_capita_gdp          0.1393931  0.0104978  13.278  < 2e-16 ***
generationSilent           -0.0732774  0.0453366  -1.616  0.10606    
generationBoomers          -0.0504396  0.0775964  -0.650  0.51569    
generationGeneration X     -0.0211655  0.0996442  -0.212  0.83179    
generationMillenials        0.0006396  0.1213584   0.005  0.99579    
log_population             -0.0252409  0.0087815  -2.874  0.00406 ** 
year                       -0.0177514  0.0021166  -8.387  < 2e-16 ***
sexmale:log_per_capita_gdp -0.0623136  0.0143055  -4.356 1.34e-05 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 0.8591 on 9374 degrees of freedom
Multiple R-squared:  0.4476,	Adjusted R-squared:  0.4469 
F-statistic: 584.3 on 13 and 9374 DF,  p-value: < 2.2e-16
```

![model_results](/assets/model_results.png){:class="img-responsive" border=0}

How can these be interpreted? First, our fit appears quite good. \\(R^2​\\) is not too high, suggesting that only 45% of the variance in suicide rates can be explained by our inputs, but the diagnostic plots indicate a good overall fit: residuals exhibit constant variance and a mean close to zero. Regarding specific parameters: 

- For two otherwise identical cohorts, male suicide rates are on average \\(\exp(1.94) - 1 = 598\%​\\) higher than female suicide rates. This is an enormous difference, but it fits with what we have seen previously.
- Suicide rates appear to increase monotonically with age. This is seen in the increasing magnitude of the parameter estimates for each age greoup.
- There is no statistically significant difference in suicide rates across generations.
- Surprisingly, per capita GDP demonstrates a small positive relationship with suicide rates. This is twice as strong for women as it is for men; an increase of 10% in per capita GDP is associated with a 1.40% increase in suicide rates for women, but only a 0.75% increase for men. 
- Larger populations appear to reduce suicide rates slightly. Specifically, a 10% increase in population size is associated with a 2.5% reduction in suicide rates.
- A highly significant time trend is observed. Ceteris paribus, each year suicide rates were expected to fall by 1.8%. 

It is important to note however that these interpretations are subject to several technical assumptions about the data, which we will not here explore. For a further discussion of the linear regression technique, see [An Introduction to Statistical Learning (James et al)](http://www-bcf.usc.edu/~gareth/ISL/).

## Conclusions

Evidently suicide is a complex and multifaceted phenomenon. Across the large countries that comprised our sample, we saw that male suicide rates almost universally exceeded those for women, and that overall suicide rates have been falling across the past two decades. Nevertheless, we also saw that aggregation can hide the differing trends that may characterise subgroups of the data. For policymakers, this suggests that initiatives to reduce suicide rates may require a different approach for different demographics; given the variation we have observed, it is unlikely that a single panacea is feasible. 

Our model meanwhile underscored some of these observations. Ceteris paribus, suicide rates for men were nearly 600% higher, and a steady overall decline in suicides rates of 1.8% per year was observed. It also suggested an unexpected conclusion: per capita GDP may in fact have a small positive effect on suicide rates, particularly so for women. Given that this result is a little unintuitive, we may wish to validate it by investigating alternative datasets, or selecting an alternate sample of countries. Nevertheless, it is an interesting example of one potential downside to increases in material wealth.