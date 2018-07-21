---
layout: post
title: "Speed Dating and Revealed Preferences"
date: 2018-07-13 12:00 +1000
categories: economics
---

*In this post, the classification technique of logistic regression is explored, alongside a discussion of revealed preferences. This is done using a dataset on speed dating, generated experimentally as part of a paper by two professors at Columbia University.*

# Introduction

A topic near and dear to all single hearts (and some coupled) the world over: what does the opposite sex desire? In this post, we make an attempt to disentangle the deceit, duplicity and downright dishonesty that so fills the romantic realm, while also learning about the concept of revealed preferences and the logistic regression model.

## Some Background

In recent years, classification models have become perhaps the most exciting application of modern statistical learning techniques. It is classification that underpins the most familiar of machine learning technologies (eg. computer vision and voice recognition), many of which draw on highly complex models like neural networks and support vector machines. In these contexts, classification goes by the name of **supervised learning**, though the fundamental problem remains exactly the same: given input data, we want to use some kind of model to predict an output. It is this problem that we will be dipping our toes into today!

Of course, one often underreported difficulty in developing classification models is the requirement of large quantities of labelled training data, which are rarely easy to come by. For example: companies like Uber and Google have racked up millions of kilometres and thousands of hours of driving in order to generate data to train their self-driving car systems. This does not come cheap! Fortunately the dataset we will be using to investigate speed dating has already been collected for us, and is composed of exactly the kind of closed-world experimental results that are ideal for developing a classification model. Moreover, we are looking at a straightforward yes or no response, so a perfect scene is set for the logistic regression tool we will be implementing.

## The Dataset

The data we will be using was gathered by two business school professors from Columbia University (Ray Fisman and Sheena Iyengar) as part of their paper, *Gender Differences in Mate Selection: Evidence From a Speed Dating Experiment*. It details the results of a series of speed dating encounters between men and women, as well as a questionnaire that each participant was asked to fill out regarding their own preferences and characteristics. To be succinct: we have a sample of 8,378 speed dates across 551 individuals, split almost exactly by sex. These participants were drawn from students in graduate and professional schools at Columbia. Each encounter has 122 variables associated, providing details such as the age, race, and employment status of those involved (see [the paper itself]([http://faculty.chicagobooth.edu/emir.kamenica/documents/genderDifferences.pdf) for a more robust description of the speed dating process).

```R
> raw_data
# A tibble: 8,378 x 122
    i_id  w_id gender  g_id condition  wave round_no position position1 order partner_id
   <int> <int>  <int> <int>     <int> <int>    <int>    <int> <chr>     <int>      <int>
 1     1     1      0     1         1     1       10        7 <NA>          4          1
 2     1     1      0     1         1     1       10        7 <NA>          3          2
 3     1     1      0     1         1     1       10        7 <NA>         10          3
 4     1     1      0     1         1     1       10        7 <NA>          5          4
 5     1     1      0     1         1     1       10        7 <NA>          7          5
 6     1     1      0     1         1     1       10        7 <NA>          6          6
 7     1     1      0     1         1     1       10        7 <NA>          1          7
 8     1     1      0     1         1     1       10        7 <NA>          2          8
 9     1     1      0     1         1     1       10        7 <NA>          8          9
10     1     1      0     1         1     1       10        7 <NA>          9         10
# ... with 8,368 more rows, and 111 more variables: p_i_id <int>, match <int>,
#   init_corr <dbl>, samerace <int>, age_o <int>, race_o <int>, pf_o_att <dbl>, ...
```

Given that we will be making predictions as to the romantic inclinations of these 551 individuals, it would be helpful to gain some kind of snapshot of their personalities. To this end, let's take a quick look at their interests. As part of the questionnaire, each individual was asked to rate their interests out of ten across a spectrum of activities - the average scores are given below, split out by gender.

![interests_gender](/assets/interests_bygender.png){:class="img-responsive"}

## A Review of Logistic Regression

The **logistic regression model** we shall be using is perhaps the simplest tool in the statisticians toolbox for tackling the problem of classification. In its simplest form, it offers a model \\(f\\) for the conditional probability of a binary response variable \\(G \in \{0, 1 \}\\), given some matrix of features \\(X\\).
\\[
G = f(X)
\\]
This model is derived by making two small adjustments to the standard linear regression model, familiarly given as
\\[
\mathbb{E}(G|X=x) = \beta_0 + \beta^\intercal x .
\\]
Firstly, the expected value \\(\mathbb{E}(G|X=x)\\) can now be called a probability. Let us denote this as \\(\text{Pr}(G =1|X = x ) = \pi\\). The above equation can therefore be rewritten as
\\[
\pi (x) = \beta_0 + \beta^\intercal x .
\\]
We thus have a simple linear model for the probability of our response variable \\(G\\). However, is this appropriate? If we were to simply estimate this model without any additional constraints, the predicted probabilities would not be bounded. In other words, we may end up with a predicted probability greater than one, or less than zero! To combat this, we apply the logistic transformation, \\(g(x) = ({1-\mathbb{e}^{-x}})^{-1}\\), which bounds all predicted values between zero and one. The final logistic regression model is hence written as
\\[
\pi(x) = \frac{1}{1-\mathbb{e}^{-(\beta_0 + \beta^\intercal x)}} ,
\\]
parameterised by the set \\(\theta = \{ \beta_0 ,  \beta \}\\).

*(As an interesting aside, the logistic curve is also used as a common model of population growth in biology)*

Since we no longer have a linear model, OLS cannot be used to estimate the parameter set \\(\theta\\). Instead, maximum likelihood estimation is employed. The log-likelihood function for \\(N\\) observations is here given by
\\[
\mathcal{l} (\theta) = \sum_{i=1}^N \log (\pi_i (x_i ; \theta)) ,
\\]
where recall that \\(\pi_i(x_i ; \theta ) = \text{Pr}(G = 1| X = x_i ; \theta)\\). This is then solved numerically -- `R` uses iteratively reweighted least squares. The final result is a trained model \\(f\\) that will predict the probability \\(\pi(x)\\) given a particular observation \\(x\\). Using a threshold value \\(\lambda\\), we can then use these probabilities to predict \\(G\\), the actual value of interest.

# Analysis

## Stated Preferences

Perhaps the most obvious place to begin our analysis of speed-dating is by looking at what people *say* they want. This was recorded as part of the questionnaire, with individuals asked to assign 100 points across six dimensions: Ambition, Attractiveness, Fun, Intelligence, Shared Interests and Sincerity. The results are pictured in the box plot below. Unsurprisingly, Attractiveness and Intelligence win the day – and moreover, Attractiveness exhibits a large number of positive outliers, suggesting that for many individuals, being unattractive is an absolute dealbreaker. Tough crowd! Overall though, the distribution is reasonably flat, with Fun and Sincerity not too far behind. Surprisingly, the latter beat out Ambition – perhaps these high powered Columbia students are romantics after all!

![preferences_overall](/assets/preferences_overall.png){:class="img-responsive"}

Looking at a more granular breakdown by gender, we see nothing too surprising. Many of our common stereotypes are confirmed: men claim to care significantly more than women about Attractiveness, whereas  women claim to care more about Ambition and Intelligence. It is nonetheless heartening to note that both sexes place almost exactly the same value on Fun!

![preferences_bygender](/assets/preferences_bygender.png){:class="img-responsive"}

We can also look at what each gender *thinks* the other desires. And in fact, we are broadly quite good at intuiting, with men and women both coming very close to estimating the true preferences articulated above. Both genders do however overestimate the value of Attractiveness!

![preferences_opp_sex](/assets/preferences_opp_sex.png){:class="img-responsive"}

## Revealed Preferences

So this is what people say they want in a partner. However, do they actually follow through and match with people who fit this bill? Or did they just respond to the questionnaire with what they thought sounded good? Of course, this need not even be a conscious deception – just as likely, people don't actually *know* what they want. Self-reported survey results are notoriously noisy for exactly these reasons (see: pollsters predictions of the 2016 US presidential elections). To combat this problem, economists have formulated the idea of **revealed preferences**, which essentially asserts that we shouldn't trust what people *say*, but rather observe what they *do*. Talk is cheap! In the dating context, this means looking at the types of people that an individual actually wants to match with, rather than the types of people who match their stated preferences. And thanks to the contrived nature of the speed dating process, this is exactly the data we have available. If only the entire romantic world could be organised by economists!

### Some Harsh Truths

To understand this idea of revealed preferences, we're going to need a 'true figure' for each individual's attributes. This can be calculated as the average rating that each individual receives across their ten potential partners – which as an aside, also gives us a glimpse into the differing ways men and women evaluate each other. The story here is that men have much more variability: across every dimension, the distribution of actual scores for men is broader.

![scores_distributions_actual](/assets/scores_distributions_actual.png){:class="img-responsive"}

And how do these 'actual' values compare with self-assessments? In other words, how accurately do people evaluate their own value? As we will see, everyone is a bit of an egotist – we all dramatically overestimate ourselves!

![scores_comparisons](/assets/scores_comparisons.png){:class="img-responsive"}

The scatterplots above show average scores compared to self-assessments for each individual, alongside a black \\(45^\circ\\) line (points have been jittered for legibility). If a dot is above the line, then the individual's average rating was higher than their own self-assessment; if is below, then their average rating was below. There are lots of dots below the line! Surprisingly, it is women who are particularly guilty of self-promotion – see the summary table below, which describes the proportion of men and women below the line across each dimension:

| Attribute               | Women           |Men           |
| -------------- | --------- | --------- |
| Ambition       | 0.74 | 0.64 |
| Attractiveness | 0.71 | 0.76 |
| Fun            | 0.81 | 0.78 |
| Intelligence   | 0.82 | 0.78 |
| Sincerity      | 0.82 | 0.71 |

Of course, the equally (if not more) likely reason for this disparity is that men are harsher critics than women. But this is a topic for another day!

### Predicting Matches

Having defined our variables (and deflated our egos), we can now form a model to test this idea of revealed preferences a little more rigourously. A proposition: if people know what they want in a partner, then—holding all else constant—we would expect them to match with people who fit their preferences profile. As such, pairings with a small distance between the preference set and true attributes would be more likely to end in a match, while those with a large distance would not. This implies that the distinction between stated and revealed preferences can be tested through the significance of this distance variable...

Let's start with a simple model, for a baseline. We take the following independent variables:
* Gender
* Pairing order in the evening
* Initial correlation of shared interests
* Whether the two individuals are of the same race
* Age difference
* Actual attributes of the partner

and we try to predict whether the individual wanted to match. Note that this only draws on the actual qualities of the partner, and not the stated preferences of the individual. Nonetheless, we would expect a reasonably significant model, with match probability likely increasing in actual attributes. The more attractive a partner is, the more likely it is that you would want to match with them – regardless of your preferences!

```R
> model <- glm(wants_to_match ~ ., family = binomial(link = 'logit'),
               data = X_train)
>
> summary(model)

Call:
glm(formula = wants_to_match ~ ., family = binomial(link = "logit"),
    data = X_train)

Deviance Residuals:
    Min       1Q   Median       3Q      Max  
-1.8425  -0.9520  -0.5513   1.0500   2.4881  

Coefficients:
                       Estimate Std. Error z value Pr(>|z|)    
(Intercept)           -5.609066   0.374861 -14.963  < 2e-16 ***
i_Gender               0.171867   0.061003   2.817  0.00484 **
order                 -0.003964   0.005019  -0.790  0.42962    
init_corr              0.096448   0.090170   1.070  0.28479    
samerace              -0.016753   0.056074  -0.299  0.76512    
actual_Attractiveness  0.588320   0.035357  16.640  < 2e-16 ***
actual_Sincerity      -0.193844   0.060349  -3.212  0.00132 **
actual_Intelligence    0.198286   0.078116   2.538  0.01114 *  
actual_Fun             0.351416   0.043718   8.038 9.12e-16 ***
actual_Ambition       -0.120263   0.057171  -2.104  0.03542 *  
age_diff               0.005255   0.005817   0.903  0.36634    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 8850.7  on 6517  degrees of freedom
Residual deviance: 7744.4  on 6507  degrees of freedom
AIC: 7766.4

Number of Fisher Scoring iterations: 3
```

Looking at the output, we see that all of the 'actuals' are significant to some degree, as would be expected. Attractiveness and Fun are the most important – and notably, Sincerity and Ambition both have a negative effect upon desire to match! None of pairing order, initial correlation of interests, whether or not the individuals are of the same race, and age difference are significant, suggesting that they do not have an effect upon the desire to match with a partner. Gender is nonetheless significant; since it is men who were encoded as 1, we deduce that men are more likely to want to match with any given partner than women. Again, a confirmation of stereotypes. What of the accuracy of the model?

```R
> results <- predict(model, newdata = X_test)
> predictions <- ifelse(results > 0.5, 1, 0)
> accuracy <- mean(predictions == X_test$wants_to_match)
> print(accuracy)
[1] 0.6388719
```

Using a threshold of \\(\lambda = 0.5\\), we have therefore achieved an accuracy of 64% on a held-out test set. Good, but not great – clearly there is more to love than just 10 variables!

Now let us incorporate stated preferences. Specifically, we incorporate a variable \\(d_i\\) defined as the (scaled) difference between actual attributes and stated preferences. For example, if this is positive, then the partner is more attractive than the preference of the individual, while if it is negative, they are less attractive.
\\[
d_i = \frac{a_i - \overline{a}}{\hat{\sigma}_a} - \frac{s_i - \overline{s}}{\hat{\sigma}_s}
\\]
Here \\(a_i\\) is the 'actual' value of the partner, demeaned and scaled to have unit variance, and \\(s_i\\) is the stated preference of the individual, also demeaned and scaled to have unit variance. As such, if the estimated parameter for \\(d\\) is significant and positive (i.e. someone sitting below your preferences decreases your desire to match with them), then people tend to match with those that fit their preferences. If it is insignificant, then stated preferences do not influence one's desire to match.

```R
> model <- glm(wants_to_match ~ ., family = binomial(link = 'logit'),
               data = X_train)
>
> summary(model)

Call:
glm(formula = wants_to_match ~ ., family = binomial(link = "logit"),
    data = X_train)

Deviance Residuals:
    Min       1Q   Median       3Q      Max  
-1.9251  -0.9819  -0.5941   1.0717   2.6485  

Coefficients:
                   Estimate Std. Error z value Pr(>|z|)    
(Intercept)       -0.618156   0.065910  -9.379  < 2e-16 ***
i_Gender           0.560790   0.055950  10.023  < 2e-16 ***
order             -0.005251   0.004982  -1.054 0.291866    
init_corr         -0.034421   0.090812  -0.379 0.704661    
samerace           0.080994   0.055380   1.463 0.143598    
age_diff           0.003046   0.005698   0.535 0.592947    
d_Attractiveness  0.461465   0.022576  20.440  < 2e-16 ***
d_Sincerity        0.028236   0.020423   1.383 0.166800    
d_Intelligence     0.070365   0.021178   3.322 0.000892 ***
d_Fun              0.249825   0.021796  11.462  < 2e-16 ***
d_Ambition         0.113530   0.022635   5.016 5.28e-07 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 8857.9  on 6492  degrees of freedom
Residual deviance: 7861.4  on 6482  degrees of freedom
AIC: 7883.4

Number of Fisher Scoring iterations: 3
```

Immediately we see that several of the attribute variables have become *more* significant. Given that the model is otherwise identical, this would suggest that incorporating stated preferences has improved our model. Moreover, all estimated coefficients are of an expected sign: our results suggest that across the dimensions of attractiveness, intelligence, fun and ambition, falling short of someone's stated preferences does indeed reduce your chance of matching with them.

Model accuracy has improved a fraction.

```R
> results <- predict(model, newdata = X_test)
> predictions <- ifelse(results > 0.5, 1, 0)
> accuracy <- mean(predictions == X_test$wants_to_match)
> print(accuracy)
[1] 0.6486154
```

# Conclusions

To be written...

<!--
So what is the conclusion from this dismal dating dilemma? Many would argue that economists should focus on revealed preferences over stated preferences -- in other words, they should analyse what people do, and avoid the temptation to theorise about *why* they do it. In this case, it would appear that we do know what we want – though of course, this makes sense, given that romance is something we probably spend a lot of time thinking about.

But this gives hope to the rest of us mortals! No one knows what they want -- at least, no one wants what they think they want.

-->
